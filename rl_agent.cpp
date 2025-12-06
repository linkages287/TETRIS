#include "rl_agent.h"
#include "game_classes.h"
#include <random>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <deque>
#include <dirent.h>
#include <cstring>

// Neural Network Implementation
NeuralNetwork::NeuralNetwork() {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // COMPLETE REWRITE: Clean initialization - He initialization for Leaky ReLU
    double stddev1 = std::sqrt(2.0 / INPUT_SIZE);
    double stddev2 = std::sqrt(2.0 / HIDDEN_SIZE);
    
    std::normal_distribution<double> dist1(0.0, stddev1);
    std::normal_distribution<double> dist2(0.0, stddev2);
    std::normal_distribution<double> bias_dist(0.0, 0.1);
    
    // Initialize weights1 (Input -> Hidden) with He initialization
    weights1.resize(INPUT_SIZE, std::vector<double>(HIDDEN_SIZE));
    for (auto& row : weights1) {
        for (auto& w : row) {
            w = dist1(gen);
        }
    }
    
    // Initialize bias1
    bias1.resize(HIDDEN_SIZE, 0.0);
    for (auto& b : bias1) {
        b = bias_dist(gen);
    }
    
    // Initialize weights2 (Hidden -> Output) with He initialization
    weights2.resize(HIDDEN_SIZE, std::vector<double>(OUTPUT_SIZE));
    for (auto& row : weights2) {
        for (auto& w : row) {
            w = dist2(gen);
        }
    }
    
    // Initialize bias2 with positive value to prevent all Q-values being negative
    // FIX: Initialize bias2 to positive value (3.0) to ensure some positive Q-values initially
    bias2.resize(OUTPUT_SIZE, 0.0);
    std::normal_distribution<double> bias2_dist(3.0, 0.2);  // FIX: Mean 3.0 (increased from 2.0) to ensure positive Q-values
    bias2[0] = bias2_dist(gen);
    // Ensure bias2 is positive and within reasonable range
    bias2[0] = std::max(1.0, std::min(5.0, bias2[0]));
}

double NeuralNetwork::relu(double x) const {
    return std::max(0.0, x);
}

double NeuralNetwork::leaky_relu(double x) const {
    // Leaky ReLU: allows small negative gradients to flow through
    // Prevents "dead neurons" that output 0 for all inputs
    return std::max(0.2 * x, x);  // Leak factor: 0.2 (20% of negative values) - standard value
}

double NeuralNetwork::forward(const std::vector<double>& input) {
    // Hidden layer - use Leaky ReLU to prevent dead neurons
    std::vector<double> hidden(HIDDEN_SIZE);
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        double sum = bias1[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            sum += input[j] * weights1[j][i];
        }
        hidden[i] = leaky_relu(sum);  // Use Leaky ReLU instead of ReLU
    }
    
    // Output layer
    double output = bias2[0];
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        output += hidden[i] * weights2[i][0];
    }
    
    // Clip output Q-value to prevent unbounded growth (new: Q-value clipping)
    const double MAX_Q_VALUE = 200.0;
    const double MIN_Q_VALUE = -200.0;
    output = std::max(MIN_Q_VALUE, std::min(MAX_Q_VALUE, output));
    
    return output;
}

void NeuralNetwork::update(const std::vector<double>& input, double target, double learning_rate) {
    // Forward pass - store intermediate values for backprop
    std::vector<double> hidden_pre_activation(HIDDEN_SIZE);
    std::vector<double> hidden(HIDDEN_SIZE);
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        double sum = bias1[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            sum += input[j] * weights1[j][i];
        }
        hidden_pre_activation[i] = sum;
        hidden[i] = leaky_relu(sum);  // Use Leaky ReLU instead of ReLU
    }
    
    double output = bias2[0];
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        output += hidden[i] * weights2[i][0];
    }
    
    // FIX: Reduced clipping limits to prevent gradient explosion
    const double MAX_ERROR = 25.0;      // Reduced from 50.0 to prevent extreme errors
    const double MAX_GRADIENT = 5.0;    // Reduced from 10.0 to prevent gradient explosion
    // FIX: Reduced weight limits to prevent saturation and weight explosion
    // Current model has bias2/weights2/bias1 hitting limits at 27.0/-27.0
    const double MAX_WEIGHT = 25.0;     // FIX: Reduced from 30.0 to 25.0 (weights still hitting limits)
    const double MIN_WEIGHT = -25.0;    // FIX: Reduced from -30.0 to -25.0 to match MAX_WEIGHT
    
    double error = target - output;
    
    // Clip error to prevent extreme gradients (prevents weight explosion)
    error = std::max(-MAX_ERROR, std::min(MAX_ERROR, error));
    
    double output_gradient = error;
    
    // Update output layer weights and bias
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        if (!std::isfinite(hidden[i])) continue;  // Skip if hidden value is invalid
        
        double weight_gradient = output_gradient * hidden[i];
        
        // Clip gradient to prevent explosion
        weight_gradient = std::max(-MAX_GRADIENT, std::min(MAX_GRADIENT, weight_gradient));
        
        weights2[i][0] += learning_rate * weight_gradient;
        
        // Clip weights to prevent explosion (new: explicit weight clipping)
        weights2[i][0] = std::max(MIN_WEIGHT, std::min(MAX_WEIGHT, weights2[i][0]));
        
        // FIX: More aggressive clipping for weights2 - clip at 80% of limit to prevent saturation
        if (weights2[i][0] > MAX_WEIGHT * 0.8) {
            weights2[i][0] = MAX_WEIGHT * 0.8;  // Clip at 20.0 (80% of 25.0)
        }
        if (weights2[i][0] < MIN_WEIGHT * 0.8) {
            weights2[i][0] = MIN_WEIGHT * 0.8;  // Clip at -20.0 (80% of -25.0)
        }
        
        // Check for NaN/Inf and fix if needed
        if (!std::isfinite(weights2[i][0])) {
            weights2[i][0] = 0.0;  // Reset to zero if invalid
        }
    }
    
    // Clip output gradient for bias update
    double bias2_gradient = std::max(-MAX_GRADIENT, std::min(MAX_GRADIENT, output_gradient));
    bias2[0] += learning_rate * bias2_gradient;
    
    // Clip bias to prevent explosion (new: explicit bias clipping)
    bias2[0] = std::max(MIN_WEIGHT, std::min(MAX_WEIGHT, bias2[0]));
    
    // FIX: More aggressive clipping for bias2 - clip at 80% of limit to prevent saturation
    if (bias2[0] > MAX_WEIGHT * 0.8) {
        bias2[0] = MAX_WEIGHT * 0.8;  // Clip at 20.0 (80% of 25.0)
    }
    if (bias2[0] < MIN_WEIGHT * 0.8) {
        bias2[0] = MIN_WEIGHT * 0.8;  // Clip at -20.0 (80% of -25.0)
    }
    
    if (!std::isfinite(bias2[0])) {
        bias2[0] = 0.0;
    }
    
    // Hidden layer gradients
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        if (!std::isfinite(weights2[i][0])) continue;  // Skip if weight is invalid
        
        double hidden_gradient = output_gradient * weights2[i][0];
        
        // Clip hidden gradient
        hidden_gradient = std::max(-MAX_GRADIENT, std::min(MAX_GRADIENT, hidden_gradient));
        
        double relu_derivative = (hidden_pre_activation[i] > 0) ? 1.0 : 0.2;
        
        // Update input-to-hidden weights
        for (int j = 0; j < INPUT_SIZE; j++) {
            if (!std::isfinite(input[j])) continue;  // Skip if input is invalid
            
            double weight_gradient = hidden_gradient * relu_derivative * input[j];
            
            // Clip weight gradient
            weight_gradient = std::max(-MAX_GRADIENT, std::min(MAX_GRADIENT, weight_gradient));
            
            weights1[j][i] += learning_rate * weight_gradient;
            
            // Clip weights to prevent explosion (new: explicit weight clipping)
            weights1[j][i] = std::max(MIN_WEIGHT, std::min(MAX_WEIGHT, weights1[j][i]));
            
            // Check for NaN/Inf and fix if needed
            if (!std::isfinite(weights1[j][i])) {
                weights1[j][i] = 0.0;  // Reset to zero if invalid
            }
        }
        
        // Update hidden bias
        double bias_gradient = hidden_gradient * relu_derivative;
        bias_gradient = std::max(-MAX_GRADIENT, std::min(MAX_GRADIENT, bias_gradient));
        
        bias1[i] += learning_rate * bias_gradient;
        
        // Clip bias to prevent explosion (new: explicit bias clipping)
        bias1[i] = std::max(MIN_WEIGHT, std::min(MAX_WEIGHT, bias1[i]));
        
        // FIX: Additional aggressive clipping for bias1 - clip at 80% of limit to prevent saturation
        if (bias1[i] > MAX_WEIGHT * 0.8) {  // Clip at 20.0 (80% of 25.0)
            bias1[i] = MAX_WEIGHT * 0.8;
        }
        if (bias1[i] < MIN_WEIGHT * 0.8) {  // Clip at -20.0 (80% of -25.0)
            bias1[i] = MIN_WEIGHT * 0.8;
        }
        
        // Check for NaN/Inf and fix if needed
        if (!std::isfinite(bias1[i])) {
            bias1[i] = 0.0;  // Reset to zero if invalid
        }
    }
    if (!std::isfinite(bias2[0])) {
        bias2[0] = ((rand() / (double)RAND_MAX) - 0.5) * 0.1;
        bias2[0] = std::max(MIN_WEIGHT, std::min(MAX_WEIGHT, bias2[0]));  // Ensure within clipping range
    }
}

void NeuralNetwork::save(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) return;
    
    // Write header with timestamp and filename
    auto now = std::time(nullptr);
    auto time_info = *std::localtime(&now);
    file << "# Neural Network Model File\n";
    file << "# Saved: " << std::put_time(&time_info, "%Y-%m-%d %H:%M:%S") << "\n";
    file << "# Filename: " << filename << "\n";
    file << "#\n";
    
    // Save weights1
    for (const auto& row : weights1) {
        for (double w : row) {
            file << w << " ";
        }
        file << "\n";
    }
    
    // Save bias1
    for (double b : bias1) {
        file << b << " ";
    }
    file << "\n";
    
    // Save weights2
    for (const auto& row : weights2) {
        for (double w : row) {
            file << w << " ";
        }
        file << "\n";
    }
    
    // Save bias2
    for (double b : bias2) {
        file << b << " ";
    }
    file << "\n";
}

bool NeuralNetwork::load(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;
    
    // Skip header lines (lines starting with #)
    // This handles both old format (no header) and new format (with header)
    std::string line;
    while (std::getline(file, line)) {
        // Skip empty lines and comment lines
        if (line.empty() || (line.length() > 0 && line[0] == '#')) {
            continue;
        }
        // Found first data line - need to parse it
        // Use stringstream to parse the line we just read
        std::istringstream iss(line);
        double w;
        // Try to read first weight from this line
        if (iss >> w) {
            // This is a data line, put it back by resetting file position
            // Get current position
            std::streampos pos = file.tellg();
            // Calculate position of start of this line
            pos -= static_cast<std::streamoff>(line.length() + 1); // +1 for newline
            file.seekg(pos);
            break;
        }
    }
    
    // Load weights1
    for (auto& row : weights1) {
        for (double& w : row) {
            if (!(file >> w)) return false;
            // FIX: Clip weights1 to valid range during load to fix corrupted models
            // FIX: Reduced limits to match new MAX_WEIGHT (25.0)
            w = std::max(-25.0, std::min(25.0, w));
            // FIX: More aggressive clipping on load - clip at 80% of limit
            if (w > 25.0 * 0.8) {
                w = 25.0 * 0.8;  // Clip at 20.0 (80% of 25.0)
            }
            if (w < -25.0 * 0.8) {
                w = -25.0 * 0.8;  // Clip at -20.0 (80% of -25.0)
            }
        }
    }
    
    // Load bias1
    for (double& b : bias1) {
        if (!(file >> b)) return false;
        // FIX: Clip bias1 to valid range during load to fix corrupted models
        // FIX: Reduced limits to match new MAX_WEIGHT (25.0)
        b = std::max(-25.0, std::min(25.0, b));
        // FIX: More aggressive clipping on load - clip at 80% of limit
        if (b > 25.0 * 0.8) {
            b = 25.0 * 0.8;  // Clip at 20.0 (80% of 25.0)
        }
        if (b < -25.0 * 0.8) {
            b = -25.0 * 0.8;  // Clip at -20.0 (80% of -25.0)
        }
        // FIX: If bias1 was stuck at old limit, reduce it further
        if (b > 26.0 || b < -26.0) {
            b = b * 0.75;  // Reduce by 25% if it was stuck at old limit
        }
    }
    
    // Load weights2
    for (auto& row : weights2) {
        for (double& w : row) {
            if (!(file >> w)) return false;
            // FIX: Clip weights2 to valid range during load to fix corrupted models
            // FIX: Reduced limits to match new MAX_WEIGHT (25.0)
            w = std::max(-25.0, std::min(25.0, w));
            // FIX: More aggressive clipping on load - clip at 80% of limit and reduce stuck weights
            if (w > 25.0 * 0.8) {
                w = 25.0 * 0.8;  // Clip at 20.0 (80% of 25.0)
            }
            if (w < -25.0 * 0.8) {
                w = -25.0 * 0.8;  // Clip at -20.0 (80% of -25.0)
            }
            // FIX: If weight was stuck at old limit, reduce it further
            if (w > 26.0 || w < -26.0) {
                w = w * 0.75;  // Reduce by 25% if it was stuck at old limit
            }
        }
    }
    
    // Load bias2
    for (double& b : bias2) {
        if (!(file >> b)) return false;
            // FIX: Clip bias2 to valid range during load to fix corrupted models
            // FIX: Reduced limits to match new MAX_WEIGHT (25.0)
            b = std::max(-25.0, std::min(25.0, b));
            // FIX: More aggressive clipping on load - clip at 80% of limit and reduce stuck weights
            if (b > 25.0 * 0.8) {
                b = 25.0 * 0.8;  // Clip at 20.0 (80% of 25.0)
            }
            if (b < -25.0 * 0.8) {
                b = -25.0 * 0.8;  // Clip at -20.0 (80% of -25.0)
            }
            // FIX: If bias2 was stuck at old limit, reduce it further
            if (b > 26.0 || b < -26.0) {
                b = b * 0.75;  // Reduce by 25% if it was stuck at old limit
            }
    }
    
    return true;
}

void NeuralNetwork::logWeightChanges(const std::string& filename, int episode, double error) {
    std::ofstream logfile(filename, std::ios::app);  // Append mode
    if (!logfile.is_open()) return;
    
    // Calculate weight statistics
    double weights1_mean = 0.0, weights1_min = weights1[0][0], weights1_max = weights1[0][0];
    double weights1_std = 0.0;
    int weights1_count = 0;
    
    for (const auto& row : weights1) {
        for (double w : row) {
            weights1_mean += w;
            weights1_min = std::min(weights1_min, w);
            weights1_max = std::max(weights1_max, w);
            weights1_count++;
        }
    }
    weights1_mean /= weights1_count;
    
    for (const auto& row : weights1) {
        for (double w : row) {
            weights1_std += (w - weights1_mean) * (w - weights1_mean);
        }
    }
    weights1_std = std::sqrt(weights1_std / weights1_count);
    
    double weights2_mean = 0.0, weights2_min = weights2[0][0], weights2_max = weights2[0][0];
    double weights2_std = 0.0;
    int weights2_count = 0;
    
    for (const auto& row : weights2) {
        for (double w : row) {
            weights2_mean += w;
            weights2_min = std::min(weights2_min, w);
            weights2_max = std::max(weights2_max, w);
            weights2_count++;
        }
    }
    weights2_mean /= weights2_count;
    
    for (const auto& row : weights2) {
        for (double w : row) {
            weights2_std += (w - weights2_mean) * (w - weights2_mean);
        }
    }
    weights2_std = std::sqrt(weights2_std / weights2_count);
    
    double bias1_mean = 0.0, bias1_min = bias1[0], bias1_max = bias1[0];
    for (double b : bias1) {
        bias1_mean += b;
        bias1_min = std::min(bias1_min, b);
        bias1_max = std::max(bias1_max, b);
    }
    bias1_mean /= bias1.size();
    
    double bias2_mean = bias2[0];
    
    // Write to log file
    logfile << "Episode: " << episode 
            << " | Error: " << error
            << " | W1: mean=" << weights1_mean << " std=" << weights1_std 
            << " min=" << weights1_min << " max=" << weights1_max
            << " | W2: mean=" << weights2_mean << " std=" << weights2_std
            << " min=" << weights2_min << " max=" << weights2_max
            << " | B1: mean=" << bias1_mean << " min=" << bias1_min << " max=" << bias1_max
            << " | B2: " << bias2_mean
            << "\n";
    
    logfile.flush();
}

// Helper function to calculate saturation for a vector of values
static double calcSaturation(const std::vector<double>& values, double& variance) {
    if (values.empty()) {
        variance = 0.0;
        return 0.0;
    }
    
    // COMPLETE REWRITE: Correct variance calculation
    // Calculate mean
    double mean = 0.0;
    int valid_count = 0;
    for (size_t i = 0; i < values.size(); i++) {
        if (std::isfinite(values[i])) {
            mean += values[i];
            valid_count++;
        }
    }
    
    if (valid_count == 0) {
        variance = 0.0;
        return 0.0;
    }
    
    mean /= valid_count;
    
    // Calculate variance (population variance)
    variance = 0.0;
    for (size_t i = 0; i < values.size(); i++) {
        if (std::isfinite(values[i])) {
            double diff = values[i] - mean;
            variance += diff * diff;
        }
    }
    variance /= valid_count;  // Population variance
    
    // FIX: Saturation doesn't make sense for single values (e.g., bias2)
    // Return 0% saturation if there's only one value (not meaningful)
    if (valid_count <= 1) {
        return 0.0;  // Single value = no saturation concept
    }
    
    // Calculate saturation: percentage of values that are "close" to each other
    const double tolerance = 1e-4;  // Tolerance for floating point comparison
    int max_count = 0;
    
    for (size_t i = 0; i < values.size(); i++) {
        if (!std::isfinite(values[i])) continue;
        
        int count = 1;  // Count itself
        for (size_t j = i + 1; j < values.size(); j++) {
            if (std::isfinite(values[j]) && std::abs(values[i] - values[j]) < tolerance) {
                count++;
            }
        }
        if (count > max_count) {
            max_count = count;
        }
    }
    
    // Saturation = percentage of most common value
    return (max_count / static_cast<double>(valid_count)) * 100.0;
}

NeuralNetwork::SaturationMetrics NeuralNetwork::calculateSaturation() const {
    SaturationMetrics metrics;
    
    // Calculate for weights1 (flatten to vector) - COMPLETE REWRITE
    std::vector<double> w1_flat;
    for (const auto& row : weights1) {
        for (double w : row) {
            if (std::isfinite(w)) {  // Only add finite values
                w1_flat.push_back(w);
            }
        }
    }
    metrics.weights1_saturation = calcSaturation(w1_flat, metrics.weights1_variance);
    
    // Calculate for bias1 - COMPLETE REWRITE
    std::vector<double> b1_valid;
    for (double b : bias1) {
        if (std::isfinite(b)) {  // Only add finite values
            b1_valid.push_back(b);
        }
    }
    metrics.bias1_saturation = calcSaturation(b1_valid, metrics.bias1_variance);
    
    // Calculate for weights2 (flatten to vector)
    std::vector<double> w2_flat;
    for (const auto& row : weights2) {
        for (double w : row) {
            if (std::isfinite(w)) {  // Only add finite values
                w2_flat.push_back(w);
            }
        }
    }
    metrics.weights2_saturation = calcSaturation(w2_flat, metrics.weights2_variance);
    
    // Calculate for bias2
    std::vector<double> b2_vec;
    for (double b : bias2) {
        if (std::isfinite(b)) {  // Only add finite values
            b2_vec.push_back(b);
        }
    }
    metrics.bias2_saturation = calcSaturation(b2_vec, metrics.bias2_variance);
    
    // Bias2 variance is already calculated by calcSaturation (will be 0 for single value)
    
    return metrics;
}

std::string NeuralNetwork::getWeightStatsString(int episode, double error, bool is_learning) {
    // COMPLETE REWRITE: Calculate weight statistics (only finite values)
    double weights1_mean = 0.0, weights1_min = 0.0, weights1_max = 0.0;
    double weights1_std = 0.0;
    int weights1_count = 0;
    bool weights1_initialized = false;
    
    for (const auto& row : weights1) {
        for (double w : row) {
            if (std::isfinite(w)) {
                if (!weights1_initialized) {
                    weights1_min = weights1_max = w;
                    weights1_initialized = true;
                }
                weights1_mean += w;
                weights1_min = std::min(weights1_min, w);
                weights1_max = std::max(weights1_max, w);
                weights1_count++;
            }
        }
    }
    if (weights1_count > 0) {
        weights1_mean /= weights1_count;
        for (const auto& row : weights1) {
            for (double w : row) {
                if (std::isfinite(w)) {
                    weights1_std += (w - weights1_mean) * (w - weights1_mean);
                }
            }
        }
        weights1_std = std::sqrt(weights1_std / weights1_count);
    }
    
    double weights2_mean = 0.0, weights2_min = 0.0, weights2_max = 0.0;
    double weights2_std = 0.0;
    int weights2_count = 0;
    bool weights2_initialized = false;
    
    for (const auto& row : weights2) {
        for (double w : row) {
            if (std::isfinite(w)) {
                if (!weights2_initialized) {
                    weights2_min = weights2_max = w;
                    weights2_initialized = true;
                }
                weights2_mean += w;
                weights2_min = std::min(weights2_min, w);
                weights2_max = std::max(weights2_max, w);
                weights2_count++;
            }
        }
    }
    if (weights2_count > 0) {
        weights2_mean /= weights2_count;
        for (const auto& row : weights2) {
            for (double w : row) {
                if (std::isfinite(w)) {
                    weights2_std += (w - weights2_mean) * (w - weights2_mean);
                }
            }
        }
        weights2_std = std::sqrt(weights2_std / weights2_count);
    }
    
    double bias1_mean = 0.0, bias1_min = 0.0, bias1_max = 0.0;
    int bias1_count = 0;
    bool bias1_initialized = false;
    for (double b : bias1) {
        if (std::isfinite(b)) {
            if (!bias1_initialized) {
                bias1_min = bias1_max = b;
                bias1_initialized = true;
            }
            bias1_mean += b;
            bias1_min = std::min(bias1_min, b);
            bias1_max = std::max(bias1_max, b);
            bias1_count++;
        }
    }
    if (bias1_count > 0) {
        bias1_mean /= bias1_count;
    }
    
    // Calculate saturation metrics
    SaturationMetrics sat = calculateSaturation();
    
    // Format with saturation info and learning status on multiple lines
    char buffer[600];
    const char* learning_status = is_learning ? "LEARNING" : "CONVERGED";
    snprintf(buffer, sizeof(buffer), 
        "Ep:%d Err:%.2f | W1: m=%.3f Sat=%.1f%% Var=%.4f | W2: m=%.2f Sat=%.1f%% Var=%.4f\n"
        "B1: Sat=%.1f%% Var=%.4f | B2: Sat=%.1f%% Var=%.4f | Status: %s",
        episode, error, 
        weights1_mean, sat.weights1_saturation, sat.weights1_variance,
        weights2_mean, sat.weights2_saturation, sat.weights2_variance,
        sat.bias1_saturation, sat.bias1_variance,
        sat.bias2_saturation, sat.bias2_variance,
        learning_status);
    
    return std::string(buffer);
}

// RL Agent Implementation
RLAgent::RLAgent(const std::string& model_file) 
    : epsilon(1.0),
    epsilon_min(0.15),        // FIX: Increased from 0.10 to 0.15 for better exploration (prevents premature convergence)
    epsilon_decay(0.9995),    // Slow decay (reaches min in ~9000 games) - allows extensive exploration
    learning_rate(0.001),     // FIX: Reduced from 0.002 to 0.001 (0.003 was too high, causing instability)
    gamma(0.95),              // Standard discount factor (balances immediate and future rewards)
    training_episodes(0),
    total_games(0),
    best_score(0),
    average_score(0.0),
    previous_avg_score(0.0),
    recent_scores_sum(0),
    last_batch_error(0.0),
    model_loaded(false),
    games_since_best_improvement(0),
    convergence_check_interval(0.0),
    last_epsilon(1.0),
    epsilon_change_reason(0.0),
    epsilon_increase_count(0),
    epsilon_decrease_count(0),
    epsilon_at_score_100(-1.0),
    epsilon_at_score_500(-1.0),
    epsilon_at_score_1000(-1.0),
    recent_batch_errors() {
    // Try to load existing model from specified file
    if (q_network.load(model_file)) {
        model_loaded = true;
        
        // Try to load training state metadata
        std::ifstream file(model_file);
        if (file.is_open()) {
            std::string line;
            bool in_metadata = false;
            
            while (std::getline(file, line)) {
                // Check if we've reached the metadata section
                if (line.find("# Training State Metadata") != std::string::npos) {
                    in_metadata = true;
                    continue;
                }
                
                if (in_metadata) {
                    std::istringstream iss(line);
                    std::string key;
                    iss >> key;
                    
                    if (key == "EPSILON") {
                        double loaded_epsilon;
                        iss >> loaded_epsilon;
                        // Store loaded epsilon, will be validated after epsilon_min is loaded
                        epsilon = loaded_epsilon;
                    } else if (key == "EPSILON_MIN") {
                        double loaded_min;
                        iss >> loaded_min;
                        // FIX: Override if loaded epsilon_min is too low (prevents premature convergence)
                        // If loaded min is less than 0.15, force to 0.15 for better exploration
                        if (loaded_min < 0.15) {
                            epsilon_min = 0.15;  // Force higher minimum for convergence
                        } else {
                            epsilon_min = loaded_min;
                        }
                    } else if (key == "EPSILON_DECAY") {
                        double loaded_decay;
                        iss >> loaded_decay;
                        // FIX: Override if loaded decay is too fast (prevents convergence issues)
                        // If loaded decay is faster than 0.999, force to 0.9995 for better exploration
                        if (loaded_decay < 0.999) {
                            epsilon_decay = 0.9995;  // Force slower decay for convergence
                        } else {
                            epsilon_decay = loaded_decay;
                        }
                    } else if (key == "LEARNING_RATE") {
                        double loaded_lr;
                        iss >> loaded_lr;
                        // FIX: Override if loaded learning rate is too high (prevents instability)
                        // If loaded LR is > 0.0015, force to 0.001 for stability
                        if (loaded_lr > 0.0015) {
                            learning_rate = 0.001;  // Force lower learning rate for stability
                        } else {
                            learning_rate = loaded_lr;
                        }
                    } else if (key == "GAMMA") {
                        iss >> gamma;
                    } else if (key == "TRAINING_EPISODES") {
                        iss >> training_episodes;
                    } else if (key == "TOTAL_GAMES") {
                        iss >> total_games;
                    } else if (key == "BEST_SCORE") {
                        iss >> best_score;
                    } else if (key == "AVERAGE_SCORE") {
                        iss >> average_score;
                    } else if (key == "PREVIOUS_AVG_SCORE") {
                        iss >> previous_avg_score;
                    }
                }
            }
        }
        
        // FIX: Reset epsilon if performance is poor OR if epsilon is below minimum (convergence issue)
        // If average score is low and epsilon is at minimum, reset to allow exploration
        // Also reset if epsilon is below the new epsilon_min (after override)
        if (epsilon < epsilon_min) {
            // Epsilon below minimum - reset to allow exploration
            epsilon = std::max(epsilon_min, std::min(0.5, epsilon_min * 3.0));  // Reset to 3x minimum or 0.5
            std::ofstream logfile("debug.log", std::ios::app);
            if (logfile.is_open()) {
                logfile << "[MODEL LOAD] Epsilon reset: " << epsilon
                        << " | Reason: Below minimum (" << epsilon_min << ")" << std::endl;
            }
        } else if (average_score > 0 && average_score < 400.0 && epsilon <= epsilon_min * 1.1 && total_games > 100) {
            // Poor performance and epsilon at minimum - reset to allow exploration
            epsilon = std::min(0.5, epsilon_min * 3.0);  // Reset to 3x minimum for exploration
            std::ofstream logfile("debug.log", std::ios::app);
            if (logfile.is_open()) {
                logfile << "[MODEL LOAD] Epsilon reset due to poor performance: " << epsilon
                        << " | Avg Score: " << average_score << " | Games: " << total_games << std::endl;
            }
        } else if (average_score >= 400.0 && epsilon <= epsilon_min * 1.1 && total_games > 50) {
            // Good performance but epsilon at minimum - reset to allow continued exploration for improvement
            epsilon = std::min(0.3, epsilon_min * 2.0);  // Reset to 2x minimum (0.30) for continued exploration
            std::ofstream logfile("debug.log", std::ios::app);
            if (logfile.is_open()) {
                logfile << "[MODEL LOAD] Epsilon reset for continued exploration: " << epsilon
                        << " | Avg Score: " << average_score << " | Games: " << total_games << std::endl;
            }
        }
        
        // If no metadata found, use default epsilon for continued training
        if (training_episodes == 0 && total_games == 0) {
            epsilon = 0.3;  // Start at 30% exploration for continued learning
        }
        
        // Log successful model load (will be written to debug.log)
        std::ofstream logfile("debug.log", std::ios::app);
        if (logfile.is_open()) {
            logfile << "[MODEL] Successfully loaded " << model_file << std::endl;
            logfile << "  Epsilon: " << epsilon << ", Episodes: " << training_episodes 
                    << ", Games: " << total_games << ", Best Score: " << best_score << std::endl;
        }
    } else {
        model_loaded = false;
        // Log that no model was found (will be written to debug.log)
        std::ofstream logfile("debug.log", std::ios::app);
        if (logfile.is_open()) {
            logfile << "[MODEL] No existing model found (" << model_file << ") - Starting fresh training" << std::endl;
        }
    }
}

std::vector<double> RLAgent::extractState(const TetrisGame& game) {
    // ZERO-BASED REDESIGN: Minimal essential features only (27 total)
    std::vector<double> state(NeuralNetwork::INPUT_SIZE, 0.0);
    int idx = 0;
    
    if (game.current_piece == nullptr) {
        return state;
    }
    
    // 1. Column Heights (10 features) - Essential spatial information
    int max_height = 0;
    for (int x = 0; x < game.WIDTH; x++) {
        int height = game.getColumnHeight(x, game.board);
        max_height = std::max(max_height, height);
        state[idx++] = height / 20.0;  // Simple normalization [0, 1]
    }
    
    // 2. Board Quality (3 features) - How bad is the board?
    state[idx++] = max_height / 20.0;  // Max height [0, 1]
    // FIX: Normalize holes properly - max possible holes = 10 columns × 20 rows = 200
    int total_holes = game.countHoles(game.board);
    state[idx++] = std::min(1.0, total_holes / 200.0);  // Total holes [0, 1]
    // FIX: Normalize bumpiness properly - max possible = 9 gaps × 20 height diff = 180
    int total_bumpiness = game.calculateBumpiness(game.board);
    state[idx++] = std::min(1.0, total_bumpiness / 180.0);  // Total bumpiness [0, 1]
    
    // 3. Current Piece (7 features) - One-hot encoding
    for (int i = 0; i < 7; i++) {
        state[idx++] = (game.current_piece->type == i) ? 1.0 : 0.0;
    }
    
    // 4. Next Piece (7 features) - One-hot encoding
    if (game.next_piece) {
        for (int i = 0; i < 7; i++) {
            state[idx++] = (game.next_piece->type == i) ? 1.0 : 0.0;
        }
    } else {
        for (int i = 0; i < 7; i++) {
            state[idx++] = 0.0;
        }
    }
    
    // Total: 10 + 3 + 7 + 7 = 27 features
    // Removed lines/level - not needed for move selection
    
    return state;
}

std::vector<double> RLAgent::extractStateFromBoard(const std::vector<std::vector<int>>& sim_board, 
                                                    int /*lines_cleared*/, int /*level*/, 
                                                    const TetrisPiece* next_piece) const {
    // ZERO-BASED REDESIGN: Minimal essential features only (27 total)
    std::vector<double> state(NeuralNetwork::INPUT_SIZE, 0.0);
    int idx = 0;
    const int WIDTH = TetrisGame::WIDTH;
    const int HEIGHT = TetrisGame::HEIGHT;
    
    // 1. Column Heights (10 features) - Essential spatial information
    std::vector<int> column_heights(WIDTH);
    int max_height = 0;
    
    for (int x = 0; x < WIDTH; x++) {
        int height = 0;
        for (int y = 0; y < HEIGHT; y++) {
            if (sim_board[y][x] != 0) {
                height = HEIGHT - y;
                break;
            }
        }
        column_heights[x] = height;
        max_height = std::max(max_height, height);
        state[idx++] = height / 20.0;  // Simple normalization [0, 1]
    }
    
    // 2. Board Quality (3 features) - How bad is the board?
    state[idx++] = max_height / 20.0;  // Max height [0, 1]
    
    // Calculate total holes
    int total_holes = 0;
    for (int x = 0; x < WIDTH; x++) {
        bool block_found = false;
        for (int y = 0; y < HEIGHT; y++) {
            if (sim_board[y][x] != 0) {
                block_found = true;
            } else if (block_found) {
                total_holes++;
            }
        }
    }
    // FIX: Normalize holes properly - max possible holes = 10 columns × 20 rows = 200
    state[idx++] = std::min(1.0, total_holes / 200.0);  // Total holes [0, 1]
    
    // Calculate total bumpiness
    int total_bumpiness = 0;
    for (int x = 0; x < WIDTH - 1; x++) {
        total_bumpiness += std::abs(column_heights[x] - column_heights[x + 1]);
    }
    // FIX: Normalize bumpiness properly - max possible = 9 gaps × 20 height diff = 180
    state[idx++] = std::min(1.0, total_bumpiness / 180.0);  // Total bumpiness [0, 1]
    
    // 3. Current Piece (7 features) - None since piece is placed
    for (int i = 0; i < 7; i++) {
        state[idx++] = 0.0;
    }
    
    // 4. Next Piece (7 features) - One-hot encoding
    if (next_piece) {
        for (int i = 0; i < 7; i++) {
            state[idx++] = (next_piece->type == i) ? 1.0 : 0.0;
        }
    } else {
        for (int i = 0; i < 7; i++) {
            state[idx++] = 0.0;
        }
    }
    
    // Total: 10 + 3 + 7 + 7 = 27 features
    // Removed lines/level - not needed for move selection
    
    return state;
}

RLAgent::Move RLAgent::findBestMove(const TetrisGame& game, bool training) {
    if (game.current_piece == nullptr) {
        return {0, 0, -999999};
    }
    
    Move best_move = {0, 0, -999999};
    TetrisPiece piece = *game.current_piece;
    
    // Epsilon-greedy: explore or exploit
    bool explore = training && (rand() / (double)RAND_MAX) < epsilon;
    
    if (explore) {
        // Random exploration
        int rot = rand() % 4;
        int x = rand() % (game.WIDTH + 1) - 2;
        return {rot, x, 0.0};
    }
    
    // Exploit: find best Q-value with optimizations
    int move_evaluations = 0;
    const int MAX_EVALUATIONS = 300;  // Safety limit to prevent CPU spinning
    const double EARLY_TERMINATION_THRESHOLD = 50.0;  // Reduced from 100.0 - stop if we find a good move (less aggressive)
    
    // Pre-calculate next piece encoding (used in all state extractions)
    const TetrisPiece* next_piece = game.next_piece;
    const int total_lines_cleared = game.lines_cleared;
    const int current_level = game.level;
    
    // Generate move order: try center positions first (more likely to be good)
    // Create ordered list of x positions: center outward, alternating left/right
    // FIX: Alternate left/right to prevent bias towards one side
    std::vector<int> x_positions;
    int center = game.WIDTH / 2;
    x_positions.push_back(center);  // Center first
    for (int offset = 1; offset <= game.WIDTH + 2; offset++) {
        // Alternate left/right to prevent bias
        // Try right first on odd offsets, left first on even offsets
        // This ensures both sides are checked equally at each distance
        if (offset % 2 == 1) {
            // Odd offset: try right first (balance left bias)
            if (center + offset < game.WIDTH + 2) x_positions.push_back(center + offset);
            if (center - offset >= -2) x_positions.push_back(center - offset);
        } else {
            // Even offset: try left first (balance right bias)
            if (center - offset >= -2) x_positions.push_back(center - offset);
            if (center + offset < game.WIDTH + 2) x_positions.push_back(center + offset);
        }
    }
    
    // Try rotations in order (0, 1, 2, 3) but could prioritize common ones
    for (int rot = 0; rot < 4 && move_evaluations < MAX_EVALUATIONS; rot++) {
        piece.rotation = rot;
        
        // Calculate piece bounds for this rotation to optimize position filtering
        std::vector<Point> blocks = piece.getBlocks();
        int min_x = 0, max_x = 0;
        if (!blocks.empty()) {
            min_x = blocks[0].x;
            max_x = blocks[0].x;
            for (const auto& block : blocks) {
                min_x = std::min(min_x, block.x);
                max_x = std::max(max_x, block.x);
            }
        }
        
        // Try positions in order (center outward)
        for (size_t pos_idx = 0; pos_idx < x_positions.size() && move_evaluations < MAX_EVALUATIONS; pos_idx++) {
            int x = x_positions[pos_idx];
            
            // Skip positions where piece would be completely off-board (optimized bounds check)
            if (x + min_x < -2 || x + max_x >= game.WIDTH + 2) {
                continue;
            }
            
            piece.x = x;
            move_evaluations++;
            piece.y = 0;
            
            // Early collision check
            if (game.checkCollision(piece)) continue;
            
            // Simulate drop (optimized with better bounds)
            int drop_y = 0;
            int max_drop = game.HEIGHT + 10;  // Safety limit
            while (!game.checkCollision(piece, 0, drop_y + 1) && drop_y < max_drop) {
                drop_y++;
            }
            if (drop_y >= max_drop) {
                // Safety: skip this move if drop calculation stuck
                continue;
            }
            
            // Create next state
            std::vector<std::vector<int>> sim_board = game.simulatePlacePiece(piece, piece.y + drop_y);
            int lines_cleared = game.simulateClearLines(sim_board);
            
            // Relaxed heuristic filter: only skip moves that create excessive holes
            // Let network learn hole avoidance naturally, but filter obviously terrible moves
            int holes = game.countHoles(sim_board);
            if (holes > 25 && total_lines_cleared < 30) {
                // Only skip moves that create excessive holes (>25) very early game (<30 lines)
                // This allows network to learn hole-avoidance strategies while filtering extreme cases
                continue;
            }
            
            // Extract state using optimized helper function
            std::vector<double> next_state = extractStateFromBoard(
                sim_board, total_lines_cleared + lines_cleared, current_level, next_piece);
            
            // Get Q-value from network
            double q_value = q_network.forward(next_state);
            
            // Clip Q-value to prevent unbounded growth (new: Q-value clipping)
            const double MAX_Q_VALUE_EVAL = 200.0;
            const double MIN_Q_VALUE_EVAL = -200.0;
            q_value = std::max(MIN_Q_VALUE_EVAL, std::min(MAX_Q_VALUE_EVAL, q_value));
            
            // Track best move
            if (q_value > best_move.q_value) {
                best_move.rotation = rot;
                best_move.x = x;
                best_move.q_value = q_value;
                
                // Early termination: if we find a very good move, stop searching
                // FIX: Increased minimum evaluations to ensure both sides are checked
                // This prevents bias towards one side due to early termination
                if (q_value > EARLY_TERMINATION_THRESHOLD && move_evaluations > 20) {
                    // Found a very good move and evaluated at least 20 moves
                    // This ensures we've checked both left and right sides before terminating
                    break;
                }
            }
        }
        
        // Early termination check: if we found a very good move, stop trying other rotations
        // FIX: Increased minimum evaluations to ensure both sides are checked across rotations
        if (best_move.q_value > EARLY_TERMINATION_THRESHOLD && move_evaluations > 40) {
            // Found a very good move and evaluated at least 40 moves
            // This ensures we've checked both sides across multiple rotations before terminating
            break;
        }
    }
    
    return best_move;
}

void RLAgent::addExperience(const Experience& exp) {
    replay_buffer.push_back(exp);
    if (replay_buffer.size() > BUFFER_SIZE) {
        // Remove oldest experience (FIFO)
        // This ensures we keep recent experiences while maintaining diversity
        replay_buffer.pop_front();
    }
    
    // Prevent buffer from being dominated by bad experiences
    // If we have too many game-over experiences, occasionally remove some
    if (replay_buffer.size() > BUFFER_SIZE / 2) {
        int game_over_count = 0;
        for (const auto& e : replay_buffer) {
            if (e.done) game_over_count++;
        }
        // If more than 30% are game-over experiences, remove some old ones
        if (game_over_count > replay_buffer.size() * 0.3) {
            // Remove oldest game-over experiences to maintain balance
            auto it = replay_buffer.begin();
            int removed = 0;
            int target_removal = game_over_count / 4;
            while (it != replay_buffer.end() && removed < target_removal) {
                if (it->done) {
                    it = replay_buffer.erase(it);
                    removed++;
                } else {
                    ++it;
                }
            }
        }
    }
}

void RLAgent::train() {
    if (replay_buffer.size() < BATCH_SIZE) return;
    
    // Constants matching NeuralNetwork::update() - must match exactly
    const double MAX_ERROR = 25.0;  // Same as in update() function (reduced from 50.0)
    const double MAX_Q_VALUE = 200.0;  // Maximum Q-value (new: prevent unbounded Q-values)
    const double MIN_Q_VALUE = -200.0; // Minimum Q-value (new: prevent unbounded Q-values)
    
    // SIMPLIFIED: Uniform random sampling (standard experience replay)
    std::vector<Experience> batch;
    for (int i = 0; i < BATCH_SIZE; i++) {
        int idx = rand() % replay_buffer.size();
        batch.push_back(replay_buffer[idx]);
    }
    
    // IMPROVED: Better error tracking and normalization
    double batch_avg_error = 0.0;
    double batch_max_error = 0.0;
    double batch_min_error = std::numeric_limits<double>::max();
    double batch_error_sum_sq = 0.0;  // For standard deviation
    int valid_updates = 0;
    int total_samples = 0;
    int clipped_errors = 0;  // Count how many errors were clipped
    
    // Track target and predicted ranges for debugging
    double min_target = std::numeric_limits<double>::max();
    double max_target = std::numeric_limits<double>::lowest();
    double min_predicted = std::numeric_limits<double>::max();
    double max_predicted = std::numeric_limits<double>::lowest();
    
    for (const auto& exp : batch) {
        total_samples++;
        
        // COMPLETE REWRITE: Clean Q-learning update with Q-value clipping
        // Q-learning target: r + gamma * max Q(s', a')
        double target = exp.reward;
        if (!exp.done) {
            double next_q = q_network.forward(exp.next_state);
            // Clip Q-value to prevent unbounded growth (new: Q-value clipping)
            next_q = std::max(MIN_Q_VALUE, std::min(MAX_Q_VALUE, next_q));
            target += gamma * next_q;
        }
        
        // Clip target Q-value to prevent extreme targets (new: target clipping)
        target = std::max(MIN_Q_VALUE, std::min(MAX_Q_VALUE, target));
        
        // Get current Q-value prediction
        double predicted = q_network.forward(exp.state);
        
        // Track ranges
        if (std::isfinite(target)) {
            min_target = std::min(min_target, target);
            max_target = std::max(max_target, target);
        }
        if (std::isfinite(predicted)) {
            min_predicted = std::min(min_predicted, predicted);
            max_predicted = std::max(max_predicted, predicted);
        }
        
        // Calculate raw error (before clipping)
        double raw_error = target - predicted;
        double abs_error = std::abs(raw_error);
        
        // Check if error would be clipped (for statistics)
        if (abs_error > MAX_ERROR) {
            clipped_errors++;
        }
        
        // Clip error for statistics (same as in update function)
        double clipped_error = std::max(-MAX_ERROR, std::min(MAX_ERROR, raw_error));
        double abs_clipped_error = std::abs(clipped_error);
        
        // Use clipped error for statistics (matches what's actually used in backprop)
        batch_avg_error += abs_clipped_error;
        batch_max_error = std::max(batch_max_error, abs_clipped_error);
        batch_min_error = std::min(batch_min_error, abs_clipped_error);
        batch_error_sum_sq += abs_clipped_error * abs_clipped_error;
        
        // Update network if values are finite
        if (std::isfinite(target) && std::isfinite(predicted)) {
            q_network.update(exp.state, target, learning_rate);
            valid_updates++;
        }
    }
    
    // Calculate statistics
    if (total_samples > 0) {
        batch_avg_error /= total_samples;
        
        // Calculate standard deviation
        double error_variance = (batch_error_sum_sq / total_samples) - (batch_avg_error * batch_avg_error);
        double error_std = std::sqrt(std::max(0.0, error_variance));
        
        // Log error statistics periodically (every 100 batches)
        if (training_episodes % 100 == 0) {
            std::ofstream logfile("debug.log", std::ios::app);
            if (logfile.is_open()) {
                logfile << "[ERROR_STATS] Ep: " << training_episodes
                        << " | Avg: " << batch_avg_error
                        << " | Min: " << batch_min_error
                        << " | Max: " << batch_max_error
                        << " | Std: " << error_std
                        << " | Clipped: " << clipped_errors << "/" << total_samples
                        << " | Target Range: [" << min_target << ", " << max_target << "]"
                        << " | Predicted Range: [" << min_predicted << ", " << max_predicted << "]"
                        << std::endl;
            }
        }
    }
    
    // Check for NaN/Inf in error (prevent crashes)
    if (!std::isfinite(batch_avg_error)) {
        batch_avg_error = 0.0;  // Reset if invalid
    }
    
    // Store error for display (clipped error, matches what's actually used)
    last_batch_error = batch_avg_error;
    
    // Track recent errors for learning detection
    recent_batch_errors.push_back(batch_avg_error);
    if (recent_batch_errors.size() > LEARNING_WINDOW) {
        recent_batch_errors.pop_front();
    }
    
    // Weight changes are now displayed on screen only (no log file)
    
    training_episodes++;
}

void RLAgent::updateEpsilonBasedOnPerformance() {
    // COMPLETE REWRITE: Adaptive epsilon decay based on score performance
    // Monitor score vs epsilon relationship to prevent premature exploitation
    
    // Track epsilon-score relationship
    if (total_games > 0 && average_score > 0) {
        epsilon_score_history.push_back({(int)average_score, epsilon});
        if (epsilon_score_history.size() > EPSILON_SCORE_HISTORY_SIZE) {
            epsilon_score_history.pop_front();
        }
        
        // Track epsilon at score milestones
        if (epsilon_at_score_100 < 0 && average_score >= 100) {
            epsilon_at_score_100 = epsilon;
        }
        if (epsilon_at_score_500 < 0 && average_score >= 500) {
            epsilon_at_score_500 = epsilon;
        }
        if (epsilon_at_score_1000 < 0 && average_score >= 1000) {
            epsilon_at_score_1000 = epsilon;
        }
    }
    
    // FIX: Adaptive epsilon decay - slower when score is low or not improving
    if (epsilon > epsilon_min && total_games > 0) {
        // Base decay rate
        double decay_rate = epsilon_decay;
        
        // Calculate improvement for decay rate adjustment
        double improvement = average_score - previous_avg_score;
        double improvement_percent = 0.0;
        if (previous_avg_score > 1.0) {
            improvement_percent = (improvement / previous_avg_score) * 100.0;
        }
        
        // Calculate recent trend
        double recent_trend = 0.0;
        if (recent_scores.size() >= 20) {
            int n = std::min(20, (int)recent_scores.size());
            double recent_sum = 0.0, older_sum = 0.0;
            for (int i = recent_scores.size() - n; i < (int)recent_scores.size(); i++) {
                recent_sum += recent_scores[i];
            }
            if (recent_scores.size() >= 40) {
                for (int i = recent_scores.size() - 40; i < (int)recent_scores.size() - 20; i++) {
                    older_sum += recent_scores[i];
                }
                recent_trend = (recent_sum - older_sum) / n;
            }
        }
        bool positive_trend = recent_trend > 2.0;
        
        // FIX: Slow down decay more aggressively when score is low or not improving
        if (average_score < 100.0) {
            // Very low score: decay 20x slower (0.999975 instead of 0.9995)
            decay_rate = 0.999975;
        } else if (average_score < 200.0) {
            // Low score: decay 10x slower (0.99995 instead of 0.9995)
            decay_rate = 0.99995;
        } else if (average_score < 500.0) {
            // Moderate score: decay 5x slower (0.9999 instead of 0.9995)
            decay_rate = 0.9999;
        } else if (average_score < 1000.0) {
            // Good score but not great: decay 2x slower (0.99975 instead of 0.9995)
            decay_rate = 0.99975;
        }
        // High score (>= 1000): use normal decay rate
        
        // FIX: Additional check - if score is not improving, slow down decay even more
        if (improvement_percent < 0.5 && !positive_trend && total_games > 50) {
            // Score not improving: slow down decay by another 2x
            decay_rate = std::min(0.99999, decay_rate * 0.9995);  // Make decay even slower
        }
        
        epsilon *= decay_rate;
        if (epsilon < epsilon_min) {
            epsilon = epsilon_min;
        }
        
        // Log epsilon-score relationship periodically
        if (total_games % 50 == 0) {
            std::ofstream logfile("debug.log", std::ios::app);
            if (logfile.is_open()) {
                logfile << "[EPSILON-SCORE] Games: " << total_games 
                        << " | Avg Score: " << average_score
                        << " | Epsilon: " << epsilon
                        << " | Decay Rate: " << decay_rate
                        << " | Eps@100: " << (epsilon_at_score_100 >= 0 ? std::to_string(epsilon_at_score_100) : "N/A")
                        << " | Eps@500: " << (epsilon_at_score_500 >= 0 ? std::to_string(epsilon_at_score_500) : "N/A")
                        << " | Eps@1000: " << (epsilon_at_score_1000 >= 0 ? std::to_string(epsilon_at_score_1000) : "N/A")
                        << std::endl;
            }
        }
    }
    
    // Only do performance-based updates if we have enough games for meaningful comparison
    if (total_games < 10) {
        // Too early, keep epsilon high for exploration
        // Initialize previous_avg_score when we have enough data
        if (total_games == 10 && average_score > 0.0) {
            previous_avg_score = average_score;
        }
        return;
    }
    
    // Initialize previous_avg_score if not set
    if (previous_avg_score == 0.0 && average_score > 0.0) {
        previous_avg_score = average_score;
        return;
    }
    
    // Use recent score trend instead of just comparing to previous average
    // This is more responsive to actual learning progress
    double recent_trend = 0.0;
    if (recent_scores.size() >= 20) {
        // Calculate trend over last 20 games
        int n = std::min(20, (int)recent_scores.size());
        double recent_sum = 0.0, older_sum = 0.0;
        for (int i = recent_scores.size() - n; i < (int)recent_scores.size(); i++) {
            recent_sum += recent_scores[i];
        }
        if (recent_scores.size() >= 40) {
            // Compare to previous 20 games
            for (int i = recent_scores.size() - 40; i < (int)recent_scores.size() - 20; i++) {
                older_sum += recent_scores[i];
            }
            recent_trend = (recent_sum - older_sum) / n;  // Average improvement per game
        }
    }
    
    // Calculate improvement (use both average and trend)
    double improvement = average_score - previous_avg_score;
    double improvement_percent = 0.0;
    if (previous_avg_score > 1.0) {  // Need meaningful baseline
        improvement_percent = (improvement / previous_avg_score) * 100.0;
    }
    
    // FIX: Make positive trend detection more strict - require significant improvement
    // Also check if recent trend is positive (learning happening)
    // Changed threshold from 0.5 to 2.0 to require more substantial improvement
    bool positive_trend = recent_trend > 2.0;  // Average score improving by >2.0 per game (more strict)
    
    // Track epsilon before change
    double epsilon_before = epsilon;
    std::string change_reason = "";
    
    // FIX: Only decrease epsilon when there's REAL improvement, not just tiny changes
    // Update epsilon based on performance (more responsive thresholds)
    // Use both improvement_percent and recent_trend for better decisions
    // Require BOTH significant improvement (>2%) AND positive trend for faster decay
    if ((improvement_percent > 2.0 && positive_trend) || improvement_percent > 5.0) {
        // Significant improvement (>2% with trend OR >5% overall): Faster decay (network is learning well)
        if (epsilon > epsilon_min) {
            epsilon *= 0.99;  // Faster decay when learning well (1% per update)
            if (epsilon < epsilon_min) {
                epsilon = epsilon_min;
            }
            change_reason = positive_trend ? "Significant positive trend" : "Major score improvement";
            epsilon_decrease_count++;
        }
    } else if (improvement_percent > 1.0 && positive_trend) {
        // Moderate improvement (>1% with trend): Normal decay (already applied above)
        change_reason = "Moderate improvement";
    } else if (improvement_percent < -1.0 && !positive_trend) {
        // Performance degrading (>1% worse): Increase epsilon (need more exploration)
        // BUT: Don't increase too aggressively to prevent oscillation
        // Only increase if score is actually low, not just slightly worse
        if (average_score < 300.0 || (average_score < previous_avg_score * 0.9)) {
            // Significant degradation or low score - increase exploration
            epsilon = std::min(1.0, epsilon * 1.10);  // Reduced from 1.15 to 1.10 (less aggressive)
            // If epsilon was at minimum, jump to a moderate value (not too high)
            if (epsilon <= epsilon_min * 1.2) {
                epsilon = std::min(1.0, epsilon_min * 2.5);  // Reduced from 4.0 to 2.5 (0.125 instead of 0.20)
            }
            change_reason = "Score degrading";
            epsilon_increase_count++;
        } else {
            // Small degradation but still good performance - don't increase epsilon
            // This prevents oscillation when performance is still good
            change_reason = "Minor degradation, maintaining";
        }
    } else {
        // Small improvement or no change: Check if score is low and not improving
        // If average score is very low (< 200) and not improving (no positive trend), increase exploration
        if (average_score < 200.0 && improvement_percent < 0.5 && !positive_trend) {
            // Low score and not improving significantly - need more exploration
            double old_epsilon = epsilon;
            epsilon = std::min(1.0, epsilon * 1.05);  // Small increase (5%)
            // If epsilon is already at minimum, increase it
            if (epsilon <= epsilon_min * 1.1) {
                epsilon = std::min(1.0, epsilon_min * 3.0);  // Jump to 3x minimum (0.15) for exploration
            }
            if (epsilon > old_epsilon) {
                change_reason = "Low score, not improving";
                epsilon_increase_count++;
            }
        } else if (average_score < 100.0) {
            // Very low score - definitely need more exploration
            double old_epsilon = epsilon;
            epsilon = std::min(1.0, epsilon * 1.08);  // Reduced from 1.1 to 1.08 (less aggressive)
            if (epsilon <= epsilon_min * 1.2) {
                epsilon = std::min(1.0, epsilon_min * 3.0);  // Reduced from 5.0 to 3.0 (0.15 instead of 0.25)
            }
            if (epsilon > old_epsilon) {
                change_reason = "Very low score";
                epsilon_increase_count++;
            }
        } else if (average_score > 500.0 && epsilon > epsilon_min * 1.5) {
            // High performance achieved - prevent epsilon from increasing too much
            // If we're performing well, don't let epsilon get too high even if there's a small dip
            if (improvement_percent > -0.5) {
                // Small dip but still good - don't increase epsilon
                change_reason = "High performance, maintaining epsilon";
            }
        } else if (epsilon > epsilon_before) {
            // Baseline decay already applied, but if it increased, count it
            epsilon_decrease_count++;
        }
        // Otherwise, use normal baseline decay (already applied above)
    }
    
    // Log epsilon changes for verification
    if (std::abs(epsilon - epsilon_before) > 0.001) {  // Significant change
        epsilon_change_reason = improvement_percent;
        std::ofstream logfile("debug.log", std::ios::app);
        if (logfile.is_open()) {
            logfile << "[EPSILON] " << epsilon_before << " -> " << epsilon 
                    << " | Score: " << average_score << " | Improvement: " 
                    << improvement_percent << "% | Reason: " << change_reason << std::endl;
        }
    }
    
    last_epsilon = epsilon_before;
    
    // Safety check: Ensure epsilon never goes below epsilon_min
    if (epsilon < epsilon_min) {
        epsilon = epsilon_min;
    }
    
    // Update previous average for next comparison (update more frequently for responsiveness)
    // Update every 3 games or on significant change to be more responsive
    if (std::abs(improvement_percent) > 1.0 || total_games % 3 == 0) {
        previous_avg_score = average_score;
    }
    
    // FIX: Convergence detection and epsilon reset for poor performance
    // If average is very low and not improving, force more exploration
    if (average_score < 100.0 && total_games > 50 && !positive_trend && improvement_percent < 0.0) {
        // Very low score, many games played, no improvement - force more exploration
        if (epsilon < epsilon_min * 2.0) {
            epsilon = std::min(1.0, epsilon_min * 2.5);  // Increase exploration
            change_reason = "Stuck at low score, forcing exploration";
            epsilon_increase_count++;
            
            // Log this for debugging
            std::ofstream logfile("debug.log", std::ios::app);
            if (logfile.is_open()) {
                logfile << "[EPSILON] Forced increase: " << epsilon_before << " -> " << epsilon
                        << " | Avg: " << average_score << " | Trend: " << recent_trend
                        << " | Games: " << total_games << std::endl;
            }
        }
    }
    
    // FIX: Convergence detection - if performance is poor and epsilon is at minimum, reset epsilon
    // This prevents getting stuck in poor strategies with no exploration
    if (epsilon <= epsilon_min * 1.1 && average_score < 400.0 && total_games > 100) {
        // Check if score has been stuck for a while
        bool stuck = true;
        if (recent_scores.size() >= 50) {
            // Check if last 50 games show improvement
            double recent_avg = 0.0;
            double older_avg = 0.0;
            for (int i = recent_scores.size() - 25; i < (int)recent_scores.size(); i++) {
                recent_avg += recent_scores[i];
            }
            for (int i = recent_scores.size() - 50; i < (int)recent_scores.size() - 25; i++) {
                older_avg += recent_scores[i];
            }
            recent_avg /= 25.0;
            older_avg /= 25.0;
            
            // If recent average is better, not stuck
            if (recent_avg > older_avg * 1.05) {  // 5% improvement
                stuck = false;
            }
        }
        
        if (stuck) {
            // Reset epsilon to allow more exploration
            epsilon = std::min(0.5, epsilon_min * 3.0);  // Reset to 3x minimum (0.45)
            change_reason = "Convergence reset: poor performance, increasing exploration";
            epsilon_increase_count++;
            
            std::ofstream logfile("debug.log", std::ios::app);
            if (logfile.is_open()) {
                logfile << "[CONVERGENCE] Epsilon reset: " << epsilon_before << " -> " << epsilon
                        << " | Avg: " << average_score << " | Games: " << total_games
                        << " | Reason: Poor performance, stuck in local minimum" << std::endl;
            }
        }
    }
}

bool RLAgent::isStillLearning() const {
    // Need at least some error history to determine learning status
    if (recent_batch_errors.size() < 20) {
        // Not enough data yet - assume still learning if we have any training
        return training_episodes > 0;
    }
    
    // Check if error is decreasing (learning) or stable (converged)
    // Compare recent errors (last 25%) vs older errors (first 25%)
    int window_size = recent_batch_errors.size();
    int quarter = window_size / 4;
    
    if (quarter < 5) {
        // Not enough data for comparison
        return training_episodes > 0;
    }
    
    // Calculate average of oldest quarter (baseline)
    double old_avg = 0.0;
    for (int i = 0; i < quarter; i++) {
        old_avg += recent_batch_errors[i];
    }
    old_avg /= quarter;
    
    // Calculate average of newest quarter (recent)
    double new_avg = 0.0;
    for (int i = window_size - quarter; i < window_size; i++) {
        new_avg += recent_batch_errors[i];
    }
    new_avg /= quarter;
    
    // Calculate error variance (high variance = still learning/changing)
    double mean_error = 0.0;
    for (double err : recent_batch_errors) {
        mean_error += err;
    }
    mean_error /= recent_batch_errors.size();
    
    double variance = 0.0;
    for (double err : recent_batch_errors) {
        double diff = err - mean_error;
        variance += diff * diff;
    }
    variance /= recent_batch_errors.size();
    
    // Network is still learning if:
    // 1. Error is decreasing significantly (>5% improvement)
    // 2. Error variance is high (weights are changing)
    // 3. Current error is still substantial (>0.1)
    bool error_decreasing = false;
    if (old_avg > 0.01) {
        double improvement = (old_avg - new_avg) / old_avg;
        error_decreasing = improvement > 0.05;  // 5% improvement
    }
    
    bool high_variance = variance > 0.1;  // Significant variance indicates changes
    bool substantial_error = mean_error > 0.1;  // Still has room to improve
    
    // Learning if error is decreasing OR (high variance AND substantial error)
    return error_decreasing || (high_variance && substantial_error);
}

bool RLAgent::checkConvergence() {
    // Need at least CONVERGENCE_WINDOW games to check convergence
    if (total_games < CONVERGENCE_WINDOW) {
        return false;
    }
    
    // Check 1: Average score stability
    // Calculate coefficient of variation over recent games
    if (recent_scores.size() < CONVERGENCE_STABILITY_THRESHOLD) {
        return false;
    }
    
    // Calculate mean and standard deviation of recent scores
    double sum = 0.0;
    double sum_sq = 0.0;
    int count = 0;
    
    for (int score : recent_scores) {
        sum += score;
        sum_sq += score * score;
        count++;
    }
    
    if (count < CONVERGENCE_STABILITY_THRESHOLD) {
        return false;
    }
    
    double mean = sum / count;
    double variance = (sum_sq / count) - (mean * mean);
    double std_dev = std::sqrt(variance);
    double coefficient_of_variation = (mean > 0.1) ? (std_dev / mean) : 1.0;
    
    // Check 2: Epsilon at minimum
    bool epsilon_at_min = (epsilon <= epsilon_min + 0.01);
    
    // Check 3: Error stability
    bool error_stable = (last_batch_error < 2.0);
    
    // Check 4: Best score plateau
    bool best_score_plateau = (games_since_best_improvement >= 1000);
    
    // Check 5: No significant upward trend
    // Calculate trend over recent scores
    double trend = 0.0;
    if (recent_scores.size() >= 100) {
        // Simple linear regression slope
        double x_sum = 0.0, y_sum = 0.0, xy_sum = 0.0, x2_sum = 0.0;
        int n = std::min(100, (int)recent_scores.size());
        for (int i = 0; i < n; i++) {
            double x = i;
            double y = recent_scores[recent_scores.size() - n + i];
            x_sum += x;
            y_sum += y;
            xy_sum += x * y;
            x2_sum += x * x;
        }
        double denominator = (n * x2_sum - x_sum * x_sum);
        if (std::abs(denominator) > 0.0001) {
            trend = (n * xy_sum - x_sum * y_sum) / denominator;
        }
    }
    
    // Normalize trend by mean score
    double normalized_trend = (mean > 0.1) ? (trend / mean) : 0.0;
    
    // Convergence criteria (all must be true):
    bool score_stable = (coefficient_of_variation < CONVERGENCE_VARIATION_THRESHOLD);
    bool no_upward_trend = (normalized_trend < 0.01);  // Less than 1% improvement per game
    
    bool converged = score_stable && epsilon_at_min && error_stable && 
                     (best_score_plateau || no_upward_trend);
    
    if (converged) {
        std::cout << "\n=== CONVERGENCE DETECTED ===" << std::endl;
        std::cout << "Games: " << total_games << std::endl;
        std::cout << "Episodes: " << training_episodes << std::endl;
        std::cout << "Average Score: " << average_score << " (CV: " << coefficient_of_variation << ")" << std::endl;
        std::cout << "Best Score: " << best_score << " (unchanged for " << games_since_best_improvement << " games)" << std::endl;
        std::cout << "Epsilon: " << epsilon << " (at minimum: " << epsilon_min << ")" << std::endl;
        std::cout << "Error: " << last_batch_error << std::endl;
        std::cout << "Trend: " << normalized_trend * 100 << "% per game" << std::endl;
        std::cout << "===========================" << std::endl;
    }
    
    return converged;
}

void RLAgent::saveModel() {
    saveModelToFile("tetris_model.txt");
}

void RLAgent::saveModelToFile(const std::string& filename) {
    // Save network weights first (includes timestamp and filename header)
    q_network.save(filename);
    
    // Append training state metadata to the model file
    std::ofstream file(filename, std::ios::app);
    if (file.is_open()) {
        // Add separator and metadata header
        file << "\n# Training State Metadata\n";
        file << "FILENAME " << filename << "\n";
        file << "EPSILON " << epsilon << "\n";
        file << "EPSILON_MIN " << epsilon_min << "\n";
        file << "EPSILON_DECAY " << epsilon_decay << "\n";
        file << "LEARNING_RATE " << learning_rate << "\n";
        file << "GAMMA " << gamma << "\n";
        file << "TRAINING_EPISODES " << training_episodes << "\n";
        file << "TOTAL_GAMES " << total_games << "\n";
        file << "BEST_SCORE " << best_score << "\n";
        file << "AVERAGE_SCORE " << average_score << "\n";
        file << "PREVIOUS_AVG_SCORE " << previous_avg_score << "\n";
    }
}

int RLAgent::readBestScoreFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return -1;  // File doesn't exist
    }
    
    std::string line;
    bool in_metadata = false;
    
    while (std::getline(file, line)) {
        // Check if we've reached the metadata section
        if (line.find("# Training State Metadata") != std::string::npos) {
            in_metadata = true;
            continue;
        }
        
        if (in_metadata) {
            std::istringstream iss(line);
            std::string key;
            iss >> key;
            
            if (key == "BEST_SCORE") {
                int score;
                iss >> score;
                return score;
            }
        }
    }
    
    return -1;  // BEST_SCORE not found
}

void RLAgent::saveBestModelIfBetter(int current_score) {
    // Find all existing best model files by searching directory
    // Pattern: tetris_model_best*.txt
    std::vector<std::string> best_model_files;
    std::vector<int> best_model_scores;
    
    // Search current directory for best model files
    DIR* dir = opendir(".");
    if (dir != nullptr) {
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            std::string filename = entry->d_name;
            // Check if filename starts with "tetris_model_best" and ends with ".txt"
            if (filename.find("tetris_model_best") == 0 && filename.find(".txt") == filename.length() - 4) {
                int score = readBestScoreFromFile(filename);
                if (score >= 0) {
                    best_model_files.push_back(filename);
                    best_model_scores.push_back(score);
                }
            }
        }
        closedir(dir);
    }
    
    // Find the best score among existing files
    int best_existing_score = -1;
    std::string best_existing_file = "";
    
    for (size_t i = 0; i < best_model_files.size(); i++) {
        if (best_model_scores[i] > best_existing_score) {
            best_existing_score = best_model_scores[i];
            best_existing_file = best_model_files[i];
        }
    }
    
    // Only save if current score is better than existing best
    if (current_score <= best_existing_score) {
        // Current score is not better, don't save
        std::ofstream logfile("debug.log", std::ios::app);
        if (logfile.is_open()) {
            logfile << "[BEST] Score " << current_score << " not better than existing best " 
                    << best_existing_score << " (" << best_existing_file << ") - skipping save" << std::endl;
        }
        return;
    }
    
    // Generate filename with timestamp and score
    auto now = std::time(nullptr);
    auto time_info = *std::localtime(&now);
    
    char timestamp[32];
    std::strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", &time_info);
    
    std::ostringstream filename;
    filename << "tetris_model_best_" << timestamp << "_score" << current_score << ".txt";
    
    std::string best_model_file = filename.str();
    
    // Save the model
    saveModelToFile(best_model_file);
    
    // Log the save
    std::ofstream logfile("debug.log", std::ios::app);
    if (logfile.is_open()) {
        logfile << "[BEST] New best score: " << current_score;
        if (best_existing_score >= 0) {
            logfile << " (previous best: " << best_existing_score << " from " << best_existing_file << ")";
        }
        logfile << " | Saved to " << best_model_file << std::endl;
    }
    
    std::cout << "[BEST] New best score: " << current_score;
    if (best_existing_score >= 0) {
        std::cout << " (previous best: " << best_existing_score << " from " << best_existing_file << ")";
    }
    std::cout << " | Saved to " << best_model_file << std::endl;
}

void RLAgent::saveBestModelWithDate() {
    // Save best model with date and max score (called on program start/exit)
    // Filename format: tetris_model_best_YYYYMMDD_scoreXXXXX.txt
    
    // Get current date
    auto now = std::time(nullptr);
    auto time_info = *std::localtime(&now);
    
    char date[32];
    std::strftime(date, sizeof(date), "%Y%m%d", &time_info);
    
    // Generate filename with date and best score
    std::ostringstream filename;
    filename << "tetris_model_best_" << date << "_score" << best_score << ".txt";
    
    std::string best_model_file = filename.str();
    
    // Save the model
    saveModelToFile(best_model_file);
    
    // Log the save
    std::ofstream logfile("debug.log", std::ios::app);
    if (logfile.is_open()) {
        logfile << "[BEST] Saved best model on program start/exit: " << best_model_file 
                << " | Best Score: " << best_score << std::endl;
    }
    
    std::cout << "[BEST] Saved best model: " << best_model_file 
              << " | Best Score: " << best_score << std::endl;
}

