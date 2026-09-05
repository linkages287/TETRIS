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
#include <limits>

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
    
    const double MAX_ERROR = 25.0;
    const double MAX_GRADIENT = 5.0;
    const double MAX_WEIGHT = 10.0;
    const double MIN_WEIGHT = -10.0;
    
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
        weights2[i][0] = std::max(MIN_WEIGHT, std::min(MAX_WEIGHT, weights2[i][0]));
        if (!std::isfinite(weights2[i][0])) {
            weights2[i][0] = 0.0;
        }
    }
    
    // Clip output gradient for bias update
    double bias2_gradient = std::max(-MAX_GRADIENT, std::min(MAX_GRADIENT, output_gradient));
    bias2[0] += learning_rate * bias2_gradient;
    bias2[0] = std::max(MIN_WEIGHT, std::min(MAX_WEIGHT, bias2[0]));
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
            weights1[j][i] = std::max(MIN_WEIGHT, std::min(MAX_WEIGHT, weights1[j][i]));
            if (!std::isfinite(weights1[j][i])) {
                weights1[j][i] = 0.0;
            }
        }
        
        // Update hidden bias
        double bias_gradient = hidden_gradient * relu_derivative;
        bias_gradient = std::max(-MAX_GRADIENT, std::min(MAX_GRADIENT, bias_gradient));
        
        bias1[i] += learning_rate * bias_gradient;
        bias1[i] = std::max(MIN_WEIGHT, std::min(MAX_WEIGHT, bias1[i]));
        if (!std::isfinite(bias1[i])) {
            bias1[i] = 0.0;
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
            if (!std::isfinite(w)) w = 0.0;
            w = std::max(-10.0, std::min(10.0, w));
        }
    }
    
    // Load bias1
    for (double& b : bias1) {
        if (!(file >> b)) return false;
        if (!std::isfinite(b)) b = 0.0;
        b = std::max(-10.0, std::min(10.0, b));
    }
    
    // Load weights2
    for (auto& row : weights2) {
        for (double& w : row) {
            if (!(file >> w)) return false;
            if (!std::isfinite(w)) w = 0.0;
            w = std::max(-10.0, std::min(10.0, w));
        }
    }
    
    // Load bias2
    for (double& b : bias2) {
        if (!(file >> b)) return false;
        if (!std::isfinite(b)) b = 0.0;
        b = std::max(-10.0, std::min(10.0, b));
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
    epsilon_min(0.08),
    epsilon_decay(0.997),
    learning_rate(0.001),
    gamma(0.95),
    heuristic_weight(6.0),
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
                        iss >> epsilon;
                    } else if (key == "EPSILON_MIN") {
                        iss >> epsilon_min;
                    } else if (key == "EPSILON_DECAY") {
                        iss >> epsilon_decay;
                    } else if (key == "LEARNING_RATE") {
                        iss >> learning_rate;
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
        
        if (epsilon < epsilon_min) {
            epsilon = epsilon_min;
        }
        if (training_episodes == 0 && total_games == 0) {
            epsilon = 0.25;
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
    // Live after-state: current board + the piece that will be placed next.
    return extractStateFromBoard(game.board, game.lines_cleared, game.level, game.current_piece);
}

std::vector<double> RLAgent::extractStateFromBoard(const std::vector<std::vector<int>>& sim_board, 
                                                    int /*lines_cleared*/, int /*level*/, 
                                                    const TetrisPiece* upcoming_piece) const {
    // After-state: board quality + the upcoming piece in the current-piece slots.
    // The last 7 features stay zero so evaluation and training use the same layout.
    std::vector<double> state(NeuralNetwork::INPUT_SIZE, 0.0);
    int idx = 0;
    const int WIDTH = TetrisGame::WIDTH;
    const int HEIGHT = TetrisGame::HEIGHT;
    
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
        state[idx++] = height / 20.0;
    }
    
    state[idx++] = max_height / 20.0;
    
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
    state[idx++] = std::min(1.0, total_holes / 200.0);
    
    int total_bumpiness = 0;
    for (int x = 0; x < WIDTH - 1; x++) {
        total_bumpiness += std::abs(column_heights[x] - column_heights[x + 1]);
    }
    state[idx++] = std::min(1.0, total_bumpiness / 180.0);
    
    for (int i = 0; i < 7; i++) {
        state[idx++] = (upcoming_piece && upcoming_piece->type == i) ? 1.0 : 0.0;
    }
    for (int i = 0; i < 7; i++) {
        state[idx++] = 0.0;
    }
    
    return state;
}

double RLAgent::boardHeuristic(const std::vector<std::vector<int>>& sim_board,
                               int lines_cleared, int landing_height) const {
    const int WIDTH = TetrisGame::WIDTH;
    const int HEIGHT = TetrisGame::HEIGHT;
    
    int holes = 0;
    int aggregate_height = 0;
    int bumpiness = 0;
    int wells = 0;
    int row_trans = 0;
    int col_trans = 0;
    std::vector<int> heights(WIDTH, 0);
    
    for (int x = 0; x < WIDTH; x++) {
        bool block_found = false;
        int height = 0;
        for (int y = 0; y < HEIGHT; y++) {
            if (sim_board[y][x] != 0) {
                if (!block_found) {
                    height = HEIGHT - y;
                    block_found = true;
                }
            } else if (block_found) {
                holes++;
            }
        }
        heights[x] = height;
        aggregate_height += height;
    }
    
    for (int x = 0; x < WIDTH - 1; x++) {
        bumpiness += std::abs(heights[x] - heights[x + 1]);
    }
    
    for (int y = 0; y < HEIGHT; y++) {
        int prev = 1;
        for (int x = 0; x < WIDTH; x++) {
            int cell = (sim_board[y][x] != 0) ? 1 : 0;
            if (cell != prev) row_trans++;
            prev = cell;
        }
        if (prev != 1) row_trans++;
    }
    
    for (int x = 0; x < WIDTH; x++) {
        int prev = 0;
        for (int y = 0; y < HEIGHT; y++) {
            int cell = (sim_board[y][x] != 0) ? 1 : 0;
            if (cell != prev) col_trans++;
            prev = cell;
        }
        if (prev != 1) col_trans++;
    }
    
    for (int x = 0; x < WIDTH; x++) {
        int depth = 0;
        for (int y = 0; y < HEIGHT; y++) {
            if (sim_board[y][x] == 0) {
                bool left_wall = (x == 0) || (sim_board[y][x - 1] != 0);
                bool right_wall = (x == WIDTH - 1) || (sim_board[y][x + 1] != 0);
                if (left_wall && right_wall) {
                    depth++;
                    wells += depth;
                } else {
                    depth = 0;
                }
            } else {
                depth = 0;
            }
        }
    }
    
    // El-Tetris-inspired weights: prefer line clears, punish holes/transitions/wells.
    return lines_cleared * 3.4
         - holes * 7.9
         - wells * 3.4
         - row_trans * 0.32
         - col_trans * 0.90
         - aggregate_height * 0.51
         - landing_height * 0.45
         - bumpiness * 0.18;
}

RLAgent::Move RLAgent::findBestMove(const TetrisGame& game, bool training) {
    if (game.current_piece == nullptr) {
        return {0, 0, -1e9};
    }
    
    struct Placement {
        int rotation;
        int x;
        int drop_y;
        int lines;
        std::vector<std::vector<int>> board;
        double heuristic;
        double value;
    };
    
    std::vector<Placement> placements;
    TetrisPiece piece = *game.current_piece;
    
    for (int rot = 0; rot < 4; rot++) {
        piece.rotation = rot;
        piece.x = 0;
        piece.y = 0;
        std::vector<Point> blocks = piece.getBlocks();
        if (blocks.empty()) continue;
        
        int min_dx = blocks[0].x;
        int max_dx = blocks[0].x;
        for (const auto& block : blocks) {
            min_dx = std::min(min_dx, block.x);
            max_dx = std::max(max_dx, block.x);
        }
        
        for (int x = -min_dx; x <= game.WIDTH - 1 - max_dx; x++) {
            piece.x = x;
            piece.y = 0;
            if (game.checkCollision(piece)) continue;
            
            int drop_y = 0;
            while (!game.checkCollision(piece, 0, drop_y + 1) && drop_y < game.HEIGHT + 4) {
                drop_y++;
            }
            if (drop_y >= game.HEIGHT + 4) continue;
            
            Placement p;
            p.rotation = rot;
            p.x = x;
            p.drop_y = drop_y;
            p.board = game.simulatePlacePiece(piece, piece.y + drop_y);
            p.lines = game.simulateClearLines(p.board);
            int landing_height = game.HEIGHT - drop_y;
            p.heuristic = boardHeuristic(p.board, p.lines, landing_height);
            p.value = 0.0;
            placements.push_back(std::move(p));
        }
    }
    
    if (placements.empty()) {
        return {game.current_piece->rotation, game.current_piece->x, -1e9};
    }
    
    bool explore = training && ((rand() / (double)RAND_MAX) < epsilon);
    if (explore) {
        // Mostly noisy-heuristic exploration so random play is not pure garbage.
        if ((rand() / (double)RAND_MAX) < 0.25) {
            const Placement& p = placements[rand() % placements.size()];
            return {p.rotation, p.x, p.heuristic};
        }
        Move best = {placements[0].rotation, placements[0].x, -1e9};
        for (const auto& p : placements) {
            double noisy = p.heuristic + ((rand() / (double)RAND_MAX) - 0.5) * 8.0;
            if (noisy > best.q_value) {
                best.rotation = p.rotation;
                best.x = p.x;
                best.q_value = noisy;
            }
        }
        return best;
    }
    
    Move best_move = {placements[0].rotation, placements[0].x, -1e9};
    for (auto& p : placements) {
        std::vector<double> after_state = extractStateFromBoard(
            p.board, 0, 0, game.next_piece);
        double q_value = q_network.forward(after_state);
        q_value = std::max(-200.0, std::min(200.0, q_value));
        p.value = heuristic_weight * p.heuristic + q_value;
        if (p.value > best_move.q_value) {
            best_move.rotation = p.rotation;
            best_move.x = p.x;
            best_move.q_value = p.value;
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
    last_epsilon = epsilon;
    
    if (total_games > 0 && average_score > 0) {
        epsilon_score_history.push_back({(int)average_score, epsilon});
        if (epsilon_score_history.size() > EPSILON_SCORE_HISTORY_SIZE) {
            epsilon_score_history.pop_front();
        }
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
    
    if (epsilon > epsilon_min) {
        epsilon *= epsilon_decay;
        if (epsilon < epsilon_min) {
            epsilon = epsilon_min;
        }
        epsilon_decrease_count++;
    }
    
    previous_avg_score = average_score;
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

