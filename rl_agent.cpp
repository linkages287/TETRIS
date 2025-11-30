#include "rl_agent.h"
#include "game_classes.h"
#include <random>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <sstream>

// Neural Network Implementation
NeuralNetwork::NeuralNetwork() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 0.1);
    
    // Initialize weights with small random values
    weights1.resize(INPUT_SIZE, std::vector<double>(HIDDEN_SIZE));
    for (auto& row : weights1) {
        for (auto& w : row) {
            w = dist(gen);
        }
    }
    
    bias1.resize(HIDDEN_SIZE, 0.0);
    for (auto& b : bias1) {
        b = dist(gen);
    }
    
    weights2.resize(HIDDEN_SIZE, std::vector<double>(OUTPUT_SIZE));
    for (auto& row : weights2) {
        for (auto& w : row) {
            w = dist(gen);
        }
    }
    
    bias2.resize(OUTPUT_SIZE, 0.0);
    for (auto& b : bias2) {
        b = dist(gen);
    }
}

double NeuralNetwork::relu(double x) const {
    return std::max(0.0, x);
}

double NeuralNetwork::forward(const std::vector<double>& input) {
    // Hidden layer
    std::vector<double> hidden(HIDDEN_SIZE);
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        double sum = bias1[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            sum += input[j] * weights1[j][i];
        }
        hidden[i] = relu(sum);
    }
    
    // Output layer
    double output = bias2[0];
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        output += hidden[i] * weights2[i][0];
    }
    
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
        hidden[i] = relu(sum);
    }
    
    double output = bias2[0];
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        output += hidden[i] * weights2[i][0];
    }
    
    // Backward pass - proper backpropagation
    double error = target - output;
    
    // Clip error to prevent extreme gradients (less aggressive clipping)
    error = std::max(-50.0, std::min(50.0, error));
    
    // Output layer gradient (dE/doutput = -error, but we use error for gradient descent)
    double output_gradient = error;  // Don't multiply by learning_rate here
    
    // Update output layer weights and bias
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        double weight_gradient = output_gradient * hidden[i];
        weights2[i][0] += learning_rate * weight_gradient;
        // Clip weights to prevent explosion
        weights2[i][0] = std::max(-50.0, std::min(50.0, weights2[i][0]));
    }
    bias2[0] += learning_rate * output_gradient;
    bias2[0] = std::max(-50.0, std::min(50.0, bias2[0]));
    
    // Hidden layer gradients
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        // Gradient from output layer: dE/dhidden = output_gradient * weight2
        double hidden_gradient = output_gradient * weights2[i][0];
        
        // ReLU derivative: 1 if hidden_pre_activation > 0, else 0
        if (hidden_pre_activation[i] > 0) {
            // Update input-to-hidden weights
            for (int j = 0; j < INPUT_SIZE; j++) {
                double weight_gradient = hidden_gradient * input[j];
                weights1[j][i] += learning_rate * weight_gradient;
                // Clip weights to prevent explosion
                weights1[j][i] = std::max(-10.0, std::min(10.0, weights1[j][i]));
            }
            // Update hidden bias
            bias1[i] += learning_rate * hidden_gradient;
            bias1[i] = std::max(-10.0, std::min(10.0, bias1[i]));
        }
    }
    
    // Check for NaN or Inf values (prevent crashes)
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        if (!std::isfinite(weights2[i][0]) || !std::isfinite(bias1[i])) {
            // Reset to small random values if NaN/Inf detected
            weights2[i][0] = ((rand() / (double)RAND_MAX) - 0.5) * 0.1;
            bias1[i] = ((rand() / (double)RAND_MAX) - 0.5) * 0.1;
        }
        for (int j = 0; j < INPUT_SIZE; j++) {
            if (!std::isfinite(weights1[j][i])) {
                weights1[j][i] = ((rand() / (double)RAND_MAX) - 0.5) * 0.1;
            }
        }
    }
    if (!std::isfinite(bias2[0])) {
        bias2[0] = ((rand() / (double)RAND_MAX) - 0.5) * 0.1;
    }
}

void NeuralNetwork::save(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) return;
    
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
    
    // Load weights1
    for (auto& row : weights1) {
        for (double& w : row) {
            if (!(file >> w)) return false;
        }
    }
    
    // Load bias1
    for (double& b : bias1) {
        if (!(file >> b)) return false;
    }
    
    // Load weights2
    for (auto& row : weights2) {
        for (double& w : row) {
            if (!(file >> w)) return false;
        }
    }
    
    // Load bias2
    for (double& b : bias2) {
        if (!(file >> b)) return false;
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

std::string NeuralNetwork::getWeightStatsString(int episode, double error) {
    // Calculate weight statistics (same as logWeightChanges but return as string)
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
    
    // Format as compact single line
    char buffer[300];
    snprintf(buffer, sizeof(buffer), 
        "Ep:%d Err:%.2f W1:m=%.3f W2:m=%.2f B2=%.2f",
        episode, error, weights1_mean, weights2_mean, bias2_mean);
    
    return std::string(buffer);
}

// RL Agent Implementation
RLAgent::RLAgent(const std::string& model_file) 
    : epsilon(1.0),
    epsilon_min(0.05),        // Minimum exploration rate
    epsilon_decay(0.995),     // Faster decay: 0.995 (reaches min in ~900 games)
    learning_rate(0.002),     // Increased learning rate for faster learning (2x)
    gamma(0.99),              // High gamma for better long-term planning
    training_episodes(0),
    total_games(0),
    best_score(0),
    average_score(0.0),
    previous_avg_score(0.0),
    recent_scores_sum(0),
    last_batch_error(0.0),
    model_loaded(false),
    last_epsilon(1.0),
    epsilon_change_reason(0.0),
    epsilon_increase_count(0),
    epsilon_decrease_count(0) {
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
    std::vector<double> state(NeuralNetwork::INPUT_SIZE, 0.0);
    int idx = 0;
    
    if (game.current_piece == nullptr) {
        return state;
    }
    
    // Column heights (10 features)
    for (int x = 0; x < game.WIDTH; x++) {
        state[idx++] = game.getColumnHeight(x, game.board) / 20.0;  // Normalize
    }
    
    // Holes count
    state[idx++] = game.countHoles(game.board) / 100.0;
    
    // Bumpiness
    state[idx++] = game.calculateBumpiness(game.board) / 50.0;
    
    // Aggregate height
    state[idx++] = game.getAggregateHeight(game.board) / 200.0;
    
    // Current piece type (7 features)
    for (int i = 0; i < 7; i++) {
        state[idx++] = (game.current_piece->type == i) ? 1.0 : 0.0;
    }
    
    // Next piece type (7 features)
    if (game.next_piece) {
        for (int i = 0; i < 7; i++) {
            state[idx++] = (game.next_piece->type == i) ? 1.0 : 0.0;
        }
    } else {
        for (int i = 0; i < 7; i++) {
            state[idx++] = 0.0;
        }
    }
    
    // Lines cleared
    state[idx++] = game.lines_cleared / 100.0;
    
    // Level
    state[idx++] = game.level / 20.0;
    
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
    
    // Exploit: find best Q-value (with safety limit)
    int move_evaluations = 0;
    const int MAX_EVALUATIONS = 200;  // Safety limit to prevent CPU spinning
    
    for (int rot = 0; rot < 4 && move_evaluations < MAX_EVALUATIONS; rot++) {
        piece.rotation = rot;
        
        for (int x = -2; x < game.WIDTH + 2 && move_evaluations < MAX_EVALUATIONS; x++) {
            piece.x = x;
            move_evaluations++;
            piece.y = 0;
            
            if (game.checkCollision(piece)) continue;
            
            // Simulate drop (with safety limit to prevent infinite loop)
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
            
            // Create temporary game state for Q-value (without copying pieces)
            // We'll manually create the state from the simulated board
            std::vector<double> next_state(NeuralNetwork::INPUT_SIZE, 0.0);
            int idx = 0;
            
            // Column heights (10 features)
            for (int x = 0; x < game.WIDTH; x++) {
                int height = 0;
                for (int y = 0; y < game.HEIGHT; y++) {
                    if (sim_board[y][x] != 0) {
                        height = game.HEIGHT - y;
                        break;
                    }
                }
                next_state[idx++] = height / 20.0;
            }
            
            // Holes count
            int holes = 0;
            for (int x = 0; x < game.WIDTH; x++) {
                bool block_found = false;
                for (int y = 0; y < game.HEIGHT; y++) {
                    if (sim_board[y][x] != 0) {
                        block_found = true;
                    } else if (block_found) {
                        holes++;
                    }
                }
            }
            next_state[idx++] = holes / 100.0;
            
            // Bumpiness
            int bumpiness = 0;
            for (int x = 0; x < game.WIDTH - 1; x++) {
                int h1 = 0, h2 = 0;
                for (int y = 0; y < game.HEIGHT; y++) {
                    if (sim_board[y][x] != 0 && h1 == 0) h1 = game.HEIGHT - y;
                    if (sim_board[y][x+1] != 0 && h2 == 0) h2 = game.HEIGHT - y;
                }
                bumpiness += abs(h1 - h2);
            }
            next_state[idx++] = bumpiness / 50.0;
            
            // Aggregate height
            int aggregate_height = 0;
            for (int x = 0; x < game.WIDTH; x++) {
                for (int y = 0; y < game.HEIGHT; y++) {
                    if (sim_board[y][x] != 0) {
                        aggregate_height += game.HEIGHT - y;
                        break;
                    }
                }
            }
            next_state[idx++] = aggregate_height / 200.0;
            
            // Current piece type (7 features) - none since piece is placed
            for (int i = 0; i < 7; i++) {
                next_state[idx++] = 0.0;
            }
            
            // Next piece type (7 features)
            if (game.next_piece) {
                for (int i = 0; i < 7; i++) {
                    next_state[idx++] = (game.next_piece->type == i) ? 1.0 : 0.0;
                }
            } else {
                for (int i = 0; i < 7; i++) {
                    next_state[idx++] = 0.0;
                }
            }
            
            // Lines cleared
            next_state[idx++] = (game.lines_cleared + lines_cleared) / 100.0;
            
            // Level
            next_state[idx++] = game.level / 20.0;
            double q_value = q_network.forward(next_state);
            
            // Don't add extra reward shaping here - the network should learn from actual rewards
            // The Q-value from the network already incorporates learned value
            
            if (q_value > best_move.q_value) {
                best_move.rotation = rot;
                best_move.x = x;
                best_move.q_value = q_value;
            }
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
    
    // Sample random batch with bias toward recent experiences (80% recent, 20% old)
    // This prevents catastrophic forgetting while still learning from new experiences
    std::vector<Experience> batch;
    int recent_size = std::min((int)replay_buffer.size() / 4, (int)replay_buffer.size());  // Last 25% are "recent"
    int old_size = replay_buffer.size() - recent_size;
    
    for (int i = 0; i < BATCH_SIZE; i++) {
        int idx;
        // 80% chance to sample from recent experiences, 20% from old
        if (old_size > 0 && (rand() % 100) < 20) {
            // Sample from old experiences
            idx = rand() % old_size;
        } else {
            // Sample from recent experiences
            idx = old_size + (rand() % recent_size);
        }
        batch.push_back(replay_buffer[idx]);
    }
    
    // Train on batch with proper Q-learning
    double batch_avg_error = 0.0;
    int valid_updates = 0;
    int total_samples = 0;
    
    for (const auto& exp : batch) {
        total_samples++;
        
        // Calculate Q-learning target: r + gamma * max Q(s', a')
        double target = exp.reward;
        if (!exp.done) {
            double next_q = q_network.forward(exp.next_state);
            // Clip next Q-value to prevent extreme values
            next_q = std::max(-100.0, std::min(100.0, next_q));
            target += gamma * next_q;
        }
        
        // Clip target to reasonable range
        target = std::max(-100.0, std::min(100.0, target));
        
        double predicted = q_network.forward(exp.state);
        double error = std::abs(target - predicted);
        batch_avg_error += error;  // Track all errors, not just valid ones
        
        // Update network with adaptive learning rate
        // Reduce learning rate when error is small (fine-tuning) to prevent overshooting
        if (std::isfinite(target) && std::isfinite(predicted)) {
            // Only skip if error is extremely large (likely bad data)
            if (error < 10000.0) {  // Increased from 1000.0
                // Adaptive learning rate: reduce when error is small (fine-tuning)
                double adaptive_lr = learning_rate;
                if (error < 1.0 && average_score > 300.0) {
                    // High performance and small error - use reduced learning rate to prevent overshooting
                    adaptive_lr = learning_rate * 0.5;  // Half learning rate for fine-tuning
                } else if (error < 0.5 && average_score > 500.0) {
                    // Very high performance - use even smaller learning rate
                    adaptive_lr = learning_rate * 0.25;  // Quarter learning rate for very stable performance
                }
                q_network.update(exp.state, target, adaptive_lr);
                valid_updates++;
            }
        }
    }
    
    // Calculate average error (use all samples for better visibility)
    if (total_samples > 0) {
        batch_avg_error /= total_samples;
    }
    
    // If no valid updates but we have errors, still show the error
    // This helps debug why updates aren't happening
    if (valid_updates == 0 && batch_avg_error > 0.0) {
        // Log that we have errors but no updates (for debugging)
        // This shouldn't happen often, but if it does, we want to know
    }
    
    // Check for NaN/Inf in error (prevent crashes)
    if (!std::isfinite(batch_avg_error)) {
        batch_avg_error = 0.0;  // Reset if invalid
    }
    
    // Store error for display (always show error, even if no updates)
    last_batch_error = batch_avg_error;
    
    // Weight changes are now displayed on screen only (no log file)
    
    training_episodes++;
}

void RLAgent::updateEpsilonBasedOnPerformance() {
    // Always apply baseline decay (ensures epsilon decreases over time)
    // Start decaying after just a few games to allow learning
    if (epsilon > epsilon_min && total_games > 5) {
        epsilon *= epsilon_decay;  // Use configured decay rate
        if (epsilon < epsilon_min) {
            epsilon = epsilon_min;
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
    
    // Calculate improvement
    double improvement = average_score - previous_avg_score;
    double improvement_percent = 0.0;
    if (previous_avg_score > 1.0) {  // Need meaningful baseline
        improvement_percent = (improvement / previous_avg_score) * 100.0;
    }
    
    // Track epsilon before change
    double epsilon_before = epsilon;
    std::string change_reason = "";
    
    // Update epsilon based on performance (more responsive thresholds)
    if (improvement_percent > 1.0) {
        // Improvement (>1%): Faster decay (network is learning well)
        if (epsilon > epsilon_min) {
            epsilon *= 0.99;  // Faster decay when learning well (1% per update)
            if (epsilon < epsilon_min) {
                epsilon = epsilon_min;
            }
            change_reason = "Score improving";
            epsilon_decrease_count++;
        }
    } else if (improvement_percent < -1.0) {
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
        // If average score is very low (< 200) and not improving, increase exploration
        if (average_score < 200.0 && improvement_percent < 0.5) {
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
    
    // Update previous average for next comparison (update every 5 games or on significant change)
    if (std::abs(improvement_percent) > 1.0 || total_games % 5 == 0) {
        previous_avg_score = average_score;
    }
}

void RLAgent::saveModel() {
    saveModelToFile("tetris_model.txt");
}

void RLAgent::saveModelToFile(const std::string& filename) {
    // Save network weights first
    q_network.save(filename);
    
    // Append training state metadata to the model file
    std::ofstream file(filename, std::ios::app);
    if (file.is_open()) {
        // Add separator and metadata header
        file << "\n# Training State Metadata\n";
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

