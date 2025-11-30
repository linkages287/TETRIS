#include "parameter_tuner.h"
#include "rl_agent.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

ParameterTuner::ParameterTuner()
    : current_iteration(0),
      last_evaluation_iteration(0),
      current_param_set_index(0),
      auto_tuning_enabled(true) {
    
    // Define parameter sets to test (grid search approach)
    // Format: (learning_rate, gamma, epsilon_decay, epsilon_min, batch_size)
    
    // Conservative set (slow but stable)
    parameter_sets.push_back(ParameterSet(0.001, 0.99, 0.995, 0.05, 32));
    
    // Moderate set (balanced)
    parameter_sets.push_back(ParameterSet(0.002, 0.99, 0.995, 0.05, 32));
    
    // Aggressive set (fast learning)
    parameter_sets.push_back(ParameterSet(0.003, 0.99, 0.995, 0.05, 32));
    
    // High exploration set
    parameter_sets.push_back(ParameterSet(0.002, 0.99, 0.998, 0.10, 32));
    
    // Low exploration set
    parameter_sets.push_back(ParameterSet(0.002, 0.99, 0.992, 0.02, 32));
    
    // Larger batch size
    parameter_sets.push_back(ParameterSet(0.002, 0.99, 0.995, 0.05, 64));
    
    // Higher gamma (more long-term)
    parameter_sets.push_back(ParameterSet(0.002, 0.995, 0.995, 0.05, 32));
}

void ParameterTuner::recordError(double error) {
    error_history.push_back(error);
    if (error_history.size() > HISTORY_SIZE) {
        error_history.pop_front();
    }
}

void ParameterTuner::recordEpsilon(double epsilon) {
    epsilon_history.push_back(epsilon);
    if (epsilon_history.size() > HISTORY_SIZE) {
        epsilon_history.pop_front();
    }
}

void ParameterTuner::recordScore(double score) {
    score_history.push_back(score);
    if (score_history.size() > HISTORY_SIZE) {
        score_history.pop_front();
    }
}

PerformanceMetrics ParameterTuner::evaluatePerformance() {
    PerformanceMetrics metrics;
    
    if (error_history.empty() || score_history.empty()) {
        return metrics;
    }
    
    // Calculate average error
    double error_sum = 0.0;
    for (double e : error_history) {
        error_sum += e;
    }
    metrics.avg_error = error_sum / error_history.size();
    
    // Calculate average score
    double score_sum = 0.0;
    for (double s : score_history) {
        score_sum += s;
    }
    metrics.avg_score = score_sum / score_history.size();
    
    // Calculate score improvement (trend)
    if (score_history.size() >= 20) {
        double recent_avg = 0.0;
        double old_avg = 0.0;
        
        // Last 10 scores
        for (size_t i = score_history.size() - 10; i < score_history.size(); i++) {
            recent_avg += score_history[i];
        }
        recent_avg /= 10.0;
        
        // Previous 10 scores
        for (size_t i = score_history.size() - 20; i < score_history.size() - 10; i++) {
            old_avg += score_history[i];
        }
        old_avg /= 10.0;
        
        if (old_avg > 0.0) {
            metrics.score_improvement = ((recent_avg - old_avg) / old_avg) * 100.0;
        }
    }
    
    // Calculate error reduction (trend)
    if (error_history.size() >= 20) {
        double recent_avg = 0.0;
        double old_avg = 0.0;
        
        // Last 10 errors
        for (size_t i = error_history.size() - 10; i < error_history.size(); i++) {
            recent_avg += error_history[i];
        }
        recent_avg /= 10.0;
        
        // Previous 10 errors
        for (size_t i = error_history.size() - 20; i < error_history.size() - 10; i++) {
            old_avg += error_history[i];
        }
        old_avg /= 10.0;
        
        if (old_avg > 0.0) {
            metrics.error_reduction = ((old_avg - recent_avg) / old_avg) * 100.0;
        }
    }
    
    metrics.games_played = score_history.size();
    
    return metrics;
}

bool ParameterTuner::shouldTestNewParameters() {
    if (!auto_tuning_enabled) return false;
    
    current_iteration++;
    
    // Check if enough iterations have passed
    if (current_iteration - last_evaluation_iteration < EVALUATION_INTERVAL) {
        return false;
    }
    
    // Need minimum games for meaningful evaluation
    if (score_history.size() < MIN_GAMES_FOR_EVAL) {
        return false;
    }
    
    // Evaluate current performance
    current_metrics = evaluatePerformance();
    
    // If this is the first evaluation, set baseline
    if (last_evaluation_iteration == 0) {
        baseline_metrics = current_metrics;
        last_evaluation_iteration = current_iteration;
        return false;  // Don't switch yet, need baseline
    }
    
    // Check if performance is poor
    bool poor_performance = false;
    
    // Low average score
    if (current_metrics.avg_score < 100.0) {
        poor_performance = true;
    }
    
    // Score not improving
    if (current_metrics.score_improvement < 1.0 && current_metrics.avg_score < 300.0) {
        poor_performance = true;
    }
    
    // Error not decreasing (or increasing)
    if (current_metrics.error_reduction < 0.0 && current_metrics.avg_error > 1.0) {
        poor_performance = true;
    }
    
    // High error with no improvement
    if (current_metrics.avg_error > 10.0 && current_metrics.error_reduction < 5.0) {
        poor_performance = true;
    }
    
    if (poor_performance) {
        last_evaluation_iteration = current_iteration;
        return true;  // Time to try new parameters
    }
    
    return false;
}

ParameterSet ParameterTuner::getNextParameterSet() {
    if (current_param_set_index >= (int)parameter_sets.size()) {
        // Cycle back to beginning
        current_param_set_index = 0;
    }
    
    return parameter_sets[current_param_set_index++];
}

void ParameterTuner::applyParameters(ParameterSet& params, RLAgent& agent) {
    agent.learning_rate = params.learning_rate;
    agent.gamma = params.gamma;
    agent.epsilon_decay = params.epsilon_decay;
    agent.epsilon_min = params.epsilon_min;
    
    // Note: batch_size is a static const, so we can't change it easily
    // But we can log it for reference
}

std::string ParameterTuner::generateReport() {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    
    oss << "=== Parameter Tuning Report ===\n";
    oss << "Iteration: " << current_iteration << "\n";
    oss << "Games Played: " << score_history.size() << "\n";
    oss << "Current Parameter Set: " << current_param_set_index << "/" << parameter_sets.size() << "\n";
    
    if (!error_history.empty() && !score_history.empty()) {
        PerformanceMetrics metrics = evaluatePerformance();
        
        oss << "\nPerformance Metrics:\n";
        oss << "  Avg Error: " << metrics.avg_error << "\n";
        oss << "  Avg Score: " << metrics.avg_score << "\n";
        oss << "  Score Improvement: " << metrics.score_improvement << "%\n";
        oss << "  Error Reduction: " << metrics.error_reduction << "%\n";
        
        if (current_param_set_index > 0 && current_param_set_index <= (int)parameter_sets.size()) {
            ParameterSet current = parameter_sets[current_param_set_index - 1];
            oss << "\nCurrent Parameters:\n";
            oss << "  Learning Rate: " << current.learning_rate << "\n";
            oss << "  Gamma: " << current.gamma << "\n";
            oss << "  Epsilon Decay: " << current.epsilon_decay << "\n";
            oss << "  Epsilon Min: " << current.epsilon_min << "\n";
            oss << "  Batch Size: " << current.batch_size << "\n";
        }
        
        // Epsilon trend
        if (epsilon_history.size() >= 2) {
            double epsilon_start = epsilon_history[0];
            double epsilon_end = epsilon_history.back();
            oss << "\nEpsilon Trend: " << epsilon_start << " -> " << epsilon_end;
            if (epsilon_start > epsilon_end) {
                oss << " (decreasing - good)";
            } else {
                oss << " (not decreasing - may need adjustment)";
            }
        }
    }
    
    return oss.str();
}

void ParameterTuner::resetForNewParameters() {
    // Clear history to start fresh evaluation
    error_history.clear();
    epsilon_history.clear();
    score_history.clear();
    baseline_metrics = PerformanceMetrics();
    current_metrics = PerformanceMetrics();
}

