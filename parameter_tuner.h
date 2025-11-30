#ifndef PARAMETER_TUNER_H
#define PARAMETER_TUNER_H

#include <vector>
#include <deque>
#include <string>

// Parameter tuning configuration
struct ParameterSet {
    double learning_rate;
    double gamma;
    double epsilon_decay;
    double epsilon_min;
    int batch_size;
    
    ParameterSet(double lr, double g, double ed, double em, int bs)
        : learning_rate(lr), gamma(g), epsilon_decay(ed), epsilon_min(em), batch_size(bs) {}
};

// Performance metrics for evaluation
struct PerformanceMetrics {
    double avg_error;
    double avg_score;
    double score_improvement;
    double error_reduction;
    int games_played;
    
    PerformanceMetrics() : avg_error(0.0), avg_score(0.0), score_improvement(0.0), 
                          error_reduction(0.0), games_played(0) {}
};

// Parameter tuner that monitors performance and adjusts hyperparameters
class ParameterTuner {
public:
    static const int EVALUATION_INTERVAL = 500;  // Evaluate every N training episodes
    static const int MIN_GAMES_FOR_EVAL = 50;    // Minimum games before evaluation
    
    std::deque<double> error_history;      // Track error over time
    std::deque<double> epsilon_history;   // Track epsilon over time
    std::deque<double> score_history;      // Track scores over time
    
    static const int HISTORY_SIZE = 100;  // Keep last N values
    
    int current_iteration;
    int last_evaluation_iteration;
    PerformanceMetrics baseline_metrics;
    PerformanceMetrics current_metrics;
    
    std::vector<ParameterSet> parameter_sets;  // Different parameter combinations to try
    int current_param_set_index;
    bool auto_tuning_enabled;
    
    ParameterTuner();
    
    // Record metrics
    void recordError(double error);
    void recordEpsilon(double epsilon);
    void recordScore(double score);
    
    // Evaluate current performance
    PerformanceMetrics evaluatePerformance();
    
    // Check if we should test new parameters
    bool shouldTestNewParameters();
    
    // Get next parameter set to test
    ParameterSet getNextParameterSet();
    
    // Apply parameter set to agent
    void applyParameters(ParameterSet& params, class RLAgent& agent);
    
    // Generate report
    std::string generateReport();
    
    // Reset for new parameter set
    void resetForNewParameters();
};

#endif // PARAMETER_TUNER_H

