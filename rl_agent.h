#ifndef RL_AGENT_H
#define RL_AGENT_H

#include <vector>
#include <deque>
#include <string>

// Forward declaration
class TetrisGame;
class TetrisPiece;

// Experience for replay buffer
struct Experience {
    std::vector<double> state;
    int action_rotation;
    int action_x;
    double reward;
    std::vector<double> next_state;
    bool done;
};

// Simple Neural Network for Q-Learning
class NeuralNetwork {
public:
    std::vector<std::vector<double>> weights1;  // Input to hidden
    std::vector<double> bias1;                  // Hidden bias
    std::vector<std::vector<double>> weights2;  // Hidden to output
    std::vector<double> bias2;                  // Output bias
    
        static const int INPUT_SIZE = 29;   // State features: 10 heights + 1 holes + 1 bumpiness + 1 aggregate + 7 current + 7 next + 1 lines + 1 level
    static const int HIDDEN_SIZE = 64;
    static const int OUTPUT_SIZE = 1;  // Q-value
    
    NeuralNetwork();
    double relu(double x) const;
    double leaky_relu(double x) const;  // Leaky ReLU to prevent dead neurons
    double forward(const std::vector<double>& input);
    void update(const std::vector<double>& input, double target, double learning_rate);
    void save(const std::string& filename);
    bool load(const std::string& filename);
    void logWeightChanges(const std::string& filename, int episode, double error);
    std::string getWeightStatsString(int episode, double error);  // Get stats as string for display
    
    // Saturation monitoring
    struct SaturationMetrics {
        double weights1_saturation;  // Percentage of identical values in weights1
        double bias1_saturation;     // Percentage of identical values in bias1
        double weights2_saturation;  // Percentage of identical values in weights2
        double bias2_saturation;     // Percentage of identical values in bias2
        double weights1_variance;    // Variance of weights1
        double bias1_variance;      // Variance of bias1
        double weights2_variance;   // Variance of weights2
        double bias2_variance;      // Variance of bias2
    };
    SaturationMetrics calculateSaturation() const;  // Calculate saturation metrics for all layers
};

// Reinforcement Learning Agent
class RLAgent {
public:
    NeuralNetwork q_network;
    std::deque<Experience> replay_buffer;
    static const int BUFFER_SIZE = 10000;
    static const int BATCH_SIZE = 32;
    
    double epsilon;           // Exploration rate
    double epsilon_min;
    double epsilon_decay;
    double learning_rate;
    double gamma;             // Discount factor
    
        int training_episodes;
        int total_games;
        int best_score;
        double average_score;      // Running average of recent scores
        double previous_avg_score;  // Previous average score for comparison
        int recent_scores_sum;      // Sum of recent scores for averaging
        double last_batch_error;   // Last training batch error for display
        bool model_loaded;         // Whether model was loaded from file
        static const int RECENT_SCORES_COUNT = 100;  // Number of recent games to average
        
        // Convergence detection
        std::deque<int> recent_scores;  // Store recent scores for convergence detection
        int games_since_best_improvement;  // Games since best score improved
        double convergence_check_interval;  // Last convergence check game count
        static const int CONVERGENCE_WINDOW = 500;  // Games to check for stability
        static const int CONVERGENCE_STABILITY_THRESHOLD = 500;  // Games with stable score
        static constexpr double CONVERGENCE_VARIATION_THRESHOLD = 0.05;  // 5% variation allowed
        
        // Epsilon-Score relationship tracking
        double last_epsilon;       // Previous epsilon value
        double epsilon_change_reason;  // Reason for epsilon change (improvement %)
        int epsilon_increase_count;    // Count of epsilon increases
        int epsilon_decrease_count;    // Count of epsilon decreases
    
    struct Move {
        int rotation;
        int x;
        double q_value;
    };
    
    RLAgent(const std::string& model_file = "tetris_model.txt");  // Allow custom model file
    
    // Extract state features from game
    std::vector<double> extractState(const TetrisGame& game);
    
    // Find best move using Q-learning
    Move findBestMove(const TetrisGame& game, bool training = false);
    
    // Experience replay
    void addExperience(const Experience& exp);
    void train();
    void updateEpsilonBasedOnPerformance();  // Adaptive epsilon based on score improvement
    bool checkConvergence();  // Check if network has converged
    void saveModel();
    void saveModelToFile(const std::string& filename);  // Save model to specific file
};

#endif // RL_AGENT_H

