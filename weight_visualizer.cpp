#include <ncurses.h>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <unistd.h>
#include <chrono>
#include <iomanip>
#include <sstream>

// Neural Network structure matching rl_agent.h
const int INPUT_SIZE = 29;
const int HIDDEN_SIZE = 64;
const int OUTPUT_SIZE = 1;

struct NeuralNetwork {
    std::vector<std::vector<double>> weights1;  // Input to hidden
    std::vector<double> bias1;                  // Hidden bias
    std::vector<std::vector<double>> weights2;  // Hidden to output
    std::vector<double> bias2;                  // Output bias
    
    NeuralNetwork() {
        weights1.resize(INPUT_SIZE, std::vector<double>(HIDDEN_SIZE, 0.0));
        bias1.resize(HIDDEN_SIZE, 0.0);
        weights2.resize(HIDDEN_SIZE, std::vector<double>(OUTPUT_SIZE, 0.0));
        bias2.resize(OUTPUT_SIZE, 0.0);
    }
    
    bool load(const std::string& filename) {
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
};

// Calculate difference between two networks
double calculateWeightDiff(const NeuralNetwork& old_net, const NeuralNetwork& new_net) {
    double total_diff = 0.0;
    int count = 0;
    
    // Compare weights1
    for (size_t i = 0; i < old_net.weights1.size(); i++) {
        for (size_t j = 0; j < old_net.weights1[i].size(); j++) {
            total_diff += std::abs(new_net.weights1[i][j] - old_net.weights1[i][j]);
            count++;
        }
    }
    
    // Compare bias1
    for (size_t i = 0; i < old_net.bias1.size(); i++) {
        total_diff += std::abs(new_net.bias1[i] - old_net.bias1[i]);
        count++;
    }
    
    // Compare weights2
    for (size_t i = 0; i < old_net.weights2.size(); i++) {
        for (size_t j = 0; j < old_net.weights2[i].size(); j++) {
            total_diff += std::abs(new_net.weights2[i][j] - old_net.weights2[i][j]);
            count++;
        }
    }
    
    // Compare bias2
    for (size_t i = 0; i < old_net.bias2.size(); i++) {
        total_diff += std::abs(new_net.bias2[i] - old_net.bias2[i]);
        count++;
    }
    
    return count > 0 ? total_diff / count : 0.0;
}

// Get color based on weight change
int getChangeColor(double old_val, double new_val, double threshold = 0.0001) {
    double diff = new_val - old_val;
    if (std::abs(diff) < threshold) {
        return 2; // Yellow (stability)
    } else if (diff > 0) {
        return 3; // Green (increase)
    } else {
        return 1; // Red (decrease)
    }
}

// Initialize colors
void initColors() {
    start_color();
    init_pair(1, COLOR_RED, COLOR_BLACK);      // Red for decrease
    init_pair(2, COLOR_YELLOW, COLOR_BLACK);    // Yellow for stability
    init_pair(3, COLOR_GREEN, COLOR_BLACK);     // Green for increase
    init_pair(4, COLOR_WHITE, COLOR_BLACK);     // White for text
    init_pair(5, COLOR_CYAN, COLOR_BLACK);      // Cyan for headers
}

// Draw weight matrix visualization
void drawWeightMatrix(WINDOW* win, const NeuralNetwork& old_net, const NeuralNetwork& new_net, 
                      int start_y, int start_x, int rows, int cols, bool is_weights1) {
    int max_rows = std::min(rows, is_weights1 ? INPUT_SIZE : HIDDEN_SIZE);
    int max_cols = std::min(cols, is_weights1 ? HIDDEN_SIZE : OUTPUT_SIZE);
    
    for (int i = 0; i < max_rows; i++) {
        for (int j = 0; j < max_cols; j++) {
            double old_val = is_weights1 ? old_net.weights1[i][j] : old_net.weights2[i][j];
            double new_val = is_weights1 ? new_net.weights1[i][j] : new_net.weights2[i][j];
            
            int color = getChangeColor(old_val, new_val, 0.0001);
            wattron(win, COLOR_PAIR(color));
            
            // Display as a character based on magnitude
            char ch = ' ';
            double abs_val = std::abs(new_val);
            double change = std::abs(new_val - old_val);
            
            // Character based on weight magnitude
            if (abs_val > 0.5) ch = '#';
            else if (abs_val > 0.1) ch = '*';
            else if (abs_val > 0.01) ch = '.';
            else ch = ' ';
            
            // If change is significant, make it more visible
            if (change > 0.001) {
                if (ch == ' ') ch = '+';  // Show small but changing weights
            }
            
            mvwaddch(win, start_y + i, start_x + j, ch);
            wattroff(win, COLOR_PAIR(color));
        }
    }
}

// Draw statistics
void drawStats(WINDOW* win, const NeuralNetwork& old_net, const NeuralNetwork& new_net, 
               int y, int x, int update_count) {
    // Calculate statistics
    double w1_min = new_net.weights1[0][0], w1_max = new_net.weights1[0][0];
    double w1_sum = 0.0, w1_abs_sum = 0.0;
    int w1_count = 0;
    
    for (const auto& row : new_net.weights1) {
        for (double w : row) {
            w1_min = std::min(w1_min, w);
            w1_max = std::max(w1_max, w);
            w1_sum += w;
            w1_abs_sum += std::abs(w);
            w1_count++;
        }
    }
    
    double w2_min = new_net.weights2[0][0], w2_max = new_net.weights2[0][0];
    double w2_sum = 0.0, w2_abs_sum = 0.0;
    int w2_count = 0;
    
    for (const auto& row : new_net.weights2) {
        for (double w : row) {
            w2_min = std::min(w2_min, w);
            w2_max = std::max(w2_max, w);
            w2_sum += w;
            w2_abs_sum += std::abs(w);
            w2_count++;
        }
    }
    
    double avg_diff = calculateWeightDiff(old_net, new_net);
    
    wattron(win, COLOR_PAIR(5));
    mvwprintw(win, y, x, "=== Neural Network Weight Visualizer ===");
    wattroff(win, COLOR_PAIR(5));
    
    wattron(win, COLOR_PAIR(4));
    mvwprintw(win, y + 1, x, "Updates: %d | Avg Weight Change: %.6f", update_count, avg_diff);
    mvwprintw(win, y + 2, x, "Weights1: min=%.3f max=%.3f mean=%.3f |W|=%.3f", 
              w1_min, w1_max, w1_sum / w1_count, w1_abs_sum / w1_count);
    mvwprintw(win, y + 3, x, "Weights2: min=%.3f max=%.3f mean=%.3f |W|=%.3f", 
              w2_min, w2_max, w2_sum / w2_count, w2_abs_sum / w2_count);
    mvwprintw(win, y + 4, x, "Bias1: min=%.3f max=%.3f | Bias2: %.3f", 
              *std::min_element(new_net.bias1.begin(), new_net.bias1.end()),
              *std::max_element(new_net.bias1.begin(), new_net.bias1.end()),
              new_net.bias2[0]);
    wattroff(win, COLOR_PAIR(4));
    
    // Legend
    wattron(win, COLOR_PAIR(1));
    mvwprintw(win, y + 5, x, "Red: Decrease");
    wattroff(win, COLOR_PAIR(1));
    wattron(win, COLOR_PAIR(2));
    mvwprintw(win, y + 5, x + 20, "Yellow: Stable");
    wattroff(win, COLOR_PAIR(2));
    wattron(win, COLOR_PAIR(3));
    mvwprintw(win, y + 5, x + 40, "Green: Increase");
    wattroff(win, COLOR_PAIR(3));
    wattron(win, COLOR_PAIR(4));
    mvwprintw(win, y + 6, x, "Symbols: # (|w|>0.5) * (|w|>0.1) . (|w|>0.01) + (changing)");
    wattroff(win, COLOR_PAIR(4));
}

int main() {
    // Initialize ncurses
    initscr();
    cbreak();
    noecho();
    curs_set(0);
    keypad(stdscr, TRUE);
    nodelay(stdscr, TRUE);
    
    initColors();
    
    NeuralNetwork old_net, new_net;
    bool first_load = true;
    int update_count = 0;
    auto last_update = std::chrono::steady_clock::now();
    
    // Try to load initial model
    if (!new_net.load("tetris_model.txt")) {
        mvprintw(0, 0, "Waiting for tetris_model.txt to be created...");
        refresh();
        
        // Wait for file to exist
        while (!new_net.load("tetris_model.txt")) {
            usleep(100000); // 100ms
            if (getch() == 'q' || getch() == 'Q') {
                endwin();
                return 0;
            }
        }
    }
    
    old_net = new_net;
    first_load = false;
    
    mvprintw(0, 0, "Model loaded! Monitoring weight changes... (Press 'q' to quit)");
    refresh();
    
    while (true) {
        // Check for quit
        int ch = getch();
        if (ch == 'q' || ch == 'Q') {
            break;
        }
        
        // Try to reload model (with retry for file locking)
        NeuralNetwork temp_net;
        bool loaded = false;
        for (int retry = 0; retry < 3; retry++) {
            if (temp_net.load("tetris_model.txt")) {
                loaded = true;
                break;
            }
            usleep(10000); // 10ms delay before retry
        }
        
        if (loaded) {
            // Check if weights actually changed
            double diff = calculateWeightDiff(new_net, temp_net);
            if (diff > 0.000001 || first_load) {  // Only update if significant change
                old_net = new_net;
                new_net = temp_net;
                update_count++;
                last_update = std::chrono::steady_clock::now();
                first_load = false;
            }
        }
        
        // Clear screen
        erase();
        
        // Get screen dimensions
        int max_y, max_x;
        getmaxyx(stdscr, max_y, max_x);
        
        // Draw statistics
        drawStats(stdscr, old_net, new_net, 0, 0, update_count);
        
        int stats_height = 7;
        int available_height = max_y - stats_height - 5;
        
        // Draw Weights1 visualization (Input to Hidden)
        wattron(stdscr, COLOR_PAIR(5));
        mvprintw(stats_height, 0, "Weights1 (Input->Hidden): %dx%d", INPUT_SIZE, HIDDEN_SIZE);
        wattroff(stdscr, COLOR_PAIR(5));
        
        // Show a subset of weights1 (first 20 inputs, first 40 hidden)
        int w1_display_rows = std::min(20, available_height / 2);
        int w1_display_cols = std::min(60, max_x);
        drawWeightMatrix(stdscr, old_net, new_net, stats_height + 1, 0, 
                        w1_display_rows, w1_display_cols, true);
        
        // Draw Weights2 visualization (Hidden to Output)
        int w2_y = stats_height + w1_display_rows + 2;
        wattron(stdscr, COLOR_PAIR(5));
        mvprintw(w2_y, 0, "Weights2 (Hidden->Output): %dx%d", HIDDEN_SIZE, OUTPUT_SIZE);
        wattroff(stdscr, COLOR_PAIR(5));
        
        // Show all weights2 (64 hidden, 1 output)
        int w2_display_rows = std::min(HIDDEN_SIZE, max_y - w2_y - 2);
        int w2_display_cols = std::min(OUTPUT_SIZE, max_x);
        drawWeightMatrix(stdscr, old_net, new_net, w2_y + 1, 0, 
                        w2_display_rows, w2_display_cols, false);
        
        // Show last update time
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_update).count();
        wattron(stdscr, COLOR_PAIR(4));
        mvprintw(max_y - 1, 0, "Last update: %ld ms ago | Press 'q' to quit", elapsed);
        wattroff(stdscr, COLOR_PAIR(4));
        
        refresh();
        
        // Sleep for a bit to avoid excessive CPU usage
        usleep(50000); // 50ms
    }
    
    endwin();
    return 0;
}

