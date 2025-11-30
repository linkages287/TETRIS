/*
 * Terminal Tetris Game (C++)
 * Controls:
 *     Arrow Keys - Move left/right, rotate, soft drop
 *     Space - Hard drop
 *     Q - Quit
 *     P - Pause
 */

#include <ncurses.h>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include "rl_agent.h"
#include "parameter_tuner.h"
#include "game_classes.h"

// Debug logging function
void debugLog(const std::string& message) {
    static std::ofstream debug_file("debug.log", std::ios::app);
    if (debug_file.is_open()) {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        debug_file << "[" << time_t << "] " << message << "\n";
        debug_file.flush();
    }
}

// Tetris pieces (tetrominoes) - each piece is defined by its shape
// Format: [rotations][y][x] where each rotation is a 4x4 grid
const int PIECES[7][4][4][4] = {
    // I piece
    {
        {{0,0,0,0}, {1,1,1,1}, {0,0,0,0}, {0,0,0,0}},
        {{0,0,1,0}, {0,0,1,0}, {0,0,1,0}, {0,0,1,0}},
        {{0,0,0,0}, {0,0,0,0}, {1,1,1,1}, {0,0,0,0}},
        {{0,1,0,0}, {0,1,0,0}, {0,1,0,0}, {0,1,0,0}}
    },
    // O piece
    {
        {{0,0,0,0}, {0,1,1,0}, {0,1,1,0}, {0,0,0,0}},
        {{0,0,0,0}, {0,1,1,0}, {0,1,1,0}, {0,0,0,0}},
        {{0,0,0,0}, {0,1,1,0}, {0,1,1,0}, {0,0,0,0}},
        {{0,0,0,0}, {0,1,1,0}, {0,1,1,0}, {0,0,0,0}}
    },
    // T piece
    {
        {{0,0,0,0}, {0,1,0,0}, {1,1,1,0}, {0,0,0,0}},
        {{0,0,0,0}, {0,1,0,0}, {0,1,1,0}, {0,1,0,0}},
        {{0,0,0,0}, {0,0,0,0}, {1,1,1,0}, {0,1,0,0}},
        {{0,0,0,0}, {0,1,0,0}, {1,1,0,0}, {0,1,0,0}}
    },
    // S piece
    {
        {{0,0,0,0}, {0,1,1,0}, {1,1,0,0}, {0,0,0,0}},
        {{0,0,0,0}, {0,1,0,0}, {0,1,1,0}, {0,0,1,0}},
        {{0,0,0,0}, {0,0,0,0}, {0,1,1,0}, {1,1,0,0}},
        {{0,0,0,0}, {1,0,0,0}, {1,1,0,0}, {0,1,0,0}}
    },
    // Z piece
    {
        {{0,0,0,0}, {1,1,0,0}, {0,1,1,0}, {0,0,0,0}},
        {{0,0,0,0}, {0,0,1,0}, {0,1,1,0}, {0,1,0,0}},
        {{0,0,0,0}, {0,0,0,0}, {1,1,0,0}, {0,1,1,0}},
        {{0,0,0,0}, {0,1,0,0}, {1,1,0,0}, {1,0,0,0}}
    },
    // J piece
    {
        {{0,0,0,0}, {1,0,0,0}, {1,1,1,0}, {0,0,0,0}},
        {{0,0,0,0}, {0,1,1,0}, {0,1,0,0}, {0,1,0,0}},
        {{0,0,0,0}, {0,0,0,0}, {1,1,1,0}, {0,0,1,0}},
        {{0,0,0,0}, {0,1,0,0}, {0,1,0,0}, {1,1,0,0}}
    },
    // L piece
    {
        {{0,0,0,0}, {0,0,1,0}, {1,1,1,0}, {0,0,0,0}},
        {{0,0,0,0}, {0,1,0,0}, {0,1,0,0}, {0,1,1,0}},
        {{0,0,0,0}, {0,0,0,0}, {1,1,1,0}, {1,0,0,0}},
        {{0,0,0,0}, {1,1,0,0}, {0,1,0,0}, {0,1,0,0}}
    }
};

// Colors for each piece type
const int PIECE_COLORS[7] = {
    1,  // I - Cyan
    2,  // O - Yellow
    3,  // T - Magenta
    4,  // S - Green
    5,  // Z - Red
    6,  // J - Blue
    7   // L - White
};

// Class definitions are in game_classes.h
// Implementations below:

TetrisPiece::TetrisPiece(int piece_type, int x, int y) 
    : type(piece_type), x(x), y(y), rotation(0), color(PIECE_COLORS[piece_type]) {}

void TetrisPiece::getShape(int shape[4][4]) const {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            shape[i][j] = PIECES[type][rotation][i][j];
        }
    }
}

void TetrisPiece::rotate() {
    rotation = (rotation + 1) % 4;
}

std::vector<Point> TetrisPiece::getBlocks() const {
    std::vector<Point> blocks;
    int shape[4][4];
    getShape(shape);
    for (int dy = 0; dy < 4; dy++) {
        for (int dx = 0; dx < 4; dx++) {
            if (shape[dy][dx]) {
                blocks.push_back(Point(x + dx, y + dy));
            }
        }
    }
    return blocks;
}

TetrisGame::TetrisGame() 
    : board(HEIGHT, std::vector<int>(WIDTH, 0)),
      current_piece(nullptr),
      next_piece(nullptr),
      score(0),
      lines_cleared(0),
      level(1),
      game_over(false),
      paused(false),
      ai_enabled(false),
      training_mode(false),
      last_score(0),
      last_lines(0),
      fall_delay(0.5),
      last_fall_time(std::chrono::steady_clock::now()),
      last_ai_time(std::chrono::steady_clock::now()) {
    spawnPiece();
}

TetrisGame::~TetrisGame() {
    // Only delete if pointers are different to avoid double deletion
    if (current_piece != nullptr) {
        delete current_piece;
        current_piece = nullptr;
    }
    if (next_piece != nullptr && next_piece != current_piece) {
        delete next_piece;
        next_piece = nullptr;
    }
}

void TetrisGame::spawnPiece() {
    // Delete old current_piece if it exists (shouldn't happen, but safety check)
    if (current_piece != nullptr && current_piece != next_piece) {
        delete current_piece;
    }
    
    if (next_piece == nullptr) {
        int piece_type = rand() % 7;
        next_piece = new TetrisPiece(piece_type);
    }
    
    // Take ownership of next_piece
    current_piece = next_piece;
    next_piece = nullptr;  // Clear next_piece so we don't double-delete
    
    current_piece->x = WIDTH / 2 - 2;
    current_piece->y = 0;
    
    // Check if game over
    if (checkCollision(*current_piece)) {
        game_over = true;
    }
    
    // Generate next piece
    int piece_type = rand() % 7;
    next_piece = new TetrisPiece(piece_type);
}

bool TetrisGame::checkCollision(const TetrisPiece& piece, int dx, int dy) const {
        std::vector<Point> blocks = piece.getBlocks();
        for (const auto& block : blocks) {
            int nx = block.x + dx;
            int ny = block.y + dy;
            // Check boundaries
            if (nx < 0 || nx >= WIDTH || ny >= HEIGHT) {
                return true;
            }
            // Check placed blocks (only check if within board)
            if (ny >= 0 && board[ny][nx] != 0) {
                return true;
        }
    }
    return false;
}

void TetrisGame::placePiece() {
        if (current_piece == nullptr) return;
        
        std::vector<Point> blocks = current_piece->getBlocks();
        for (const auto& block : blocks) {
            if (block.y >= 0 && block.y < HEIGHT && block.x >= 0 && block.x < WIDTH) {
                board[block.y][block.x] = current_piece->color;
            }
        }
        
        // Clear lines
        int cleared = clearLines();
        lines_cleared += cleared;
        
        // Update score
        if (cleared > 0) {
            int points[] = {0, 100, 300, 500, 800};
            score += points[std::min(cleared, 4)] * level;
        }
        
        // Update level (every 10 lines)
        level = lines_cleared / 10 + 1;
        fall_delay = std::max(0.05, 0.5 - (level - 1) * 0.05);
        
    delete current_piece;
    current_piece = nullptr;
}

int TetrisGame::clearLines() {
    std::vector<int> lines_to_clear;
    for (int y = 0; y < HEIGHT; y++) {
        bool full = true;
        for (int x = 0; x < WIDTH; x++) {
            if (board[y][x] == 0) {
                full = false;
                break;
            }
        }
        if (full) {
            lines_to_clear.push_back(y);
        }
    }
    
    // Remove lines from bottom to top
    for (auto it = lines_to_clear.rbegin(); it != lines_to_clear.rend(); ++it) {
        board.erase(board.begin() + *it);
        board.insert(board.begin(), std::vector<int>(WIDTH, 0));
    }
    
    return lines_to_clear.size();
}

bool TetrisGame::movePiece(int dx, int dy) {
    if (current_piece == nullptr) return false;
    
    if (!checkCollision(*current_piece, dx, dy)) {
        current_piece->x += dx;
        current_piece->y += dy;
        return true;
    }
    return false;
}

bool TetrisGame::rotatePiece() {
        if (current_piece == nullptr) return false;
        
        int old_rotation = current_piece->rotation;
        current_piece->rotate();
        
        if (checkCollision(*current_piece)) {
            // Try wall kicks
            int kicks[] = {-1, 1, -2, 2};
            for (int dx : kicks) {
                if (!checkCollision(*current_piece, dx, 0)) {
                    current_piece->x += dx;
                    return true;
                }
            }
            // Rotation failed, revert
            current_piece->rotation = old_rotation;
            return false;
        }
        return true;
}

void TetrisGame::hardDrop() {
    if (current_piece == nullptr) return;
    
    // Safety limit to prevent infinite loop
    int drop_attempts = 0;
    while (movePiece(0, 1) && drop_attempts < HEIGHT * 2) {
        score += 2;  // Bonus points for hard drop
        drop_attempts++;
    }
    if (drop_attempts >= HEIGHT * 2) {
        // Safety: force place piece if stuck
        placePiece();
        spawnPiece();
        return;
    }
    placePiece();
    spawnPiece();
}

void TetrisGame::update() {
    if (game_over || paused) return;
    
    auto current_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        current_time - last_fall_time).count() / 1000.0;
    
    if (elapsed >= fall_delay) {
        if (current_piece == nullptr) {
            spawnPiece();
        } else if (!movePiece(0, 1)) {
            placePiece();
            spawnPiece();
        }
        last_fall_time = current_time;
    }
}

void TetrisGame::executeAIMove(int rotation, int x_pos) {
    if (current_piece == nullptr) return;
    
    // Rotate to desired rotation (with safety limit)
    int rotation_attempts = 0;
    while (current_piece->rotation != rotation && rotation_attempts < 10) {
        rotatePiece();
        rotation_attempts++;
    }
    if (rotation_attempts >= 10) {
        // Debug: rotation stuck
        static int debug_count = 0;
        if (debug_count++ % 100 == 0) {
            // Could log to file or screen, but avoiding I/O for now
        }
    }
    
    // Move to desired x position (with safety limits)
    int target_x = x_pos;
    int move_attempts = 0;
    while (current_piece->x < target_x && movePiece(1, 0) && move_attempts < WIDTH * 2) {
        move_attempts++;
    }
    move_attempts = 0;
    while (current_piece->x > target_x && movePiece(-1, 0) && move_attempts < WIDTH * 2) {
        move_attempts++;
    }
    
    // Hard drop
    hardDrop();
}

std::vector<std::vector<int>> TetrisGame::simulatePlacePiece(const TetrisPiece& piece, int drop_y) const {
        std::vector<std::vector<int>> sim_board = board;
        std::vector<Point> blocks = piece.getBlocks();
        for (const auto& block : blocks) {
            int y = block.y + drop_y - piece.y;
            int x = block.x;
            if (y >= 0 && y < HEIGHT && x >= 0 && x < WIDTH) {
                sim_board[y][x] = piece.color;
            }
        }
    return sim_board;
}

int TetrisGame::simulateClearLines(std::vector<std::vector<int>>& sim_board) const {
        std::vector<int> lines_to_clear;
        for (int y = 0; y < HEIGHT; y++) {
            bool full = true;
            for (int x = 0; x < WIDTH; x++) {
                if (sim_board[y][x] == 0) {
                    full = false;
                    break;
                }
            }
            if (full) {
                lines_to_clear.push_back(y);
            }
        }
        
        for (auto it = lines_to_clear.rbegin(); it != lines_to_clear.rend(); ++it) {
            sim_board.erase(sim_board.begin() + *it);
            sim_board.insert(sim_board.begin(), std::vector<int>(WIDTH, 0));
        }
        
    return lines_to_clear.size();
}

int TetrisGame::getColumnHeight(int x, const std::vector<std::vector<int>>& sim_board) const {
        for (int y = 0; y < HEIGHT; y++) {
            if (sim_board[y][x] != 0) {
                return HEIGHT - y;
            }
        }
    return 0;
}

int TetrisGame::countHoles(const std::vector<std::vector<int>>& sim_board) const {
        int holes = 0;
        for (int x = 0; x < WIDTH; x++) {
            bool block_found = false;
            for (int y = 0; y < HEIGHT; y++) {
                if (sim_board[y][x] != 0) {
                    block_found = true;
                } else if (block_found) {
                    holes++;
                }
            }
        }
    return holes;
}

int TetrisGame::calculateBumpiness(const std::vector<std::vector<int>>& sim_board) const {
        int bumpiness = 0;
        for (int x = 0; x < WIDTH - 1; x++) {
            int h1 = getColumnHeight(x, sim_board);
            int h2 = getColumnHeight(x + 1, sim_board);
            bumpiness += abs(h1 - h2);
        }
    return bumpiness;
}

int TetrisGame::getAggregateHeight(const std::vector<std::vector<int>>& sim_board) const {
    int height = 0;
    for (int x = 0; x < WIDTH; x++) {
        height += getColumnHeight(x, sim_board);
    }
    return height;
}

// AI classes moved to rl_agent.h and rl_agent.cpp

void drawBoard(WINDOW* win, TetrisGame& game, RLAgent* agent = nullptr, ParameterTuner* tuner = nullptr) {
    int height, width;
    getmaxyx(win, height, width);
    (void)height;  // height not used, but required by getmaxyx macro
    
    // Reset attributes to normal
    attrset(0);
    
    // Board dimensions
    int board_x = width / 2 - game.WIDTH / 2 - 1;
    int board_y = 2;
    
    // Draw board border
    mvaddch(board_y - 1, board_x - 1, '+');
    for (int i = 0; i < game.WIDTH * 2; i++) {
        addch('-');
    }
    addch('+');
    
    for (int y = 0; y < game.HEIGHT; y++) {
        mvaddch(board_y + y, board_x - 1, '|');
        mvaddch(board_y + y, board_x + game.WIDTH * 2, '|');
    }
    
    mvaddch(board_y + game.HEIGHT, board_x - 1, '+');
    for (int i = 0; i < game.WIDTH * 2; i++) {
        addch('-');
    }
    addch('+');
    
    // Draw placed blocks
    for (int y = 0; y < game.HEIGHT; y++) {
        for (int x = 0; x < game.WIDTH; x++) {
            if (game.board[y][x] != 0) {
                mvaddstr(board_y + y, board_x + x * 2, "[]");
                mvchgat(board_y + y, board_x + x * 2, 2, A_NORMAL, game.board[y][x], NULL);
            } else {
                // Clear empty spaces to prevent artifacts
                mvaddstr(board_y + y, board_x + x * 2, "  ");
            }
        }
    }
    
    // Draw current piece
    if (game.current_piece) {
        std::vector<Point> blocks = game.current_piece->getBlocks();
        for (const auto& block : blocks) {
            if (block.y >= 0 && block.y < game.HEIGHT && block.x >= 0 && block.x < game.WIDTH) {
                int screen_y = board_y + block.y;
                int screen_x = board_x + block.x * 2;
                if (screen_y >= 0) {
                    mvaddstr(screen_y, screen_x, "[]");
                    mvchgat(screen_y, screen_x, 2, A_NORMAL, game.current_piece->color, NULL);
                }
            }
        }
    }
    
    // Draw next piece preview
    if (game.next_piece) {
        int preview_x = board_x + game.WIDTH * 2 + 5;
        int preview_y = board_y + 2;
        mvaddstr(preview_y - 1, preview_x, "Next:");
        int shape[4][4];
        game.next_piece->getShape(shape);
        for (int dy = 0; dy < 4; dy++) {
            for (int dx = 0; dx < 4; dx++) {
                if (shape[dy][dx]) {
                    mvaddstr(preview_y + dy, preview_x + dx * 2, "[]");
                    mvchgat(preview_y + dy, preview_x + dx * 2, 2, A_NORMAL, game.next_piece->color, NULL);
                }
            }
        }
    }
    
    // Draw score and stats
    int info_x = board_x - 15;
    int info_y = board_y + 2;
    char score_str[50];
    snprintf(score_str, sizeof(score_str), "Score: %d", game.score);
    mvaddstr(info_y, info_x, score_str);
    snprintf(score_str, sizeof(score_str), "Lines: %d", game.lines_cleared);
    mvaddstr(info_y + 1, info_x, score_str);
    snprintf(score_str, sizeof(score_str), "Level: %d", game.level);
    mvaddstr(info_y + 2, info_x, score_str);
    if (game.ai_enabled) {
        if (game.training_mode) {
            mvaddstr(info_y + 3, info_x, "AI: TRAINING");
            mvchgat(info_y + 3, info_x, 13, A_BOLD | A_REVERSE, 1, NULL);
        } else {
            mvaddstr(info_y + 3, info_x, "AI: ON");
            mvchgat(info_y + 3, info_x, 6, A_BOLD | A_REVERSE, 0, NULL);
        }
    } else {
        mvaddstr(info_y + 3, info_x, "AI: OFF");
    }
    
    // Draw controls
    int controls_y = board_y + game.HEIGHT + 2;
    mvaddstr(controls_y, board_x, "Controls:");
    mvaddstr(controls_y + 1, board_x, "Left/Right: Move");
    mvaddstr(controls_y + 2, board_x, "Up: Rotate");
    mvaddstr(controls_y + 3, board_x, "Down: Soft Drop");
    mvaddstr(controls_y + 4, board_x, "Space: Hard Drop");
    mvaddstr(controls_y + 5, board_x, "A: Toggle AI");
    mvaddstr(controls_y + 6, board_x, "T: Training Mode");
    mvaddstr(controls_y + 7, board_x, "P: Pause  Q: Quit");
    
    // Draw training stats if in training mode
    if (game.training_mode && agent != nullptr) {
        int stats_y = controls_y + 9;
        char stats_str[250];
        const char* model_status = agent->model_loaded ? "[LOADED]" : "[NEW]";
        snprintf(stats_str, sizeof(stats_str), 
                "Training: Games=%d Episodes=%d Best=%d Avg=%.0f Epsilon=%.3f Buffer=%zu %s", 
                agent->total_games, agent->training_episodes, 
                agent->best_score, agent->average_score, agent->epsilon, agent->replay_buffer.size(), model_status);
        mvaddstr(stats_y, board_x, stats_str);
        
        // Display epsilon-score relationship tracking
        int epsilon_track_y = stats_y + 3;
        char epsilon_track_str[200];
        double epsilon_change = agent->epsilon - agent->last_epsilon;
        const char* epsilon_trend = "";
        if (epsilon_change > 0.001) {
            epsilon_trend = "↑ (increasing)";
        } else if (epsilon_change < -0.001) {
            epsilon_trend = "↓ (decreasing)";
        } else {
            epsilon_trend = "→ (stable)";
        }
        snprintf(epsilon_track_str, sizeof(epsilon_track_str),
                "Epsilon-Score: Avg=%.0f Eps=%.3f %s | Inc=%d Dec=%d | Change=%.1f%%",
                agent->average_score, agent->epsilon, epsilon_trend,
                agent->epsilon_increase_count, agent->epsilon_decrease_count,
                agent->epsilon_change_reason);
        mvaddstr(epsilon_track_y, board_x, epsilon_track_str);
        
        // Draw weight changes on next line
        if (agent->training_episodes > 0) {
            int weight_y = stats_y + 1;
            std::string weight_stats = agent->q_network.getWeightStatsString(
                agent->training_episodes, agent->last_batch_error);
            mvaddstr(weight_y, board_x, weight_stats.c_str());
        }
        
        // Draw parameter tuning info
        if (tuner != nullptr && agent->training_episodes > 0) {
            int tuner_y = stats_y + 2;
            char tuner_str[250];
            snprintf(tuner_str, sizeof(tuner_str), 
                    "Tuner: LR=%.4f Gamma=%.3f EpsDec=%.4f EpsMin=%.3f Set=%d/%zu",
                    agent->learning_rate, agent->gamma, agent->epsilon_decay, 
                    agent->epsilon_min, tuner->current_param_set_index, tuner->parameter_sets.size());
            mvaddstr(tuner_y, board_x, tuner_str);
        }
    }
    
    // Draw game over or pause message
    if (game.game_over) {
        const char* msg = "GAME OVER!";
        mvaddstr(board_y + game.HEIGHT / 2, board_x + game.WIDTH - 5, msg);
        mvchgat(board_y + game.HEIGHT / 2, board_x + game.WIDTH - 5, 10, A_BOLD, 1, NULL);
    } else if (game.paused) {
        const char* msg = "PAUSED";
        mvaddstr(board_y + game.HEIGHT / 2, board_x + game.WIDTH - 3, msg);
        mvchgat(board_y + game.HEIGHT / 2, board_x + game.WIDTH - 3, 6, A_BOLD, 0, NULL);
    }
    
    // Reset attributes to normal after drawing
    attrset(0);
}

void initColors() {
    start_color();
    init_pair(1, COLOR_CYAN, COLOR_BLACK);      // I piece
    init_pair(2, COLOR_YELLOW, COLOR_BLACK);    // O piece
    init_pair(3, COLOR_MAGENTA, COLOR_BLACK);    // T piece
    init_pair(4, COLOR_GREEN, COLOR_BLACK);     // S piece
    init_pair(5, COLOR_RED, COLOR_BLACK);       // Z piece
    init_pair(6, COLOR_BLUE, COLOR_BLACK);      // J piece
    init_pair(7, COLOR_WHITE, COLOR_BLACK);    // L piece
}

int main(int argc, char* argv[]) {
    // Parse command line arguments (before ncurses initialization)
    std::string model_file = "tetris_model.txt";
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" || arg == "-m") {
            if (i + 1 < argc) {
                model_file = argv[++i];
            } else {
                std::cerr << "Error: --model requires a filename\n";
                std::cerr << "Usage: " << argv[0] << " [--model|-m <filename>] [--help|-h]\n";
                return 1;
            }
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Tetris Game with Reinforcement Learning AI\n";
            std::cout << "==========================================\n\n";
            std::cout << "Usage: " << argv[0] << " [OPTIONS]\n\n";
            std::cout << "Options:\n";
            std::cout << "  --model, -m <filename>  Load neural network model from specified file\n";
            std::cout << "                          (default: tetris_model.txt)\n";
            std::cout << "  --help, -h              Show this help message\n\n";
            std::cout << "Examples:\n";
            std::cout << "  " << argv[0] << "                    # Use default model (tetris_model.txt)\n";
            std::cout << "  " << argv[0] << " -m tetris_model_best.txt  # Load best model\n";
            std::cout << "  " << argv[0] << " --help              # Show this help\n\n";
            std::cout << "Controls:\n";
            std::cout << "  Left/Right Arrow  - Move piece left/right\n";
            std::cout << "  Up Arrow          - Rotate piece\n";
            std::cout << "  Down Arrow        - Soft drop\n";
            std::cout << "  Space             - Hard drop\n";
            std::cout << "  A                 - Toggle AI\n";
            std::cout << "  T                 - Toggle training mode\n";
            std::cout << "  P                 - Pause/Unpause\n";
            std::cout << "  Q                 - Quit\n";
            return 0;
        }
    }
    
    // Initialize random seed
    srand(time(nullptr));
    
    // Setup ncurses
    initscr();
    curs_set(0);      // Hide cursor
    nodelay(stdscr, TRUE);  // Non-blocking input
    timeout(1);       // Refresh timeout
    keypad(stdscr, TRUE);   // Enable special keys
    noecho();         // Don't echo input
    initColors();
    
    TetrisGame game;
    RLAgent agent(model_file);  // Load from specified model file
    ParameterTuner tuner;
    
    // Auto-start in training mode for continuous learning
    game.training_mode = true;
    game.ai_enabled = true;
    
    // Apply initial parameter set
    ParameterSet initial_params = tuner.getNextParameterSet();
    tuner.applyParameters(initial_params, agent);
    
    std::vector<double> last_state;
    int last_action_rot = 0;
    int last_action_x = 0;
    
    // Game loop with debugging
    int loop_count = 0;
    auto last_debug_time = std::chrono::steady_clock::now();
    int stuck_detection_count = 0;
    
    debugLog("Program started");
    
    while (true) {
        loop_count++;
        
        // Debug: Check for stuck loop (CPU spinning)
        auto current_debug_time = std::chrono::steady_clock::now();
        auto debug_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            current_debug_time - last_debug_time).count();
        
        if (debug_elapsed > 5000) {  // More than 5 seconds without progress
            stuck_detection_count++;
            std::stringstream ss;
            ss << "Stuck detection #" << stuck_detection_count << " - Loop: " << loop_count 
               << " Games: " << agent.total_games << " Episodes: " << agent.training_episodes;
            debugLog(ss.str());
            
            if (stuck_detection_count > 3) {
                debugLog("FORCE UPDATE: Breaking potential deadlock");
                // Force game update to break potential deadlock
                if (!game.game_over && !game.paused) {
                    game.update();
                }
                stuck_detection_count = 0;
            }
            last_debug_time = current_debug_time;
        }
        
        // Periodic debug logging
        if (loop_count % 1000 == 0) {
            std::stringstream ss;
            ss << "Loop: " << loop_count << " Games: " << agent.total_games 
               << " Episodes: " << agent.training_episodes 
               << " GameOver: " << (game.game_over ? "YES" : "NO")
               << " Paused: " << (game.paused ? "YES" : "NO")
               << " AI: " << (game.ai_enabled ? "ON" : "OFF");
            debugLog(ss.str());
            last_debug_time = current_debug_time;
            stuck_detection_count = 0;
        }
        // Clear screen and reset attributes
        erase();
        attrset(0);  // Reset all attributes to normal
        
        // Handle input
        int key = getch();
        
        if (key == 'q' || key == 'Q') {
            break;
        } else if (key == 'p' || key == 'P') {
            game.paused = !game.paused;
        } else if (key == 'a' || key == 'A') {
            game.ai_enabled = !game.ai_enabled;
        } else if (key == 't' || key == 'T') {
            game.training_mode = !game.training_mode;
            if (game.training_mode) {
                game.ai_enabled = true;  // Auto-enable AI in training mode
            }
        } else if (!game.game_over && !game.paused && !game.ai_enabled) {
            if (key == KEY_LEFT) {
                game.movePiece(-1, 0);
            } else if (key == KEY_RIGHT) {
                game.movePiece(1, 0);
            } else if (key == KEY_DOWN) {
                if (game.movePiece(0, 1)) {
                    game.score += 1;  // Bonus for soft drop
                }
            } else if (key == KEY_UP) {
                game.rotatePiece();
            } else if (key == ' ') {
                game.hardDrop();
            }
        }
        
        // RL Agent logic
        if (game.ai_enabled && !game.game_over && !game.paused && game.current_piece != nullptr) {
            auto current_time = std::chrono::steady_clock::now();
            auto ai_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                current_time - game.last_ai_time).count();
            
            // Execute AI move every 100ms (with timeout protection)
            if (ai_elapsed >= 100) {
                // Safety: Limit AI computation time to prevent CPU spinning
                auto ai_start_time = std::chrono::steady_clock::now();
                
                // Extract current state
                std::vector<double> current_state = agent.extractState(game);
                
                // Find best move (with timeout check)
                RLAgent::Move best_move = agent.findBestMove(game, game.training_mode);
                
                // Check if AI computation took too long
                auto ai_compute_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - ai_start_time).count();
                if (ai_compute_time > 1000) {
                    // AI computation took more than 1 second - skip this move to prevent CPU spinning
                    game.last_ai_time = current_time;
                    continue;
                }
                
                game.executeAIMove(best_move.rotation, best_move.x);
                
                // Collect experience for training
                if (game.training_mode && last_state.size() > 0) {
                    // Calculate reward (simplified and clearer reward shaping)
                    double reward = 0.0;
                    int score_diff = game.score - game.last_score;
                    int lines_diff = game.lines_cleared - game.last_lines;
                    
                    // Primary rewards (most important) - more generous
                    reward += lines_diff * 5.0;       // Increased reward for clearing lines (main objective)
                    reward += score_diff * 0.1;        // Increased reward for score increases
                    
                    // Survival bonus (encourage staying alive) - more significant
                    if (!game.game_over) {
                        reward += 1.0;  // Increased survival bonus
                    }
                    
                    // Game over penalty
                    if (game.game_over) {
                        reward -= 50.0;  // Increased penalty for losing (more significant)
                    }
                    
                    // State quality penalties (prevent bad states) - less harsh
                    int aggregate_height = game.getAggregateHeight(game.board);
                    reward -= aggregate_height * 0.05;  // Reduced penalty for high stacks
                    
                    int holes = game.countHoles(game.board);
                    reward -= holes * 0.3;  // Reduced penalty for creating holes
                    
                    int bumpiness = game.calculateBumpiness(game.board);
                    reward -= bumpiness * 0.02;  // Reduced penalty for uneven surface
                    
                    // Bonus for keeping board low (encourage good play)
                    int max_height = 0;
                    for (int x = 0; x < game.WIDTH; x++) {
                        int h = game.getColumnHeight(x, game.board);
                        if (h > max_height) max_height = h;
                    }
                    if (max_height < 10) {
                        reward += 0.5;  // Bonus for keeping board low
                    }
                    
                    // Store experience
                    Experience exp;
                    exp.state = last_state;
                    exp.action_rotation = last_action_rot;
                    exp.action_x = last_action_x;
                    exp.reward = reward;
                    exp.next_state = current_state;
                    exp.done = game.game_over;
                    
                    agent.addExperience(exp);
                    
                    // Train periodically
                    if (agent.replay_buffer.size() >= RLAgent::BATCH_SIZE) {
                        auto train_start = std::chrono::steady_clock::now();
                        agent.train();
                        auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - train_start).count();
                        if (train_time > 500) {
                            std::stringstream ss;
                            ss << "Slow training: " << train_time << "ms";
                            debugLog(ss.str());
                        }
                        
                        // Record metrics for parameter tuning
                        tuner.recordError(agent.last_batch_error);
                        tuner.recordEpsilon(agent.epsilon);
                        
                        // Check if we should test new parameters
                        if (tuner.shouldTestNewParameters()) {
                            debugLog("Testing new parameter set");
                            ParameterSet new_params = tuner.getNextParameterSet();
                            tuner.applyParameters(new_params, agent);
                            tuner.resetForNewParameters();
                            
                            // Log parameter change
                            std::ofstream logfile("debug.log", std::ios::app);
                            if (logfile.is_open()) {
                                logfile << "[TUNER] Switched to new parameters: LR=" << new_params.learning_rate
                                        << " Gamma=" << new_params.gamma
                                        << " EpsDecay=" << new_params.epsilon_decay
                                        << " EpsMin=" << new_params.epsilon_min << std::endl;
                            }
                        }
                    }
                }
                
                // Update last state/action
                last_state = current_state;
                last_action_rot = best_move.rotation;
                last_action_x = best_move.x;
                game.last_score = game.score;
                game.last_lines = game.lines_cleared;
                
                game.last_ai_time = current_time;
                
                // Auto-restart in training mode (handled in main loop)
            }
        }
        
        // Auto-restart in training mode when game over
        if (game.training_mode && game.game_over) {
            debugLog("Game over - restarting");
            // Update training statistics
            agent.total_games++;
            if (game.score > agent.best_score) {
                int old_best = agent.best_score;
                agent.best_score = game.score;
                
                // Save best model to separate file
                std::string best_model_file = "tetris_model_best.txt";
                agent.saveModelToFile(best_model_file);
                
                // Log best score achievement
                std::ofstream logfile("debug.log", std::ios::app);
                if (logfile.is_open()) {
                    logfile << "[BEST] New best score: " << old_best << " -> " << game.score 
                            << " | Saved to " << best_model_file << std::endl;
                }
                
                debugLog("New best score! Model saved to tetris_model_best.txt");
            }
            
            // Update running average (simple moving average of last N games)
            agent.recent_scores_sum += game.score;
            if (agent.total_games <= RLAgent::RECENT_SCORES_COUNT) {
                agent.average_score = agent.recent_scores_sum / (double)agent.total_games;
            } else {
                // For sliding window, we'd need to track individual scores
                // For simplicity, use exponential moving average
                agent.average_score = agent.average_score * 0.99 + game.score * 0.01;
            }
            
            // Record score for parameter tuning
            tuner.recordScore(game.score);
            
            // Update epsilon based on performance (adaptive decay)
            agent.updateEpsilonBasedOnPerformance();
            
            // Save model periodically
            if (agent.training_episodes % 100 == 0) {
                agent.saveModel();
            }
            
            // Reset game immediately
            game.~TetrisGame();
            new (&game) TetrisGame();
            game.training_mode = true;
            game.ai_enabled = true;
            last_state.clear();
            
            // Small delay to show game over briefly and allow screen refresh
            napms(100);
            
            // Force screen refresh after restart to prevent black screen
            erase();
            attrset(0);  // Reset attributes
        }
        
        // Update game (only if not game over in training mode, as we'll restart immediately)
        if (!(game.training_mode && game.game_over)) {
            game.update();
        }
        
        // Draw everything
        drawBoard(stdscr, game, &agent, &tuner);
        
        // Refresh screen and ensure cursor is hidden
        refresh();
        curs_set(0);  // Keep cursor hidden
        
        // Small delay to prevent excessive CPU usage
        // Shorter delay in training mode for faster training
        napms(game.training_mode ? 50 : 200);
    }
    
    // Save model on exit if training
    if (game.training_mode) {
        agent.saveModel();
    }
    
    endwin();
    return 0;
}

