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
#include <cstring>
#include <deque>
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

// Helper function to update string only if different (prevents flickering)
static void updateStringIfChanged(int y, int x, const std::string& new_str, std::string& prev_str) {
    if (new_str != prev_str) {
        // Clear old string first if it was longer
        if (prev_str.length() > new_str.length()) {
            mvaddstr(y, x, std::string(prev_str.length(), ' ').c_str());
        }
        mvaddstr(y, x, new_str.c_str());
        prev_str = new_str;
    }
}

// Global static variables for screen state tracking (accessible from both drawBoard and main)
// Note: Board, current piece, and next piece are always redrawn for smooth movement
static int prev_score_global = -1;
static int prev_lines_global = -1;
static int prev_level_global = -1;
static bool prev_ai_enabled_global = false;
static bool prev_training_mode_global = false;
static bool prev_game_over_global = false;
static bool prev_paused_global = false;
static std::chrono::steady_clock::time_point game_over_start_time;
static bool game_over_timer_active = false;

void drawBoard(WINDOW* win, TetrisGame& game, RLAgent* agent = nullptr, ParameterTuner* tuner = nullptr, bool score_graph_visible = false, bool stats_visible = false) {
    int height, width;
    getmaxyx(win, height, width);
    (void)height;  // height not used, but required by getmaxyx macro
    
    // Static variables to store previous strings (prevents flickering)
    static std::string prev_stats_str1 = "";
    static std::string prev_stats_str2 = "";
    static std::string prev_epsilon_str1 = "";
    static std::string prev_epsilon_str2 = "";
    static std::string prev_weight_stats = "";
    static std::vector<std::string> prev_weight_lines;
    static std::string prev_sat_str1 = "";
    static std::string prev_sat_str2 = "";
    static int prev_sat_color = -1;
    static std::string prev_tuner_str = "";
    
    // Reset attributes to normal
    attrset(0);
    
    // Board dimensions
    int board_x = width / 2 - game.WIDTH / 2 - 1;
    int board_y = 2;
    
    // Draw board border (only once)
    static bool border_drawn = false;
    if (!border_drawn) {
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
        border_drawn = true;
    }
    
    // ALWAYS redraw game board section (for smooth piece movement)
    // Draw placed blocks
    for (int y = 0; y < game.HEIGHT; y++) {
        for (int x = 0; x < game.WIDTH; x++) {
            if (game.board[y][x] != 0) {
                mvaddstr(board_y + y, board_x + x * 2, "[]");
                mvchgat(board_y + y, board_x + x * 2, 2, A_NORMAL, game.board[y][x], NULL);
            } else {
                // Clear empty spaces
                mvaddstr(board_y + y, board_x + x * 2, "  ");
            }
        }
    }
    
    // Draw current piece (always redraw for smooth movement)
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
    
    // Draw next piece preview (always redraw for smooth display)
    // Position next piece on left side if score graph is visible, otherwise on right side
    static int prev_preview_x = -1;
    static int prev_preview_y = -1;
    static bool prev_score_graph_visible = false;
    
    int preview_x, preview_y;
    if (score_graph_visible) {
        // Place next piece on the left side (near score info) when graph is visible
        preview_x = board_x - 15;
        preview_y = board_y + 9;  // 3 lines down from previous position (was board_y + 6, now board_y + 9)
    } else {
        // Place next piece on the right side when graph is not visible
        preview_x = board_x + game.WIDTH * 2 + 5;
        preview_y = board_y + 2;
    }
    
    // Clear old position if position changed (due to graph toggle)
    if (prev_preview_x != -1 && prev_preview_y != -1 && 
        (prev_preview_x != preview_x || prev_preview_y != preview_y || prev_score_graph_visible != score_graph_visible)) {
        // Clear old "Next:" label
        mvaddstr(prev_preview_y - 1, prev_preview_x, "     ");
        // Clear old next piece area
        for (int dy = 0; dy < 4; dy++) {
            for (int dx = 0; dx < 4; dx++) {
                mvaddstr(prev_preview_y + dy, prev_preview_x + dx * 2, "  ");
            }
        }
    }
    
    // Update previous position
    prev_preview_x = preview_x;
    prev_preview_y = preview_y;
    prev_score_graph_visible = score_graph_visible;
    
    // Always redraw "Next:" label (in case it was cleared)
    mvaddstr(preview_y - 1, preview_x, "Next:");
    
    // Always clear and redraw next piece area
    for (int dy = 0; dy < 4; dy++) {
        for (int dx = 0; dx < 4; dx++) {
            mvaddstr(preview_y + dy, preview_x + dx * 2, "  ");
        }
    }
    
    // Draw next piece if it exists
    if (game.next_piece) {
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
    
    // Draw score and stats (only update if changed)
    int info_x = board_x - 15;
    int info_y = board_y + 2;
    
    // Clear and redraw score field when it changes (to avoid overlapping text)
    if (prev_score_global != game.score) {
        char score_str[50];
        snprintf(score_str, sizeof(score_str), "Score: %d", game.score);
        // Clear enough space for long numbers (e.g., "Score: 999999")
        mvaddstr(info_y, info_x, "Score:        ");
        mvaddstr(info_y, info_x, score_str);
        prev_score_global = game.score;
    }
    
    if (prev_lines_global != game.lines_cleared) {
        char lines_str[50];
        snprintf(lines_str, sizeof(lines_str), "Lines: %d", game.lines_cleared);
        mvaddstr(info_y + 1, info_x, lines_str);
        prev_lines_global = game.lines_cleared;
    }
    
    if (prev_level_global != game.level) {
        char level_str[50];
        snprintf(level_str, sizeof(level_str), "Level: %d", game.level);
        mvaddstr(info_y + 2, info_x, level_str);
        prev_level_global = game.level;
    }
    
    if (prev_ai_enabled_global != game.ai_enabled || prev_training_mode_global != game.training_mode) {
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
        prev_ai_enabled_global = game.ai_enabled;
        prev_training_mode_global = game.training_mode;
    }
    
    // Draw controls (only once)
    static bool controls_drawn = false;
    int controls_y = board_y + game.HEIGHT + 2;
    if (!controls_drawn) {
        mvaddstr(controls_y, board_x, "Controls:");
        mvaddstr(controls_y + 1, board_x, "Left/Right: Move");
        mvaddstr(controls_y + 2, board_x, "Up: Rotate");
        mvaddstr(controls_y + 3, board_x, "Down: Soft Drop");
        mvaddstr(controls_y + 4, board_x, "Space: Hard Drop");
        mvaddstr(controls_y + 5, board_x, "A: Toggle AI");
        mvaddstr(controls_y + 6, board_x, "T: Training Mode");
        mvaddstr(controls_y + 7, board_x, "S: Score Graph  V: Stats");
        mvaddstr(controls_y + 8, board_x, "P: Pause  Q: Quit");
        controls_drawn = true;
    }
    
    // Clear stats area if stats are hidden (to prevent leftover text)
    static bool prev_stats_visible = false;
    if (!stats_visible && prev_stats_visible) {
        // Clear the stats area (from controls_y + 9 to bottom of screen)
        int stats_y = controls_y + 9;
        int height, width;
        getmaxyx(win, height, width);
        for (int y = stats_y; y < height - 1; y++) {
            for (int x = board_x; x < board_x + 80; x++) {
                if (x >= 0 && x < width && y >= 0 && y < height) {
                    mvaddch(y, x, ' ');
                }
            }
        }
    }
    prev_stats_visible = stats_visible;
    
    // Draw training stats if in training mode and stats are visible
    if (game.training_mode && agent != nullptr && stats_visible) {
        int stats_y = controls_y + 9;
        char stats_str[250];
        const char* model_status = agent->model_loaded ? "[LOADED]" : "[NEW]";
        snprintf(stats_str, sizeof(stats_str), 
                "Training: Games=%d Episodes=%d Best=%d Avg=%.0f", 
                agent->total_games, agent->training_episodes, 
                agent->best_score, agent->average_score);
        updateStringIfChanged(stats_y, board_x, std::string(stats_str), prev_stats_str1);
        
        char stats_str2[200];
        snprintf(stats_str2, sizeof(stats_str2),
                "Epsilon=%.3f Buffer=%zu %s",
                agent->epsilon, agent->replay_buffer.size(), model_status);
        updateStringIfChanged(stats_y + 1, board_x, std::string(stats_str2), prev_stats_str2);
        
        // Display epsilon-score relationship tracking
        int epsilon_track_y = stats_y + 2;
        char epsilon_track_str1[200];
        char epsilon_track_str2[200];
        double epsilon_change = agent->epsilon - agent->last_epsilon;
        const char* epsilon_trend = "";
        if (epsilon_change > 0.001) {
            epsilon_trend = "↑ (increasing)";
        } else if (epsilon_change < -0.001) {
            epsilon_trend = "↓ (decreasing)";
        } else {
            epsilon_trend = "→ (stable)";
        }
        snprintf(epsilon_track_str1, sizeof(epsilon_track_str1),
                "Epsilon-Score: Avg=%.0f Eps=%.3f %s",
                agent->average_score, agent->epsilon, epsilon_trend);
        snprintf(epsilon_track_str2, sizeof(epsilon_track_str2),
                "Inc=%d Dec=%d | Change=%.1f%%",
                agent->epsilon_increase_count, agent->epsilon_decrease_count,
                agent->epsilon_change_reason);
        updateStringIfChanged(epsilon_track_y, board_x, std::string(epsilon_track_str1), prev_epsilon_str1);
        updateStringIfChanged(epsilon_track_y + 1, board_x, std::string(epsilon_track_str2), prev_epsilon_str2);
        
        // Draw weight changes and saturation metrics after epsilon info
        if (agent->training_episodes > 0) {
            int weight_y = epsilon_track_y + 2;  // Start after epsilon info
            std::string weight_stats = agent->q_network.getWeightStatsString(
                agent->training_episodes, agent->last_batch_error);
            
            // Split weight stats on multiple lines if it contains newline
            // Only update if different
            if (weight_stats != prev_weight_stats) {
                std::stringstream ss(weight_stats);
                std::string line;
                int current_y = weight_y;
                size_t line_num = 0;
                while (std::getline(ss, line, '\n')) {
                    // Clear previous line if it was longer
                    if (line_num < prev_weight_lines.size() && prev_weight_lines[line_num].length() > line.length()) {
                        mvaddstr(current_y, board_x, std::string(prev_weight_lines[line_num].length(), ' ').c_str());
                    }
                    mvaddstr(current_y, board_x, line.c_str());
                    if (line_num >= prev_weight_lines.size()) {
                        prev_weight_lines.push_back(line);
                    } else {
                        prev_weight_lines[line_num] = line;
                    }
                    current_y++;
                    line_num++;
                }
                // Clear any extra previous lines
                while (line_num < prev_weight_lines.size()) {
                    mvaddstr(weight_y + static_cast<int>(line_num), board_x, std::string(prev_weight_lines[line_num].length(), ' ').c_str());
                    line_num++;
                }
                if (line_num < prev_weight_lines.size()) {
                    prev_weight_lines.resize(line_num);
                }
                prev_weight_stats = weight_stats;
            }
            
            // Calculate current_y for saturation display
            int current_y = weight_y;
            std::stringstream ss(weight_stats);
            std::string line;
            while (std::getline(ss, line, '\n')) {
                current_y++;
            }
            
            // Draw saturation warning if saturation is too high
            NeuralNetwork::SaturationMetrics sat = agent->q_network.calculateSaturation();
            int sat_y = current_y;
            char sat_str1[200];
            char sat_str2[200];
            const char* sat_status = "";
            int sat_color = 0;
            
            // Check if any layer has high saturation (> 50%)
            double max_sat = std::max({sat.weights1_saturation, sat.bias1_saturation, 
                                      sat.weights2_saturation, sat.bias2_saturation});
            
            if (max_sat > 80.0) {
                sat_status = "⚠️ HIGH SATURATION!";
                sat_color = 1;  // Red
            } else if (max_sat > 50.0) {
                sat_status = "⚠️ Medium Saturation";
                sat_color = 3;  // Yellow/Magenta
            } else {
                sat_status = "✓ Low Saturation";
                sat_color = 2;  // Green
            }
            
            snprintf(sat_str1, sizeof(sat_str1),
                "Saturation: W1=%.1f%% B1=%.1f%% W2=%.1f%% B2=%.1f%%",
                sat.weights1_saturation, sat.bias1_saturation,
                sat.weights2_saturation, sat.bias2_saturation);
            snprintf(sat_str2, sizeof(sat_str2), "Status: %s", sat_status);
            
            std::string sat_str1_str(sat_str1);
            std::string sat_str2_str(sat_str2);
            
            updateStringIfChanged(sat_y, board_x, sat_str1_str, prev_sat_str1);
            updateStringIfChanged(sat_y + 1, board_x, sat_str2_str, prev_sat_str2);
            
            // Update color only if changed
            if (sat_color != prev_sat_color) {
                mvchgat(sat_y + 1, board_x, sat_str2_str.length(), A_BOLD, sat_color, NULL);
                prev_sat_color = sat_color;
            }
        }
        
        // Draw parameter tuning info
        if (tuner != nullptr && agent->training_episodes > 0) {
            int tuner_y = stats_y + 2;
            char tuner_str[250];
            snprintf(tuner_str, sizeof(tuner_str), 
                    "Tuner: LR=%.4f Gamma=%.3f EpsDec=%.4f EpsMin=%.3f Set=%d/%zu",
                    agent->learning_rate, agent->gamma, agent->epsilon_decay, 
                    agent->epsilon_min, tuner->current_param_set_index, tuner->parameter_sets.size());
            updateStringIfChanged(tuner_y, board_x, std::string(tuner_str), prev_tuner_str);
        }
    }
    
    // Draw game over or pause message (positioned 1 line above "Controls:")
    // Use the same controls_y that was calculated earlier
    int msg_y = board_y + game.HEIGHT + 1;  // 1 line above Controls (controls_y is board_y + HEIGHT + 2)
    int msg_x = board_x;  // Same X position as Controls
    
    // Always clear message area first (clear enough space)
    mvaddstr(msg_y, msg_x, "                    ");  // Clear larger area
    
    // Handle game over with 200ms visibility timer (non-blocking)
    if (game.game_over && !prev_game_over_global) {
        // Game over just started - start timer
        game_over_start_time = std::chrono::steady_clock::now();
        game_over_timer_active = true;
    }
    
    // Check if game over timer is still active (200ms duration)
    bool show_game_over = false;
    if (game.game_over && game_over_timer_active) {
        auto current_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            current_time - game_over_start_time).count();
        
        if (elapsed < 200) {
            // Still within 200ms window - show message
            show_game_over = true;
        } else {
            // 200ms elapsed - hide message but keep timer active flag until game restarts
            game_over_timer_active = false;
        }
    }
    
    // Always redraw message to ensure visibility
    if (show_game_over) {
        const char* msg = "GAME OVER!";
        mvaddstr(msg_y, msg_x, msg);
        mvchgat(msg_y, msg_x, strlen(msg), A_BOLD | A_REVERSE, 1, NULL);  // Bold + reverse for visibility
    } else if (game.paused) {
        const char* msg = "PAUSED";
        mvaddstr(msg_y, msg_x, msg);
        mvchgat(msg_y, msg_x, strlen(msg), A_BOLD | A_REVERSE, 0, NULL);  // Bold + reverse for visibility
    }
    
    // Update previous state
    prev_game_over_global = game.game_over;
    prev_paused_global = game.paused;
    
    // Reset timer when game restarts (game_over goes from true to false)
    if (!game.game_over && prev_game_over_global) {
        game_over_timer_active = false;
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

// Score history for graph
std::deque<int> score_history;
bool show_score_graph = false;
bool show_stats = false;  // Toggle for showing/hiding training stats
const int MAX_HISTORY = 200;  // Maximum scores to display in graph

void drawScoreGraph(RLAgent* agent) {
    if (!show_score_graph) return;
    
    // Show message if no data yet
    if (score_history.empty()) {
        int width;
        int height_unused;
        getmaxyx(stdscr, height_unused, width);
        (void)height_unused;  // Suppress unused variable warning
        int graph_x = width - 60;
        int graph_y = 2;
        int graph_width = 55;
        int graph_height = 20;
        
        // Draw graph border even if empty
        mvaddch(graph_y, graph_x, '+');
        mvaddch(graph_y, graph_x + graph_width, '+');
        mvaddch(graph_y + graph_height, graph_x, '+');
        mvaddch(graph_y + graph_height, graph_x + graph_width, '+');
        
        for (int i = 1; i < graph_width; i++) {
            mvaddch(graph_y, graph_x + i, '-');
            mvaddch(graph_y + graph_height, graph_x + i, '-');
        }
        for (int i = 1; i < graph_height; i++) {
            mvaddch(graph_y + i, graph_x, '|');
            mvaddch(graph_y + i, graph_x + graph_width, '|');
        }
        
        mvaddstr(graph_y, graph_x + 2, "Score History (Press S to toggle)");
        mvaddstr(graph_y + graph_height / 2, graph_x + 15, "Waiting for data...");
        return;
    }
    
    int width;
    int height;
    getmaxyx(stdscr, height, width);
    
    // Graph dimensions
    int graph_x = width - 60;  // Right side of screen
    int graph_y = 2;
    int graph_width = 55;
    int graph_height = 20;
    
    // Clear entire graph area before redrawing (including border, labels, and stats below)
    for (int y = graph_y - 1; y <= graph_y + graph_height + 3; y++) {
        for (int x = graph_x - 15; x <= graph_x + graph_width + 1; x++) {
            if (x >= 0 && x < width && y >= 0 && y < height) {
                mvaddch(y, x, ' ');
            }
        }
    }
    
    // Draw graph border
    mvaddch(graph_y, graph_x, '+');
    mvaddch(graph_y, graph_x + graph_width, '+');
    mvaddch(graph_y + graph_height, graph_x, '+');
    mvaddch(graph_y + graph_height, graph_x + graph_width, '+');
    
    for (int i = 1; i < graph_width; i++) {
        mvaddch(graph_y, graph_x + i, '-');
        mvaddch(graph_y + graph_height, graph_x + i, '-');
    }
    for (int i = 1; i < graph_height; i++) {
        mvaddch(graph_y + i, graph_x, '|');
        mvaddch(graph_y + i, graph_x + graph_width, '|');
    }
    
    // Title
    mvaddstr(graph_y, graph_x + 2, "Score History (Press S to toggle)");
    
    if (score_history.size() < 2) {
        mvaddstr(graph_y + graph_height / 2, graph_x + 15, "Not enough data");
        return;
    }
    
    // Find min/max for scaling
    int min_score = *std::min_element(score_history.begin(), score_history.end());
    int max_score = *std::max_element(score_history.begin(), score_history.end());
    
    // Add some padding
    int range = max_score - min_score;
    if (range == 0) range = 1;
    min_score = std::max(0, min_score - range / 10);
    max_score = max_score + range / 10;
    range = max_score - min_score;
    
    // Draw Y-axis labels
    char label[20];
    snprintf(label, sizeof(label), "%d", max_score);
    mvaddstr(graph_y + 1, graph_x - strlen(label) - 1, label);
    snprintf(label, sizeof(label), "%d", min_score);
    mvaddstr(graph_y + graph_height - 1, graph_x - strlen(label) - 1, label);
    snprintf(label, sizeof(label), "%d", (min_score + max_score) / 2);
    mvaddstr(graph_y + graph_height / 2, graph_x - strlen(label) - 1, label);
    
    // Draw graph points and lines
    int num_points = std::min((int)score_history.size(), graph_width - 2);
    int start_idx = std::max(0, (int)score_history.size() - num_points);
    
    std::vector<int> graph_y_positions;
    for (int i = 0; i < num_points; i++) {
        int score = score_history[start_idx + i];
        int y_pos = graph_y + graph_height - 1 - 
                   ((score - min_score) * (graph_height - 2) / range);
        y_pos = std::max(graph_y + 1, std::min(graph_y + graph_height - 1, y_pos));
        graph_y_positions.push_back(y_pos);
    }
    
    // Draw lines connecting points
    for (int i = 0; i < num_points - 1; i++) {
        int x1 = graph_x + 1 + i;
        int y1 = graph_y_positions[i];
        int x2 = graph_x + 1 + i + 1;
        int y2 = graph_y_positions[i + 1];
        
        // Draw line using Bresenham-like algorithm
        int dx = abs(x2 - x1);
        int dy = abs(y2 - y1);
        int sx = (x1 < x2) ? 1 : -1;
        int sy = (y1 < y2) ? 1 : -1;
        int err = dx - dy;
        
        int x = x1, y = y1;
        while (true) {
            if (x >= graph_x + 1 && x < graph_x + graph_width && 
                y >= graph_y + 1 && y < graph_y + graph_height) {
                mvaddch(y, x, '*');
            }
            
            if (x == x2 && y == y2) break;
            
            int e2 = 2 * err;
            if (e2 > -dy) {
                err -= dy;
                x += sx;
            }
            if (e2 < dx) {
                err += dx;
                y += sy;
            }
        }
    }
    
    // Draw current stats
    int stats_y = graph_y + graph_height + 1;
    char stats[100];
    snprintf(stats, sizeof(stats), "Games: %d | Best: %d | Avg: %.0f", 
             agent->total_games, agent->best_score, agent->average_score);
    mvaddstr(stats_y, graph_x, stats);
    
    // Draw X-axis label
    mvaddstr(graph_y + graph_height + 2, graph_x + graph_width / 2 - 5, "Time (games)");
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
            std::cout << "  S                 - Toggle score graph\n";
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
        // Don't erase screen - only update what changes
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
        } else if (key == 's' || key == 'S') {
            show_score_graph = !show_score_graph;
        } else if (key == 'v' || key == 'V') {
            show_stats = !show_stats;
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
                    // Calculate reward (improved reward shaping for better learning)
                    double reward = 0.0;
                    int score_diff = game.score - game.last_score;
                    int lines_diff = game.lines_cleared - game.last_lines;
                    
                    // Primary rewards (most important) - significantly increased
                    reward += lines_diff * 20.0;      // Increased from 5.0 (4x) - line clearing is main objective
                    reward += score_diff * 0.5;        // Increased from 0.1 (5x) - score increases are important
                    
                    // Survival bonus (encourage staying alive) - increased
                    if (!game.game_over) {
                        reward += 2.0;  // Increased from 1.0 (2x) - survival is crucial
                    }
                    
                    // Game over penalty - significantly increased
                    if (game.game_over) {
                        reward -= 200.0;  // Increased from 50.0 (4x) - game over is very bad
                    }
                    
                    // State quality penalties (prevent bad states) - increased for better guidance
                    int aggregate_height = game.getAggregateHeight(game.board);
                    reward -= aggregate_height * 0.2;  // Increased from 0.05 (4x) - high stacks are dangerous
                    
                    int holes = game.countHoles(game.board);
                    reward -= holes * 1.5;  // Increased from 0.3 (5x) - holes are very bad
                    
                    int bumpiness = game.calculateBumpiness(game.board);
                    reward -= bumpiness * 0.1;  // Increased from 0.02 (5x) - uneven surface is problematic
                    
                    // Bonus for keeping board low (encourage good play) - increased
                    int max_height = 0;
                    for (int x = 0; x < game.WIDTH; x++) {
                        int h = game.getColumnHeight(x, game.board);
                        if (h > max_height) max_height = h;
                    }
                    if (max_height < 10) {
                        reward += 2.0;  // Increased from 0.5 (4x) - low board is good
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
        
        // Check for convergence periodically (every 50 games)
        if (game.training_mode && agent.total_games > 0 && agent.total_games % 50 == 0) {
            if (agent.checkConvergence()) {
                std::cout << "\nNetwork has converged! Saving model and exiting..." << std::endl;
                agent.saveModel();
                std::cout << "Model saved. Exiting..." << std::endl;
                break;  // Exit main loop
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
            
            // Track recent scores for convergence detection
            agent.recent_scores.push_back(game.score);
            if (agent.recent_scores.size() > RLAgent::CONVERGENCE_WINDOW) {
                agent.recent_scores.pop_front();
            }
            
            // Track scores for graph display
            score_history.push_back(game.score);
            if (score_history.size() > MAX_HISTORY) {
                score_history.pop_front();
            }
            
            // Track best score improvement
            if (game.score > agent.best_score) {
                agent.games_since_best_improvement = 0;
            } else {
                agent.games_since_best_improvement++;
            }
            
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
            
            // Reset previous states after restart
            prev_score_global = -1;
            prev_lines_global = -1;
            prev_level_global = -1;
            prev_game_over_global = false;
            prev_paused_global = false;
            attrset(0);  // Reset attributes
        }
        
        // Update game (only if not game over in training mode, as we'll restart immediately)
        if (!(game.training_mode && game.game_over)) {
            game.update();
        }
        
        // Draw or clear score graph based on toggle state (draw BEFORE board to ensure visibility)
        static bool prev_show_score_graph = false;
        if (show_score_graph) {
            drawScoreGraph(&agent);
        } else if (prev_show_score_graph && !show_score_graph) {
            // Graph was just toggled off - clear the graph area
            int width, height;
            getmaxyx(stdscr, height, width);
            int graph_x = width - 60;
            int graph_y = 2;
            int graph_width = 55;
            int graph_height = 20;
            
            // Clear the entire graph area
            for (int y = graph_y; y <= graph_y + graph_height; y++) {
                for (int x = graph_x - 10; x <= graph_x + graph_width; x++) {
                    if (x >= 0 && x < width && y >= 0 && y < height) {
                        mvaddch(y, x, ' ');
                    }
                }
            }
        }
        prev_show_score_graph = show_score_graph;
        
        // Draw everything (after graph to ensure graph stays visible)
        drawBoard(stdscr, game, &agent, &tuner, show_score_graph, show_stats);
        
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

