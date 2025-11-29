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

struct Point {
    int x, y;
    Point(int x = 0, int y = 0) : x(x), y(y) {}
};

class TetrisPiece {
public:
    int type;
    int x, y;
    int rotation;
    int color;
    
    TetrisPiece(int piece_type = 0, int x = 0, int y = 0) 
        : type(piece_type), x(x), y(y), rotation(0), color(PIECE_COLORS[piece_type]) {}
    
    void getShape(int shape[4][4]) const {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                shape[i][j] = PIECES[type][rotation][i][j];
            }
        }
    }
    
    void rotate() {
        rotation = (rotation + 1) % 4;
    }
    
    std::vector<Point> getBlocks() const {
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
};

class TetrisGame {
public:
    static const int WIDTH = 10;
    static const int HEIGHT = 20;
    
    std::vector<std::vector<int>> board;
    TetrisPiece* current_piece;
    TetrisPiece* next_piece;
    int score;
    int lines_cleared;
    int level;
    bool game_over;
    bool paused;
    double fall_delay;
    std::chrono::steady_clock::time_point last_fall_time;
    
    TetrisGame() 
        : board(HEIGHT, std::vector<int>(WIDTH, 0)),
          current_piece(nullptr),
          next_piece(nullptr),
          score(0),
          lines_cleared(0),
          level(1),
          game_over(false),
          paused(false),
          fall_delay(0.5),
          last_fall_time(std::chrono::steady_clock::now()) {
        spawnPiece();
    }
    
    ~TetrisGame() {
        delete current_piece;
        delete next_piece;
    }
    
    void spawnPiece() {
        if (next_piece == nullptr) {
            int piece_type = rand() % 7;
            next_piece = new TetrisPiece(piece_type);
        }
        
        current_piece = next_piece;
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
    
    bool checkCollision(const TetrisPiece& piece, int dx = 0, int dy = 0) const {
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
    
    void placePiece() {
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
    
    int clearLines() {
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
    
    bool movePiece(int dx, int dy) {
        if (current_piece == nullptr) return false;
        
        if (!checkCollision(*current_piece, dx, dy)) {
            current_piece->x += dx;
            current_piece->y += dy;
            return true;
        }
        return false;
    }
    
    bool rotatePiece() {
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
    
    void hardDrop() {
        if (current_piece == nullptr) return;
        
        while (movePiece(0, 1)) {
            score += 2;  // Bonus points for hard drop
        }
        placePiece();
        spawnPiece();
    }
    
    void update() {
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
};

void drawBoard(WINDOW* win, TetrisGame& game) {
    int height, width;
    getmaxyx(win, height, width);
    (void)height;  // height not used, but required by getmaxyx macro
    
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
    
    // Draw controls
    int controls_y = board_y + game.HEIGHT + 2;
    mvaddstr(controls_y, board_x, "Controls:");
    mvaddstr(controls_y + 1, board_x, "Left/Right: Move");
    mvaddstr(controls_y + 2, board_x, "Up: Rotate");
    mvaddstr(controls_y + 3, board_x, "Down: Soft Drop");
    mvaddstr(controls_y + 4, board_x, "Space: Hard Drop");
    mvaddstr(controls_y + 5, board_x, "P: Pause  Q: Quit");
    
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

int main() {
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
    
    // Game loop
    while (true) {
        // Clear screen
        clear();
        
        // Handle input
        int key = getch();
        
        if (key == 'q' || key == 'Q') {
            break;
        } else if (key == 'p' || key == 'P') {
            game.paused = !game.paused;
        } else if (!game.game_over && !game.paused) {
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
        
        // Update game
        game.update();
        
        // Draw everything
        drawBoard(stdscr, game);
        
        // Refresh screen
        refresh();
        
        // Small delay to prevent excessive CPU usage
        napms(10);
    }
    
    endwin();
    return 0;
}

