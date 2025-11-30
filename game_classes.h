#ifndef GAME_CLASSES_H
#define GAME_CLASSES_H

#include <vector>
#include <chrono>

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
    
    TetrisPiece(int piece_type = 0, int x = 0, int y = 0);
    void getShape(int shape[4][4]) const;
    void rotate();
    std::vector<Point> getBlocks() const;
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
    bool ai_enabled;
    bool training_mode;
    int last_score;
    int last_lines;
    double fall_delay;
    std::chrono::steady_clock::time_point last_fall_time;
    std::chrono::steady_clock::time_point last_ai_time;
    
    TetrisGame();
    ~TetrisGame();
    
    // Disable copy constructor and assignment to prevent double deletion
    TetrisGame(const TetrisGame&) = delete;
    TetrisGame& operator=(const TetrisGame&) = delete;
    
    // Game control methods
    void spawnPiece();
    void placePiece();
    int clearLines();
    bool movePiece(int dx, int dy);
    bool rotatePiece();
    void hardDrop();
    void update();
    void executeAIMove(int rotation, int x_pos);
    
    // Methods needed by RL agent
    bool checkCollision(const TetrisPiece& piece, int dx = 0, int dy = 0) const;
    std::vector<std::vector<int>> simulatePlacePiece(const TetrisPiece& piece, int drop_y) const;
    int simulateClearLines(std::vector<std::vector<int>>& sim_board) const;
    int getColumnHeight(int x, const std::vector<std::vector<int>>& board) const;
    int countHoles(const std::vector<std::vector<int>>& board) const;
    int calculateBumpiness(const std::vector<std::vector<int>>& board) const;
    int getAggregateHeight(const std::vector<std::vector<int>>& board) const;
};

#endif // GAME_CLASSES_H

