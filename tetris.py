#!/usr/bin/env python3
"""
Terminal Tetris Game
Controls:
    Arrow Keys - Move left/right, rotate, soft drop
    Space - Hard drop
    Q - Quit
    P - Pause
"""

import curses
import random
import time
from typing import List, Tuple, Optional

# Tetris pieces (tetrominoes) - each piece is defined by its shape
# Format: [rotations][y][x] where each rotation is a 4x4 grid
PIECES = {
    'I': [
        [[0,0,0,0], [1,1,1,1], [0,0,0,0], [0,0,0,0]],
        [[0,0,1,0], [0,0,1,0], [0,0,1,0], [0,0,1,0]],
        [[0,0,0,0], [0,0,0,0], [1,1,1,1], [0,0,0,0]],
        [[0,1,0,0], [0,1,0,0], [0,1,0,0], [0,1,0,0]]
    ],
    'O': [
        [[0,0,0,0], [0,1,1,0], [0,1,1,0], [0,0,0,0]],
        [[0,0,0,0], [0,1,1,0], [0,1,1,0], [0,0,0,0]],
        [[0,0,0,0], [0,1,1,0], [0,1,1,0], [0,0,0,0]],
        [[0,0,0,0], [0,1,1,0], [0,1,1,0], [0,0,0,0]]
    ],
    'T': [
        [[0,0,0,0], [0,1,0,0], [1,1,1,0], [0,0,0,0]],
        [[0,0,0,0], [0,1,0,0], [0,1,1,0], [0,1,0,0]],
        [[0,0,0,0], [0,0,0,0], [1,1,1,0], [0,1,0,0]],
        [[0,0,0,0], [0,1,0,0], [1,1,0,0], [0,1,0,0]]
    ],
    'S': [
        [[0,0,0,0], [0,1,1,0], [1,1,0,0], [0,0,0,0]],
        [[0,0,0,0], [0,1,0,0], [0,1,1,0], [0,0,1,0]],
        [[0,0,0,0], [0,0,0,0], [0,1,1,0], [1,1,0,0]],
        [[0,0,0,0], [1,0,0,0], [1,1,0,0], [0,1,0,0]]
    ],
    'Z': [
        [[0,0,0,0], [1,1,0,0], [0,1,1,0], [0,0,0,0]],
        [[0,0,0,0], [0,0,1,0], [0,1,1,0], [0,1,0,0]],
        [[0,0,0,0], [0,0,0,0], [1,1,0,0], [0,1,1,0]],
        [[0,0,0,0], [0,1,0,0], [1,1,0,0], [1,0,0,0]]
    ],
    'J': [
        [[0,0,0,0], [1,0,0,0], [1,1,1,0], [0,0,0,0]],
        [[0,0,0,0], [0,1,1,0], [0,1,0,0], [0,1,0,0]],
        [[0,0,0,0], [0,0,0,0], [1,1,1,0], [0,0,1,0]],
        [[0,0,0,0], [0,1,0,0], [0,1,0,0], [1,1,0,0]]
    ],
    'L': [
        [[0,0,0,0], [0,0,1,0], [1,1,1,0], [0,0,0,0]],
        [[0,0,0,0], [0,1,0,0], [0,1,0,0], [0,1,1,0]],
        [[0,0,0,0], [0,0,0,0], [1,1,1,0], [1,0,0,0]],
        [[0,0,0,0], [1,1,0,0], [0,1,0,0], [0,1,0,0]]
    ]
}

# Colors for each piece type
PIECE_COLORS = {
    'I': 1,  # Cyan
    'O': 2,  # Yellow
    'T': 3,  # Magenta
    'S': 4,  # Green
    'Z': 5,  # Red
    'J': 6,  # Blue
    'L': 7,  # Orange/White
}

class TetrisPiece:
    """Represents a falling Tetris piece"""
    def __init__(self, piece_type: str, x: int = 0, y: int = 0):
        self.type = piece_type
        self.x = x
        self.y = y
        self.rotation = 0
        self.shape = PIECES[piece_type]
        self.color = PIECE_COLORS[piece_type]
    
    def get_shape(self) -> List[List[int]]:
        """Get the current rotation of the piece"""
        return self.shape[self.rotation]
    
    def rotate(self):
        """Rotate the piece clockwise"""
        self.rotation = (self.rotation + 1) % 4
    
    def get_blocks(self) -> List[Tuple[int, int]]:
        """Get list of (x, y) positions of blocks in the piece"""
        blocks = []
        shape = self.get_shape()
        for dy in range(4):
            for dx in range(4):
                if shape[dy][dx]:
                    blocks.append((self.x + dx, self.y + dy))
        return blocks

class TetrisGame:
    """Main Tetris game class"""
    def __init__(self, width: int = 10, height: int = 20):
        self.width = width
        self.height = height
        self.board = [[0 for _ in range(width)] for _ in range(height)]
        self.current_piece: Optional[TetrisPiece] = None
        self.next_piece: Optional[TetrisPiece] = None
        self.score = 0
        self.lines_cleared = 0
        self.level = 1
        self.game_over = False
        self.paused = False
        self.fall_time = 0
        self.fall_delay = 0.5  # Initial fall delay in seconds
        self.last_fall_time = time.time()
        
    def spawn_piece(self):
        """Spawn a new piece at the top"""
        if self.next_piece is None:
            piece_type = random.choice(list(PIECES.keys()))
            self.next_piece = TetrisPiece(piece_type)
        
        self.current_piece = self.next_piece
        self.current_piece.x = self.width // 2 - 2
        self.current_piece.y = 0
        
        # Check if game over
        if self.check_collision(self.current_piece):
            self.game_over = True
        
        # Generate next piece
        piece_type = random.choice(list(PIECES.keys()))
        self.next_piece = TetrisPiece(piece_type)
    
    def check_collision(self, piece: TetrisPiece, dx: int = 0, dy: int = 0) -> bool:
        """Check if piece collides with board boundaries or placed blocks"""
        blocks = piece.get_blocks()
        for x, y in blocks:
            nx, ny = x + dx, y + dy
            # Check boundaries
            if nx < 0 or nx >= self.width or ny >= self.height:
                return True
            # Check placed blocks (only check if within board)
            if ny >= 0 and self.board[ny][nx] != 0:
                return True
        return False
    
    def place_piece(self):
        """Place the current piece on the board"""
        if self.current_piece is None:
            return
        
        blocks = self.current_piece.get_blocks()
        for x, y in blocks:
            if 0 <= y < self.height and 0 <= x < self.width:
                self.board[y][x] = self.current_piece.color
        
        # Clear lines
        lines_cleared = self.clear_lines()
        self.lines_cleared += lines_cleared
        
        # Update score
        if lines_cleared > 0:
            points = [0, 100, 300, 500, 800][min(lines_cleared, 4)]
            self.score += points * self.level
        
        # Update level (every 10 lines)
        self.level = self.lines_cleared // 10 + 1
        self.fall_delay = max(0.05, 0.5 - (self.level - 1) * 0.05)
        
        self.current_piece = None
    
    def clear_lines(self) -> int:
        """Clear completed lines and return number of lines cleared"""
        lines_to_clear = []
        for y in range(self.height):
            if all(self.board[y][x] != 0 for x in range(self.width)):
                lines_to_clear.append(y)
        
        # Remove lines from bottom to top
        for y in reversed(lines_to_clear):
            del self.board[y]
            self.board.insert(0, [0 for _ in range(self.width)])
        
        return len(lines_to_clear)
    
    def move_piece(self, dx: int, dy: int) -> bool:
        """Try to move the current piece. Returns True if successful"""
        if self.current_piece is None:
            return False
        
        if not self.check_collision(self.current_piece, dx, dy):
            self.current_piece.x += dx
            self.current_piece.y += dy
            return True
        return False
    
    def rotate_piece(self) -> bool:
        """Try to rotate the current piece. Returns True if successful"""
        if self.current_piece is None:
            return False
        
        old_rotation = self.current_piece.rotation
        self.current_piece.rotate()
        
        if self.check_collision(self.current_piece):
            # Try wall kicks
            for dx in [-1, 1, -2, 2]:
                if not self.check_collision(self.current_piece, dx, 0):
                    self.current_piece.x += dx
                    return True
            # Rotation failed, revert
            self.current_piece.rotation = old_rotation
            return False
        return True
    
    def hard_drop(self):
        """Drop the piece to the bottom instantly"""
        if self.current_piece is None:
            return
        
        while self.move_piece(0, 1):
            self.score += 2  # Bonus points for hard drop
        self.place_piece()
        self.spawn_piece()
    
    def update(self):
        """Update game state (falling pieces)"""
        if self.game_over or self.paused:
            return
        
        current_time = time.time()
        if current_time - self.last_fall_time >= self.fall_delay:
            if self.current_piece is None:
                self.spawn_piece()
            elif not self.move_piece(0, 1):
                self.place_piece()
                self.spawn_piece()
            self.last_fall_time = current_time

def draw_board(stdscr, game: TetrisGame):
    """Draw the game board"""
    height, width = stdscr.getmaxyx()
    
    # Board dimensions
    board_x = width // 2 - game.width // 2 - 1
    board_y = 2
    
    # Draw board border
    stdscr.addstr(board_y - 1, board_x - 1, "+" + "-" * (game.width * 2) + "+")
    for y in range(game.height):
        stdscr.addstr(board_y + y, board_x - 1, "|")
        stdscr.addstr(board_y + y, board_x + game.width * 2, "|")
    stdscr.addstr(board_y + game.height, board_x - 1, "+" + "-" * (game.width * 2) + "+")
    
    # Draw placed blocks
    for y in range(game.height):
        for x in range(game.width):
            if game.board[y][x] != 0:
                stdscr.addstr(board_y + y, board_x + x * 2, "[]", curses.color_pair(game.board[y][x]))
    
    # Draw current piece
    if game.current_piece:
        blocks = game.current_piece.get_blocks()
        for x, y in blocks:
            if 0 <= y < game.height and 0 <= x < game.width:
                screen_y = board_y + y
                screen_x = board_x + x * 2
                if screen_y >= 0:
                    stdscr.addstr(screen_y, screen_x, "[]", curses.color_pair(game.current_piece.color))
    
    # Draw next piece preview
    if game.next_piece:
        preview_x = board_x + game.width * 2 + 5
        preview_y = board_y + 2
        stdscr.addstr(preview_y - 1, preview_x, "Next:")
        shape = game.next_piece.get_shape()
        for dy in range(4):
            for dx in range(4):
                if shape[dy][dx]:
                    stdscr.addstr(preview_y + dy, preview_x + dx * 2, "[]", 
                                curses.color_pair(game.next_piece.color))
    
    # Draw score and stats
    info_x = board_x - 15
    info_y = board_y + 2
    stdscr.addstr(info_y, info_x, f"Score: {game.score}")
    stdscr.addstr(info_y + 1, info_x, f"Lines: {game.lines_cleared}")
    stdscr.addstr(info_y + 2, info_x, f"Level: {game.level}")
    
    # Draw controls
    controls_y = board_y + game.height + 2
    stdscr.addstr(controls_y, board_x, "Controls:")
    stdscr.addstr(controls_y + 1, board_x, "Left/Right: Move")
    stdscr.addstr(controls_y + 2, board_x, "Up: Rotate")
    stdscr.addstr(controls_y + 3, board_x, "Down: Soft Drop")
    stdscr.addstr(controls_y + 4, board_x, "Space: Hard Drop")
    stdscr.addstr(controls_y + 5, board_x, "P: Pause  Q: Quit")
    
    # Draw game over or pause message
    if game.game_over:
        msg = "GAME OVER!"
        stdscr.addstr(board_y + game.height // 2, board_x + game.width - len(msg) // 2, 
                     msg, curses.A_BOLD | curses.color_pair(1))
    elif game.paused:
        msg = "PAUSED"
        stdscr.addstr(board_y + game.height // 2, board_x + game.width - len(msg) // 2, 
                     msg, curses.A_BOLD)

def init_colors():
    """Initialize color pairs"""
    curses.start_color()
    curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)    # I piece
    curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # O piece
    curses.init_pair(3, curses.COLOR_MAGENTA, curses.COLOR_BLACK) # T piece
    curses.init_pair(4, curses.COLOR_GREEN, curses.COLOR_BLACK)   # S piece
    curses.init_pair(5, curses.COLOR_RED, curses.COLOR_BLACK)     # Z piece
    curses.init_pair(6, curses.COLOR_BLUE, curses.COLOR_BLACK)    # J piece
    curses.init_pair(7, curses.COLOR_WHITE, curses.COLOR_BLACK)   # L piece

def main(stdscr):
    """Main game loop"""
    # Setup
    curses.curs_set(0)  # Hide cursor
    stdscr.nodelay(1)   # Non-blocking input
    stdscr.timeout(1)  # Refresh every 50ms
    init_colors()
    
    game = TetrisGame()
    
    # Game loop
    while True:
        # Clear screen
        stdscr.clear()
        
        # Handle input
        key = stdscr.getch()
        
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('p') or key == ord('P'):
            game.paused = not game.paused
        elif not game.game_over and not game.paused:
            if key == curses.KEY_LEFT:
                game.move_piece(-1, 0)
            elif key == curses.KEY_RIGHT:
                game.move_piece(1, 0)
            elif key == curses.KEY_DOWN:
                if game.move_piece(0, 1):
                    game.score += 1  # Bonus for soft drop
            elif key == curses.KEY_UP:
                game.rotate_piece()
            elif key == ord(' '):
                game.hard_drop()
        
        # Update game
        game.update()
        
        # Draw everything
        draw_board(stdscr, game)
        
        # Refresh screen
        stdscr.refresh()
        
        # Small delay to prevent excessive CPU usage
        time.sleep(0.01)

if __name__ == "__main__":
    curses.wrapper(main)

