# Terminal Tetris Game (C++)

A classic Tetris game implemented in C++ for the terminal using the `ncurses` library.

## Features

- All 7 classic Tetromino pieces (I, O, T, S, Z, J, L)
- Piece rotation with wall-kick mechanics
- Line clearing with scoring
- Progressive difficulty (speed increases with level)
- Next piece preview
- Score tracking
- Pause functionality
- Colorful terminal graphics

## Requirements

- C++ compiler with C++11 support (g++, clang++, etc.)
- `ncurses` development library

### Installing ncurses

**Ubuntu/Debian:**
```bash
sudo apt-get install libncurses5-dev libncursesw5-dev
```

**Fedora/RHEL:**
```bash
sudo dnf install ncurses-devel
```

**macOS:**
```bash
brew install ncurses
```

**Arch Linux:**
```bash
sudo pacman -S ncurses
```

## Building

### Using Makefile (Recommended)

```bash
make
```

This will create the `tetris` executable.

### Manual Compilation

```bash
g++ -Wall -Wextra -std=c++11 -O2 -o tetris tetris.cpp -lncurses
```

## How to Run

After building:

```bash
./tetris
```

Or use the Makefile:

```bash
make run
```

## Controls

- **← →** : Move piece left/right
- **↑** : Rotate piece clockwise
- **↓** : Soft drop (piece falls faster)
- **Space** : Hard drop (piece drops instantly)
- **P** : Pause/Unpause game
- **Q** : Quit game

## Game Rules

- Clear lines by filling them completely with blocks
- Score points for clearing lines:
  - 1 line: 100 × level
  - 2 lines: 300 × level
  - 3 lines: 500 × level
  - 4 lines (Tetris): 800 × level
- Game speed increases every 10 lines cleared
- Game ends when pieces reach the top of the board

## Scoring

- Line clears: Points based on number of lines (see above)
- Soft drop: +1 point per cell
- Hard drop: +2 points per cell

## Building Options

- `make` or `make all` - Build the game
- `make clean` - Remove build artifacts
- `make run` - Build and run the game
- `make install` - Make the executable executable (chmod +x)

## Notes

- The game requires a terminal that supports colors and Unicode characters
- Minimum terminal size: 40x30 characters recommended
- For best experience, use a terminal with true color support

Enjoy playing Tetris!
