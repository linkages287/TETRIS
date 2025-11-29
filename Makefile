# Makefile for Terminal Tetris (C++)

CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++11 -O2
LDFLAGS = -lncurses
TARGET = tetris
SOURCE = tetris.cpp

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCE) $(LDFLAGS)

# Clean build artifacts
clean:
	rm -f $(TARGET)

# Install (optional - just makes executable)
install: $(TARGET)
	chmod +x $(TARGET)

# Run the game
run: $(TARGET)
	./$(TARGET)

.PHONY: all clean install run

