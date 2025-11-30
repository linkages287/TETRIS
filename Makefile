# Makefile for Terminal Tetris (C++)

CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++11 -O2
LDFLAGS = -lncurses
SDLFLAGS = $(shell sdl2-config --cflags --libs) -lGL -lGLU
TARGET = tetris
VISUALIZER = weight_visualizer
SOURCES = tetris.cpp rl_agent.cpp parameter_tuner.cpp
OBJECTS = $(SOURCES:.cpp=.o)
VISUALIZER_OBJ = weight_visualizer.o

# Default target
all: $(TARGET) $(VISUALIZER)

# Build the executable
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS)

# Build the weight visualizer
$(VISUALIZER): $(VISUALIZER_OBJ)
	$(CXX) $(CXXFLAGS) -o $(VISUALIZER) $(VISUALIZER_OBJ) $(SDLFLAGS)

# Build object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(TARGET) $(VISUALIZER) $(OBJECTS) $(VISUALIZER_OBJ)

# Install (optional - just makes executable)
install: $(TARGET) $(VISUALIZER)
	chmod +x $(TARGET) $(VISUALIZER)

# Run the game
run: $(TARGET)
	./$(TARGET)

# Run the visualizer
visualize: $(VISUALIZER)
	./$(VISUALIZER)

.PHONY: all clean install run visualize

