# Makefile for Terminal Tetris (C++)

CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++11 -O2
LDFLAGS = -lncurses
TARGET = tetris
VISUALIZER = weight_visualizer
SOURCES = tetris.cpp rl_agent.cpp parameter_tuner.cpp
OBJECTS = $(SOURCES:.cpp=.o)
VISUALIZER_OBJ = weight_visualizer.o

# SDL2: sdl2-config points at .../include/SDL2 (for <SDL.h>).
# Also add the parent include dir so <SDL2/SDL.h> works.
SDL_PREFIX := $(shell sdl2-config --prefix 2>/dev/null)
SDLCFLAGS := $(shell sdl2-config --cflags 2>/dev/null) -I$(SDL_PREFIX)/include
SDLLIBS := $(shell sdl2-config --libs 2>/dev/null)

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    SDLCFLAGS += -DGL_SILENCE_DEPRECATION
    SDLLIBS += -framework OpenGL
else
    SDLLIBS += -lGL -lGLU
endif

# Default target
all: $(TARGET) $(VISUALIZER)

# Build the executable
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS)

# Build the weight visualizer
$(VISUALIZER): $(VISUALIZER_OBJ)
	$(CXX) $(CXXFLAGS) -o $(VISUALIZER) $(VISUALIZER_OBJ) $(SDLLIBS)

$(VISUALIZER_OBJ): weight_visualizer.cpp
	$(CXX) $(CXXFLAGS) $(SDLCFLAGS) -c $< -o $@

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
