# Neural Network Weight Visualizer (C++)

A real-time C++ visualizer for monitoring neural network weights during Tetris AI training.

## Features

- **Dynamic File Monitoring**: Automatically detects changes to `tetris_model.txt` and updates visualization
- **Color-Coded Weights**: 
  - **Red** = Negative weights
  - **White** = Near-zero weights
  - **Blue** = Positive weights
- **Real-Time Updates**: Visualization refreshes automatically when model file is saved
- **Timestamp Display**: Shows when the model was last updated (in console and visually)
- **Four-Panel Layout**:
  - Top Left: Weights1 (Input → Hidden, 29×64)
  - Top Right: Bias1 (Hidden layer, 64 values)
  - Bottom Left: Weights2 (Hidden → Output, 64×1)
  - Bottom Right: Bias2 (Output layer, 1 value)

## Building

The visualizer is built automatically with the main project:

```bash
make weight_visualizer
```

Or build everything:

```bash
make all
```

## Running

```bash
./weight_visualizer [model_file]
```

If no filename is provided, it defaults to `tetris_model.txt`.

## Usage

1. **Start the visualizer**:
   ```bash
   ./weight_visualizer
   ```

2. **Run Tetris training** in another terminal:
   ```bash
   ./tetris
   ```

3. **Watch the visualization update** automatically when the model file is saved (every 100 training episodes or on exit).

## Controls

- **ESC** or **Q**: Quit the visualizer
- **R**: Force reload the model file
- **Window Resize**: Visualization adapts to window size

## Color Legend

The bottom of the window shows a color gradient:
- **Left (Red)**: Negative weights
- **Middle (White)**: Zero/near-zero weights  
- **Right (Blue)**: Positive weights

## Timestamp Information

When the model file is updated:
- A **green indicator** appears in the bottom panel
- The **timestamp** is printed to the console:
  ```
  === MODEL FILE UPDATED ===
  Timestamp: 2025-11-30 21:45:12
  Update Count: #5
  ===========================
  ```

## Technical Details

- **Graphics Library**: SDL2
- **File Monitoring**: Uses `stat()` to check file modification time
- **Update Frequency**: Checks for file changes every 100ms
- **Window Size**: Default 1600×900 (resizable)

## Requirements

- SDL2 development libraries
- C++11 compiler
- Linux/Unix system (for file monitoring)

## Troubleshooting

**Visualizer shows "Waiting for model file..."**:
- Make sure `tetris_model.txt` exists
- Run Tetris training to generate the model file

**Visualization doesn't update**:
- Check that Tetris is saving the model (every 100 episodes)
- Press **R** to force reload
- Check console for error messages

**Colors look wrong**:
- Colors are normalized per matrix (each matrix has its own min/max)
- This allows better visualization of weight patterns within each layer

## Integration with Training

The visualizer is designed to run alongside the Tetris training:

1. **Terminal 1**: Run the visualizer
   ```bash
   ./weight_visualizer
   ```

2. **Terminal 2**: Run Tetris training
   ```bash
   ./tetris
   ```

The visualizer will automatically update whenever Tetris saves the model (every 100 training episodes or when you quit).

## Notes

- The visualizer uses SDL2 for graphics rendering
- Text rendering is limited (SDL2 doesn't include text rendering by default)
- Timestamp information is displayed in the console and as a visual indicator
- The visualization scales with window size for better viewing

