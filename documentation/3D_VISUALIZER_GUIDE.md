# 3D Weight Visualizer Guide

## Features

The weight visualizer now supports both **2D** and **3D** visualization modes with interactive hover detection.

## Controls

### General Controls
- **K**: Toggle between 2D and 3D view
- **Q** or **ESC**: Quit
- **R**: Force reload model file

### 3D View Controls
- **Mouse Drag (Left Button)**: Rotate camera around the network
- **Mouse Wheel**: Zoom in/out
- **Mouse Hover**: Display weight values for connections

## 3D Visualization

### Network Structure
The 3D view shows the neural network as three layers:

1. **Input Layer** (Left, Red nodes)
   - 29 nodes representing input features
   - Connected to hidden layer

2. **Hidden Layer** (Center, Green nodes)
   - 64 nodes
   - Receives connections from input layer
   - Sends connections to output layer

3. **Output Layer** (Right, Blue nodes)
   - 1 node for Q-value output
   - Receives connections from hidden layer

### Connection Visualization
- **Connections** are drawn as colored lines between nodes
- **Color coding**:
  - **Red**: Negative weights
  - **White**: Near-zero weights
  - **Blue**: Positive weights
- **Hovered connections** are highlighted in **yellow** and thicker

### Hover Information
When you hover over a connection in 3D mode:
- The connection is highlighted in yellow
- Weight information is printed to the console:
  ```
  [HOVER] Connection: Layer 0 Node 5 -> Layer 1 Node 12 | Weight: -0.123456
  ```

## Usage

1. **Start the visualizer**:
   ```bash
   ./weight_visualizer
   ```

2. **Switch to 3D view**: Press **K**

3. **Explore the network**:
   - Drag with left mouse button to rotate
   - Scroll mouse wheel to zoom
   - Move mouse over connections to see weight values

4. **Switch back to 2D**: Press **K** again

## Tips

- **3D view** is best for understanding network structure and connection patterns
- **2D view** is better for detailed weight matrix analysis
- **Hover detection** works best when you move the mouse slowly over connections
- **Zoom out** (scroll down) to see the full network structure
- **Zoom in** (scroll up) to examine specific connections in detail

## Technical Details

- Uses **OpenGL** for 3D rendering
- **Real-time updates** when model file changes
- **Interactive camera** with rotation and zoom
- **Color-coded weights** for easy pattern recognition
- **Console output** for hover information (since text rendering requires additional libraries)

## Troubleshooting

**3D view doesn't appear**:
- Make sure OpenGL is installed: `sudo apt-get install libgl1-mesa-dev libglu1-mesa-dev`
- Check console for OpenGL errors

**Hover doesn't work**:
- Move mouse slowly over connections
- Try zooming in for better detection
- Check console for hover messages

**Performance issues**:
- Reduce window size if rendering is slow
- The visualizer renders at ~60 FPS by default

