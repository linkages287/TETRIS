# Neuron Disposition in Visualizers

## Overview

Both visualizers have been updated to correctly display the new 49-feature structure. Here's how neurons are arranged and displayed.

## Python Visualizer (`visualize_weights.py`)

### Main Heatmap View

**Weights1 (Input → Hidden):**
- **Rows**: 49 input features (y-axis)
- **Columns**: 64 hidden neurons (x-axis)
- **Y-axis labels**: Shows feature names with smart spacing
  - Every ~2-3 features shown to avoid crowding
  - Labels: `H0-H9` (heights), `HMax/HMin/HMean/HStd` (stats), `HL0-HL9` (holes), `HD0-HD8` (differences), `CP0-CP6` (current piece), `NP0-NP6` (next piece), `Lines`, `Level`

**Weights2 (Hidden → Output):**
- **Rows**: 64 hidden neurons (y-axis)
- **Columns**: 1 output neuron (x-axis)

### Detailed View (`--detail` flag)

**Left Panel - Full Heatmap:**
- Shows all 49 input features with full labels
- Each row labeled: `H0`, `H1`, ..., `H9`, `HMax`, `HMin`, `HMean`, `HStd`, `HL0`, ..., `HL9`, `HD0`, ..., `HD8`, `CP0-CP6`, `NP0-NP6`, `Lines`, `Level`

**Right Panel - Weight Magnitude:**
- Bar chart showing average |weight| per input feature
- Helps identify which features are most important
- Same feature labels as left panel

### Feature Label Mapping

| Index | Label | Description |
|-------|-------|-------------|
| 0-9 | H0-H9 | Column heights (centered & normalized) |
| 10 | HMax | Maximum column height |
| 11 | HMin | Minimum column height |
| 12 | HMean | Mean column height |
| 13 | HStd | Height standard deviation |
| 14-23 | HL0-HL9 | Holes per column (spatial) |
| 24-32 | HD0-HD8 | Height differences between adjacent columns |
| 33-39 | CP0-CP6 | Current piece type (one-hot) |
| 40-46 | NP0-NP6 | Next piece type (one-hot) |
| 47 | Lines | Lines cleared |
| 48 | Level | Current level |

## C++ 3D Visualizer (`weight_visualizer`)

### 3D Layout

**Input Layer (49 neurons):**
- **Grid**: 7×7 square grid (perfect for 49 nodes)
- **Position**: Left side of 3D space
- **Color**: Light red
- **Spacing**: 0.4 units between nodes

**Hidden Layer (64 neurons):**
- **Grid**: 8×8 square grid (perfect for 64 nodes)
- **Position**: Center of 3D space
- **Color**: Light green
- **Spacing**: 0.4 units between nodes

**Output Layer (1 neuron):**
- **Grid**: Single node (centered)
- **Position**: Right side of 3D space
- **Color**: Light blue
- **Spacing**: N/A (single node)

### 2D View (Press '2' key)

**Top Left Panel:**
- Weights1 matrix (49×64)
- Each cell represents weight value
- Color-coded: Red (negative) → White (zero) → Blue (positive)

**Top Right Panel:**
- Bias1 vector (64 values)
- Horizontal bar chart

**Bottom Left Panel:**
- Weights2 matrix (64×1)
- Single column representing output weights

**Bottom Right Panel:**
- Bias2 value (single value)
- Single bar

### Grid Layout Details

**Input Layer (49 nodes):**
```
7×7 Grid Layout:
[0]  [1]  [2]  [3]  [4]  [5]  [6]
[7]  [8]  [9]  [10] [11] [12] [13]
[14] [15] [16] [17] [18] [19] [20]
[21] [22] [23] [24] [25] [26] [27]
[28] [29] [30] [31] [32] [33] [34]
[35] [36] [37] [38] [39] [40] [41]
[42] [43] [44] [45] [46] [47] [48]
```

**Hidden Layer (64 nodes):**
```
8×8 Grid Layout:
[0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]
[8]  [9]  [10] [11] [12] [13] [14] [15]
... (continues for all 64 nodes)
```

## Visual Features

### Python Visualizer

1. **Color Mapping:**
   - Blue: Negative weights
   - White: Zero weights
   - Red: Positive weights
   - Intensity: Magnitude of weight

2. **Interactive Features:**
   - Hover tooltips showing exact weight values
   - Row/column indices on hover
   - Dynamic updates when model file changes

3. **Statistics Display:**
   - Mean, std dev, min, max for each weight matrix
   - Histograms for bias distributions
   - Weight magnitude distribution

### C++ Visualizer

1. **3D View:**
   - Rotatable camera (mouse drag)
   - Zoom (mouse wheel)
   - Connection lines colored by weight
   - Hover to see connection details

2. **2D View:**
   - Color-coded weight matrices
   - Legend showing color scale
   - Timestamp of last update

## Usage Tips

### Python Visualizer

```bash
# View all features with labels
python3 visualize_weights.py

# Detailed view with all feature names
python3 visualize_weights.py --detail

# Monitor during training
python3 visualize_weights.py --monitor

# Save visualization
python3 visualize_weights.py --save weights.png
```

### C++ Visualizer

```bash
# Run 3D visualizer
./weight_visualizer

# Controls:
# - Mouse drag: Rotate camera
# - Mouse wheel: Zoom
# - '2' key: Switch to 2D view
# - '3' key: Switch to 3D view
# - ESC: Exit
```

## Feature Grouping Visualization

The visualizers help identify feature importance:

1. **Column Heights (H0-H9)**: First 10 rows
2. **Height Statistics (HMax-HStd)**: Rows 10-13
3. **Holes Per Column (HL0-HL9)**: Rows 14-23
4. **Height Differences (HD0-HD8)**: Rows 24-32
5. **Current Piece (CP0-CP6)**: Rows 33-39
6. **Next Piece (NP0-NP6)**: Rows 40-46
7. **Game State (Lines, Level)**: Rows 47-48

## Verification

Both visualizers correctly:
- ✅ Display 49 input features
- ✅ Show proper neuron arrangement
- ✅ Handle 64 hidden neurons
- ✅ Display 1 output neuron
- ✅ Update dynamically when model changes

## Notes

- **Old models (33 features) will NOT display correctly** - need 49-feature models
- **Feature labels help identify** which inputs are most important
- **3D layout** shows spatial relationships between layers
- **2D view** provides detailed weight inspection



