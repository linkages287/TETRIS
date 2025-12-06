# Visualization Update for New Feature Structure

## Changes Made

Updated both visualization tools to work with the new 49-feature structure.

### Files Updated

1. **visualize_weights.py** (Python visualizer)
   - Updated `INPUT_SIZE` from 29 to 49
   - Added descriptive feature labels:
     - `H0-H9`: Column heights (0-9)
     - `HMax, HMin, HMean, HStd`: Height statistics (10-13)
     - `HL0-HL9`: Holes per column (14-23)
     - `HD0-HD8`: Height differences (24-32)
     - `CP0-CP6`: Current piece type (33-39)
     - `NP0-NP6`: Next piece type (40-46)
     - `Lines`: Lines cleared (47)
     - `Level`: Level (48)

2. **weight_visualizer.cpp** (C++ OpenGL visualizer)
   - Updated `INPUT_SIZE` from 29 to 49
   - Recompiled successfully

3. **README.md**
   - Updated network architecture documentation
   - Reflects new 49-feature structure

## Feature Mapping

| Index Range | Feature Type | Count | Labels |
|-------------|--------------|-------|--------|
| 0-9 | Column Heights | 10 | H0-H9 |
| 10-13 | Height Statistics | 4 | HMax, HMin, HMean, HStd |
| 14-23 | Holes Per Column | 10 | HL0-HL9 |
| 24-32 | Height Differences | 9 | HD0-HD8 |
| 33-39 | Current Piece | 7 | CP0-CP6 |
| 40-46 | Next Piece | 7 | NP0-NP6 |
| 47 | Lines Cleared | 1 | Lines |
| 48 | Level | 1 | Level |

## Usage

### Python Visualizer
```bash
# Single visualization
python3 visualize_weights.py

# Save to file
python3 visualize_weights.py --save weights.png

# Monitor for changes
python3 visualize_weights.py --monitor

# Detailed view
python3 visualize_weights.py --detail
```

### C++ Visualizer
```bash
# Run 3D visualizer
./weight_visualizer
```

## Verification

Both visualizers now correctly:
- ✅ Load models with 49 input features
- ✅ Display weight matrices (49×64 for weights1)
- ✅ Show feature labels (Python visualizer)
- ✅ Handle file monitoring and updates

## Testing

To test the visualizations:

1. **Train a new model** (or use existing 49-feature model):
   ```bash
   ./tetris
   ```

2. **Visualize weights**:
   ```bash
   python3 visualize_weights.py
   ```

3. **Monitor during training**:
   ```bash
   python3 visualize_weights.py --monitor
   ```

## Notes

- **Old models (33 features) will NOT work** with updated visualizers
- **New models (49 features) required** for visualization
- Feature labels help identify which features are most important
- Weight magnitude visualization shows which features the network prioritizes


