# Weight Visualizer for Tetris AI

A Python-based graphical visualization tool for viewing neural network weights from the Tetris AI model.

## Features

- **Heatmap Visualization**: Color-coded matrices showing weight values
- **Statistical Analysis**: Mean, std, min, max for all weight layers
- **Distribution Plots**: Histograms of bias values and weight magnitudes
- **Real-time Monitoring**: Watch weights update as the model trains
- **Detailed Views**: Focus on specific layers with detailed visualizations

## Installation

The visualizer requires Python 3 with numpy and matplotlib:

```bash
pip install numpy matplotlib
```

Or install from requirements.txt:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Visualization

Display all weight matrices:

```bash
python3 visualize_weights.py
```

### Save Visualization to File

```bash
python3 visualize_weights.py --save weights.png
```

### Monitor Model Changes in Real-time

Watch the model file and update visualization when weights change:

```bash
python3 visualize_weights.py --monitor
```

Or with custom interval (default is 2 seconds):

```bash
python3 visualize_weights.py --monitor --interval 5.0
```

### Detailed Weights1 Visualization

Show detailed view of Input→Hidden layer weights:

```bash
python3 visualize_weights.py --detail
```

### Custom Model File

Specify a different model file:

```bash
python3 visualize_weights.py --file my_model.txt
```

## What You'll See

The visualization includes:

1. **Weights1 Heatmap** (29×64): Input features → Hidden neurons
   - Color coding: Blue (negative) → White (zero) → Red (positive)
   - Shows which input features connect strongly to which hidden neurons

2. **Weights2 Heatmap** (64×1): Hidden neurons → Output
   - Shows which hidden neurons contribute most to the Q-value

3. **Bias1 Distribution**: Histogram of hidden layer biases
   - Shows the distribution of bias values

4. **Bias2 Value**: Output layer bias (single value)

5. **Weight Magnitude Distribution**: Histogram of all weight absolute values
   - Shows how weights are distributed in magnitude

## Understanding the Visualizations

### Color Scheme
- **Blue**: Negative weights (inhibitory connections)
- **White**: Near-zero weights (weak connections)
- **Red**: Positive weights (excitatory connections)
- **Intensity**: Darker = stronger magnitude

### What to Look For

**Good Learning Signs:**
- Weights are not all zero (network is learning)
- Some structure/patterns in the heatmap (not completely random)
- Bias values are reasonable (not extreme)
- Weight distribution shows variety (not all same value)

**Potential Issues:**
- All weights very small → learning might be too slow
- All weights very large → might be unstable
- Completely random pattern → network might not be learning structure
- Extreme bias values → might indicate numerical issues

## Example Workflow

1. **Start Tetris training** in one terminal:
   ```bash
   ./tetris
   ```

2. **Start visualizer in another terminal**:
   ```bash
   python3 visualize_weights.py --monitor
   ```

3. **Watch weights update** as the model trains. The visualization will automatically refresh when `tetris_model.txt` is updated (every 100 training episodes).

## Command Line Options

```
usage: visualize_weights.py [-h] [--file FILE] [--save SAVE] [--monitor] 
                            [--interval INTERVAL] [--detail]

optional arguments:
  -h, --help            show this help message and exit
  --file FILE, -f FILE  Path to model file (default: tetris_model.txt)
  --save SAVE, -s SAVE Save visualization to file (e.g., weights.png)
  --monitor, -m         Monitor model file for changes
  --interval INTERVAL, -i INTERVAL
                        Monitoring interval in seconds (default: 2.0)
  --detail, -d          Show detailed Weights1 visualization
```

## Tips

- Use `--monitor` mode to watch training progress in real-time
- Save visualizations periodically with `--save` to track weight evolution
- The detailed view (`--detail`) is useful for understanding which input features are most important
- Compare visualizations from different training stages to see how weights evolve

