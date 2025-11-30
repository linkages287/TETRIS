# Dynamic Weight Visualization Guide

## Overview

The Python weight visualizer now supports **dynamic updates** - the plots refresh automatically when the model file changes, without creating new windows or closing/reopening the display.

## Key Features

✅ **Single Window**: One window stays open and updates in place  
✅ **Real-time Updates**: Plots refresh automatically when `tetris_model.txt` changes  
✅ **Non-blocking**: Uses matplotlib's interactive mode for smooth updates  
✅ **Efficient**: Only updates data, doesn't recreate entire figure  

## Usage

### Basic Monitoring (Recommended)

```bash
python3 visualize_weights.py --monitor
```

This will:
1. Load the initial model and display it
2. Monitor `tetris_model.txt` for changes (default: every 2 seconds)
3. **Update the same window** when weights change
4. Show console messages when updates occur

### Custom Monitoring Interval

```bash
python3 visualize_weights.py --monitor --interval 1.0
```

Monitor every 1 second (faster updates, more CPU usage)

### Save Static Snapshot

```bash
python3 visualize_weights.py --save weights.png
```

Saves a single snapshot to file (doesn't monitor)

## How It Works

### Interactive Mode
- Uses `plt.ion()` to enable interactive matplotlib mode
- Keeps figure and axes references for reuse
- Updates image data and histograms in place

### Update Process
1. **Detect Change**: Checks file modification time
2. **Load New Data**: Reads updated weights from file
3. **Update Plots**:
   - Heatmaps: Updates image data with `set_data()` and `set_clim()`
   - Histograms: Clears and redraws with new data
   - Titles: Updates statistics and timestamps
4. **Refresh Display**: Calls `canvas.draw()` and `canvas.flush_events()`

### What Gets Updated

- ✅ **Weights1 Heatmap**: Color-coded weight matrix (Input → Hidden)
- ✅ **Weights2 Heatmap**: Weight matrix (Hidden → Output)
- ✅ **Bias1 Histogram**: Distribution of hidden layer biases
- ✅ **Bias2 Bar**: Output bias value
- ✅ **Weight Magnitude Histogram**: Distribution of all weight magnitudes
- ✅ **Titles**: Statistics, timestamps, metadata (epsilon, games, scores)

## Console Output

When monitoring, you'll see:

```
Monitoring tetris_model.txt for changes (every 2.0s)
Press Ctrl+C to stop
------------------------------------------------------------
[14:30:15] 📊 Initial model loaded - displaying visualization
[14:30:25] ○ Checking... | File age: 10.2s | Checks: 5
[14:30:35] ○ Checking... | File age: 0.3s | Checks: 10
============================================================
[14:30:35] ✅ NEW WEIGHTS LOADED! (Load #2)
    Load duration: 12.5ms
    Timestamp: 2025-11-30 14:30:35
============================================================
[14:30:35] ⚡ WEIGHTS UPDATED!
  Weights1 - Max change: 0.023456, Mean change: 0.001234
  Weights2 - Max change: 0.012345, Mean change: 0.000567
------------------------------------------------------------
```

## Tips

### Best Performance
- Use `--interval 2.0` or higher for less CPU usage
- Close other matplotlib windows to reduce memory
- Use `--save` for static snapshots instead of monitoring

### Troubleshooting

**Window doesn't update:**
- Check that matplotlib backend supports interactive mode
- Try: `export MPLBACKEND=Qt5Agg` (or TkAgg)
- Restart the visualizer

**Updates are slow:**
- Increase `--interval` (e.g., `--interval 5.0`)
- Close other applications
- Check file I/O performance

**Window closes unexpectedly:**
- Check console for error messages
- Ensure model file format is correct
- Verify file permissions

## Example Workflow

1. **Start training:**
   ```bash
   ./tetris
   ```

2. **Start monitoring (in another terminal):**
   ```bash
   python3 visualize_weights.py --monitor
   ```

3. **Watch the plots update** as the network trains!

4. **Stop monitoring:** Press `Ctrl+C`

## Technical Details

### Matplotlib Backend
The visualizer uses matplotlib's interactive mode (`plt.ion()`). Supported backends:
- `Qt5Agg` / `QtAgg` (recommended)
- `TkAgg`
- `GTKAgg`

Check your backend:
```bash
python3 -c "import matplotlib; print(matplotlib.get_backend())"
```

### Memory Management
- Figure is kept open during monitoring
- Only data is updated, not the figure structure
- Memory usage remains constant
- Figure closes on `Ctrl+C` or error

### Update Frequency
- Default: Check every 2 seconds
- Minimum recommended: 0.5 seconds
- Maximum practical: 10 seconds
- Balance between responsiveness and CPU usage

## Comparison: Before vs After

### Before (Static)
- ❌ New window opened each update
- ❌ Old windows stayed open (memory leak)
- ❌ Had to manually close windows
- ❌ No smooth updates

### After (Dynamic)
- ✅ Single window updates in place
- ✅ Efficient memory usage
- ✅ Smooth, automatic updates
- ✅ Better user experience

---

**Enjoy watching your neural network learn in real-time!** 🎯

