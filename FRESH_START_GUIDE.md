# Fresh Start Guide - Testing Weight Saturation Fixes

## Why Fresh Start?

The current model has **56.4% of weights saturated at 10.0**, which means:
- ❌ Network can't learn new patterns
- ❌ Performance was degrading cyclically
- ❌ The fixes will immediately clip all 10.0 weights to 2.0 (sudden change)

**Starting fresh allows:**
- ✅ Clean initialization with Xavier/Glorot method
- ✅ Proper weight ranges from the beginning
- ✅ See the true impact of all fixes
- ✅ Better learning curve

---

## What Was Backed Up

- `tetris_model_saturated_backup.txt` - Original saturated model
- `tetris_model_best_saturated_backup.txt` - Original best model (if existed)
- Full code backup: `TETRIS_backup_20251130_141628/`

---

## Starting Fresh Training

### Option 1: Run with Default Settings
```bash
./tetris
```

This will:
- Create a new `tetris_model.txt` with Xavier initialization
- Start training from scratch
- Apply all the new fixes (gradient clipping, weight decay, etc.)

### Option 2: Monitor Training in Real-Time
```bash
# Terminal 1: Run the game
./tetris

# Terminal 2: Monitor weights
python3 visualize_weights.py --monitor
```

---

## What to Expect

### Initial State (First Few Games)
- **Weights:** Should be in range [-0.3, 0.3] (Xavier initialization)
- **Epsilon:** Starts at 1.0 (100% exploration)
- **Score:** Low (random exploration)

### Early Training (Games 10-50)
- **Weights:** Gradually increase/decrease, staying in [-2.0, 2.0] range
- **Epsilon:** Decaying slowly (adaptive based on performance)
- **Score:** Should improve steadily

### Mid Training (Games 50-200)
- **Weights:** Should be well-distributed, NOT saturated
- **Mean weight:** Should be around 0.5-1.0 (not 8.46 like before)
- **Score:** Should continue improving
- **No cyclic degradation**

### Success Indicators
✅ **Weight distribution:** Most weights in [-1, 1] range  
✅ **No saturation:** < 5% of weights at clipping limits  
✅ **Stable learning:** Average score increases over time  
✅ **No degradation:** Performance doesn't collapse after improvement  

---

## Monitoring Commands

### Check Weight Statistics
```bash
python3 -c "
import numpy as np
with open('tetris_model.txt', 'r') as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]
weights1 = []
for i in range(29):
    values = list(map(float, lines[i].split()))
    weights1.extend(values)
weights1 = np.array(weights1)
print(f'Mean: {weights1.mean():.3f}')
print(f'Max: {weights1.max():.3f}')
print(f'Min: {weights1.min():.3f}')
print(f'At max (2.0): {np.sum(weights1 >= 1.99)} ({100*np.sum(weights1 >= 1.99)/len(weights1):.1f}%)')
print(f'In [-1,1]: {np.sum(np.abs(weights1) <= 1.0)} ({100*np.sum(np.abs(weights1) <= 1.0)/len(weights1):.1f}%)')
"
```

### Visual Monitoring
```bash
python3 visualize_weights.py --monitor --interval 5
```

---

## Comparison: Before vs After

### Before Fixes (Saturated Model)
- Mean weight: **8.46** (very high)
- 56.4% at max (10.0)
- Only 13.4% in normal range [-1, 1]
- Network couldn't learn

### After Fixes (Expected)
- Mean weight: **~0.5-1.0** (normal)
- < 5% at clipping limits
- Most weights in [-1, 1] range
- Network learns continuously

---

## If You Want to Restore Old Model

```bash
# Restore saturated model
cp tetris_model_saturated_backup.txt tetris_model.txt

# Or restore from full backup
cp -r /home/linkages/cursor/TETRIS_backup_20251130_141628/* /home/linkages/cursor/TETRIS/
```

---

## Troubleshooting

### If weights still saturate:
1. Check that the new code is running (recompile: `make clean && make`)
2. Verify gradient clipping is working (should see gradients < 1.0)
3. Check learning rate (should be 0.001)

### If learning is too slow:
- This is normal initially - the network is learning properly now
- Give it 50-100 games to see improvement
- The old "fast" learning was actually saturation, not real learning

### If performance is worse initially:
- Expected - starting from scratch
- Should improve steadily after 20-30 games
- Old model had saturated weights that couldn't adapt

---

## Expected Timeline

- **Games 0-20:** Random exploration, low scores
- **Games 20-50:** Learning begins, scores improve
- **Games 50-100:** Steady improvement, weights stabilize
- **Games 100+:** Continued learning, no saturation

---

## Success Criteria

After 100+ games, you should see:
1. ✅ Average score increasing over time
2. ✅ Weights distributed normally (not saturated)
3. ✅ No cyclic performance degradation
4. ✅ Epsilon adapting based on performance
5. ✅ Network learning new patterns continuously

