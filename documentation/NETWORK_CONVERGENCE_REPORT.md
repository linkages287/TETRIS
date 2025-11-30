# Neural Network Convergence Detection Report

**Date:** 2025-11-30  
**Project:** Tetris AI Reinforcement Learning

---

## Executive Summary

This report explains how to detect when the neural network has converged during training. Convergence indicates that the network has learned optimal or near-optimal strategies and is no longer improving significantly.

---

## What is Network Convergence?

**Convergence** occurs when the neural network has reached a stable state where:
1. **Performance plateaus** - The network stops improving or improves very slowly
2. **Weights stabilize** - Weight updates become minimal
3. **Error stabilizes** - Training error reaches a minimum and fluctuates around it
4. **Exploration decreases** - Epsilon reaches minimum, network relies on learned knowledge

---

## Key Metrics to Monitor

### 1. **Average Score (Most Important)**

**What it is:**
- Exponential moving average of game scores
- Displayed as "Avg" in the game interface
- Most reliable indicator of learning progress

**Convergence Signs:**
- ✅ **Converged**: Average score plateaus for 500+ games with <5% variation
- ✅ **Converged**: Average score consistently within 10% of best score
- ⚠️ **Not Converged**: Average score still increasing steadily
- ⚠️ **Not Converged**: Large fluctuations (>20%) in average score

**Example:**
```
Games 1000-1500: Avg = 3200 ± 50 (stable) → Likely converged
Games 1000-1500: Avg = 2000 → 3500 (increasing) → Not converged
```

### 2. **Best Score**

**What it is:**
- Highest score achieved during training
- Shows the network's peak performance potential

**Convergence Signs:**
- ✅ **Converged**: Best score hasn't increased for 1000+ games
- ✅ **Converged**: Best score is 2-3x the average (normal distribution)
- ⚠️ **Not Converged**: Best score still increasing periodically

**Example:**
```
Best: 373,894 (unchanged for 2000 games)
Avg: 3,294
Ratio: 113:1 → Normal, likely converged
```

### 3. **Training Error (Batch Error)**

**What it is:**
- Average error between predicted and target Q-values
- Displayed during training
- Lower error = better predictions

**Convergence Signs:**
- ✅ **Converged**: Error stabilizes around 0.1-1.0 (depending on scale)
- ✅ **Converged**: Error fluctuations <10% of mean value
- ⚠️ **Not Converged**: Error still decreasing significantly
- ⚠️ **Not Converged**: Error >10.0 (network still learning)

**Example:**
```
Error: 0.5 ± 0.05 (stable for 1000+ episodes) → Converged
Error: 5.0 → 2.0 → 1.0 (decreasing) → Not converged
```

### 4. **Epsilon (Exploration Rate)**

**What it is:**
- Probability of random exploration vs. using learned knowledge
- Starts at 1.0 (100% exploration), decays to epsilon_min

**Convergence Signs:**
- ✅ **Converged**: Epsilon at minimum (0.05-0.1) for 500+ games
- ✅ **Converged**: Epsilon stable (not increasing due to poor performance)
- ⚠️ **Not Converged**: Epsilon still decreasing
- ⚠️ **Not Converged**: Epsilon increasing (performance degrading)

**Example:**
```
Epsilon: 0.05 (stable, at minimum) → Converged
Epsilon: 0.8 → 0.6 → 0.4 (decreasing) → Not converged
```

### 5. **Weight Stability**

**What it is:**
- How much network weights change during updates
- Can be monitored using the weight visualizer

**Convergence Signs:**
- ✅ **Converged**: Weight changes <0.1% per update
- ✅ **Converged**: Weight distribution stable (no saturation)
- ⚠️ **Not Converged**: Large weight changes (>1% per update)
- ⚠️ **Not Converged**: Weights still converging to similar values

**How to Check:**
- Use `./weight_visualizer` to monitor weight changes
- Look for stable color patterns (weights not changing much)
- Check weight statistics in console output

---

## Convergence Detection Criteria

### Primary Criteria (All Must Be True)

1. **Average Score Stability**
   - Average score within ±5% for 500+ consecutive games
   - No significant upward trend for 1000+ games

2. **Error Stability**
   - Training error stable (fluctuations <10% of mean)
   - Error <2.0 for Q-learning (reasonable threshold)

3. **Epsilon at Minimum**
   - Epsilon = epsilon_min (0.05-0.1) for 500+ games
   - Epsilon not increasing (performance not degrading)

4. **Best Score Plateau**
   - Best score unchanged for 1000+ games
   - Or only occasional small improvements (<5%)

### Secondary Criteria (Supporting Evidence)

5. **Weight Stability**
   - Weight changes minimal (<0.1% per update)
   - No weight saturation (all weights not at max/min)

6. **Performance Consistency**
   - Score variance decreasing
   - Network making consistent, strategic moves

---

## Convergence Detection Methods

### Method 1: Visual Inspection (Easiest)

**Monitor the game display:**
```
Games: 1500
Episodes: 84057
Best: 373894
Avg: 3294.48    ← Watch this!
Epsilon: 0.05   ← Should be at minimum
Buffer: 10000
```

**Check for:**
- Avg score stable for many games
- Epsilon at minimum
- Best score not increasing

### Method 2: Statistical Analysis

**Calculate metrics over sliding window:**

```python
# Pseudo-code for convergence detection
def is_converged(scores, window_size=500):
    recent_scores = scores[-window_size:]
    
    # Check average stability
    mean = np.mean(recent_scores)
    std = np.std(recent_scores)
    cv = std / mean  # Coefficient of variation
    
    # Check trend
    trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
    
    # Converged if:
    # - Low variation (CV < 0.1)
    # - No upward trend (trend < 0.01)
    return cv < 0.1 and abs(trend) < 0.01
```

### Method 3: Automated Monitoring

**Add convergence detection to the code:**

```cpp
// In RLAgent class
bool checkConvergence() {
    // Check average score stability
    if (recent_scores.size() < 500) return false;
    
    double mean = calculateMean(recent_scores);
    double std = calculateStd(recent_scores);
    double cv = std / mean;
    
    // Check epsilon
    bool epsilon_stable = (epsilon <= epsilon_min + 0.01);
    
    // Check error
    bool error_stable = (last_batch_error < 2.0 && 
                         error_variance < 0.1);
    
    return (cv < 0.1 && epsilon_stable && error_stable);
}
```

---

## Typical Convergence Timeline

### Early Training (Games 0-500)
- **Avg Score**: 0-500 (random play)
- **Epsilon**: 1.0 → 0.8 (high exploration)
- **Error**: 10-50 (high, network learning)
- **Status**: ❌ Not converged

### Learning Phase (Games 500-2000)
- **Avg Score**: 500 → 2000 (rapid improvement)
- **Epsilon**: 0.8 → 0.2 (decreasing exploration)
- **Error**: 50 → 5 (decreasing)
- **Status**: ❌ Not converged (still learning)

### Refinement Phase (Games 2000-5000)
- **Avg Score**: 2000 → 3000 (slower improvement)
- **Epsilon**: 0.2 → 0.1 (approaching minimum)
- **Error**: 5 → 1 (stabilizing)
- **Status**: ⚠️ Possibly converging

### Convergence Phase (Games 5000+)
- **Avg Score**: 3000 ± 100 (stable)
- **Epsilon**: 0.05-0.1 (at minimum)
- **Error**: 0.5-1.0 (stable)
- **Status**: ✅ Likely converged

---

## What to Do When Converged

### 1. **Save the Model**
```bash
# Best model is automatically saved to tetris_model_best.txt
# Or manually:
cp tetris_model.txt tetris_model_converged.txt
```

### 2. **Evaluate Performance**
- Run the network in test mode (no training)
- Play multiple games to verify consistency
- Check if performance matches training metrics

### 3. **Consider Further Training**
- **If converged early** (<2000 games): May need more exploration
- **If converged late** (>10000 games): Likely optimal
- **If performance low**: May need hyperparameter tuning

### 4. **Document Results**
- Record final metrics
- Note convergence game/episode count
- Save best model for future use

---

## Common Convergence Issues

### Issue 1: Premature Convergence
**Symptoms:**
- Converges too early (<1000 games)
- Average score low (<1000)
- Network making poor decisions

**Causes:**
- Epsilon decaying too fast
- Learning rate too high
- Network stuck in local minimum

**Solutions:**
- Increase epsilon_min (0.1 → 0.15)
- Slow epsilon decay (0.998 → 0.999)
- Reset model and retrain

### Issue 2: Non-Convergence
**Symptoms:**
- Average score keeps increasing after 10000+ games
- Error still decreasing
- Weights still changing significantly

**Causes:**
- Learning rate too low
- Network capacity too high
- Reward shaping issues

**Solutions:**
- Check if improvement is meaningful (>5%)
- May be normal for complex tasks
- Consider if convergence is necessary

### Issue 3: Oscillating Performance
**Symptoms:**
- Average score fluctuates widely
- Performance improves then degrades
- Epsilon increasing/decreasing cyclically

**Causes:**
- Overfitting to recent experiences
- Replay buffer too small
- Learning rate too high

**Solutions:**
- Increase replay buffer size
- Reduce learning rate
- Add more regularization

---

## Monitoring Tools

### 1. **In-Game Display**
- Real-time metrics during training
- Shows: Games, Episodes, Best, Avg, Epsilon, Buffer

### 2. **Weight Visualizer**
```bash
./weight_visualizer
```
- Monitor weight changes visually
- Detect weight saturation
- See if weights are stabilizing

### 3. **Console Output**
- Training error messages
- Epsilon change notifications
- Model save confirmations

### 4. **Model File Analysis**
```bash
# Check model file for training state
grep "TRAINING_EPISODES\|AVERAGE_SCORE\|EPSILON" tetris_model.txt
```

---

## Convergence Checklist

Use this checklist to verify convergence:

- [ ] Average score stable (±5%) for 500+ games
- [ ] Training error <2.0 and stable
- [ ] Epsilon at minimum (0.05-0.1) for 500+ games
- [ ] Best score unchanged for 1000+ games
- [ ] Weight changes minimal (<0.1% per update)
- [ ] Performance consistent (low variance)
- [ ] Network making strategic moves (not random)

**If all checked:** ✅ Network has converged

---

## Recommendations

### For This Tetris AI:

1. **Monitor Average Score** - Most reliable indicator
2. **Wait for Stability** - At least 500 games of stable performance
3. **Check Epsilon** - Should be at minimum when converged
4. **Verify Error** - Should be <1.0 and stable
5. **Save Best Model** - Automatically saved when best score reached

### Typical Convergence:
- **Games**: 3000-10000
- **Episodes**: 50000-200000
- **Average Score**: 2000-5000 (depends on difficulty)
- **Error**: 0.5-1.0
- **Epsilon**: 0.05-0.1

---

## Conclusion

Network convergence is detected by monitoring multiple metrics over time:
- **Primary**: Average score stability, error stability, epsilon at minimum
- **Secondary**: Weight stability, performance consistency

The network is likely converged when:
1. Average score is stable for 500+ games
2. Training error is low and stable
3. Epsilon is at minimum
4. Best score has plateaued

Use the in-game display, weight visualizer, and console output to monitor these metrics and detect convergence.

---

## References

- Training metrics displayed in-game
- Model file: `tetris_model.txt` (contains training state)
- Weight visualizer: `./weight_visualizer`
- Main README: `../README.md`

