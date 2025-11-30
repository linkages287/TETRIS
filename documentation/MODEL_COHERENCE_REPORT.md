# Tetris Model File Coherence Report

**File:** `tetris_model.txt`  
**Analysis Date:** 2025-11-30  
**Last Saved:** 2025-11-30 21:21:46

---

## ✅ Overall Status: **COHERENT**

The model file structure is correct and all data is valid.

---

## File Structure Analysis

### Header Section ✅
```
# Neural Network Model File
# Saved: 2025-11-30 21:21:46
# Filename: tetris_model.txt
```
- ✅ Header present and correctly formatted
- ✅ Timestamp included
- ✅ Filename included

### Data Section ✅

**Expected Structure:**
- Weights1: 29 rows × 64 columns = 1,856 values
- Bias1: 1 row × 64 values = 64 values
- Weights2: 64 rows × 1 column = 64 values
- Bias2: 1 row × 1 value = 1 value
- **Total:** 95 data lines

**Actual Structure:**
- ✅ Weights1: 29 rows, 64 values each = 1,856 values
- ✅ Bias1: 1 row, 64 values = 64 values
- ✅ Weights2: 64 rows, 1 value each = 64 values
- ✅ Bias2: 1 row, 1 value = 1 value
- ✅ Metadata: 11 additional lines (expected)

---

## Weight Statistics

### Weights1 (Input → Hidden Layer)
- **Total Values:** 1,856 (29 × 64) ✅
- **Mean:** -0.023223
- **Standard Deviation:** 0.129287
- **Range:** [-0.254067, 0.131958]
- **Clipping Range:** [-2.0, 2.0] ✅ **All values within range**
- **NaN Check:** ✅ No NaN values
- **Inf Check:** ✅ No Inf values

**Observation:** 
- ⚠️ **Low variance detected** - Many weights have very similar values
- Most values cluster around -0.14 to 0.13 range
- This suggests weights may be converging or not learning diverse patterns

### Bias1 (Hidden Layer)
- **Total Values:** 64 ✅
- **Mean:** 0.666152
- **Range:** [0.665196, 0.666351]
- **Clipping Range:** [-2.0, 2.0] ✅ **All values within range**
- **NaN/Inf Check:** ✅ Valid

**Observation:**
- ⚠️ **Very low variance** - All bias values are nearly identical (~0.666)
- This suggests hidden neurons may not be learning distinct features

### Weights2 (Hidden → Output Layer)
- **Total Values:** 64 (64 × 1) ✅
- **Mean:** 1.214436
- **Range:** [1.213290, 1.219550]
- **Clipping Range:** [-20.0, 20.0] ✅ **All values within range**
- **NaN/Inf Check:** ✅ Valid

**Observation:**
- ⚠️ **Very low variance** - All weights are nearly identical (~1.213)
- This suggests the output layer is not differentiating between hidden neurons

### Bias2 (Output Layer)
- **Value:** -19.976300 ✅
- **Clipping Range:** [-20.0, 20.0] ✅ **Within range**
- **NaN/Inf Check:** ✅ Valid

---

## Training Metadata

```
FILENAME: tetris_model.txt
EPSILON: 0.0240217 (2.4% exploration)
EPSILON_MIN: 0.1
EPSILON_DECAY: 0.998
LEARNING_RATE: 0.002
GAMMA: 0.99
TRAINING_EPISODES: 84,057
TOTAL_GAMES: 1,185
BEST_SCORE: 373,894
AVERAGE_SCORE: 3,294.48
PREVIOUS_AVG_SCORE: 3,294.48
```

**Analysis:**
- ✅ High training episodes (84K+) - Network has trained extensively
- ✅ Good best score (373K) - Network has achieved high performance
- ⚠️ Epsilon is very low (0.024) - Minimal exploration
- ⚠️ Epsilon_MIN is 0.1 but epsilon is 0.024 - This is inconsistent (epsilon should not go below epsilon_min)

---

## Potential Issues Identified

### 1. ⚠️ Low Weight Variance
**Problem:** Weights show very little variation
- Weights1: Most values cluster around -0.14 to 0.13
- Weights2: All values are nearly identical (~1.213)
- Bias1: All values nearly identical (~0.666)

**Impact:** Network may not be learning diverse features, potentially limiting expressiveness

**Possible Causes:**
- Gradient clipping too aggressive
- Learning rate too low
- Weight decay too high
- Network converged to local minimum

### 2. ⚠️ Epsilon Inconsistency
**Problem:** Epsilon (0.024) is below epsilon_min (0.1)
- This should not happen - epsilon should never go below epsilon_min

**Impact:** May indicate a bug in epsilon update logic

### 3. ✅ File Format
**Status:** Correct
- Header format: ✅ Valid
- Data structure: ✅ Valid
- Metadata: ✅ Present and complete

---

## Recommendations

### 1. **Weight Diversity**
- Consider increasing learning rate slightly
- Reduce weight decay if it's too aggressive
- Check if gradient clipping is preventing learning

### 2. **Epsilon Bug**
- Fix epsilon update logic to ensure it never goes below epsilon_min
- Current epsilon (0.024) should be clamped to epsilon_min (0.1)

### 3. **Model Performance**
- Despite low variance, the model shows good performance (best score: 373K)
- Monitor if performance plateaus or degrades
- Consider resetting weights if learning stalls

---

## Validation Summary

| Check | Status | Details |
|-------|--------|---------|
| File Structure | ✅ PASS | Correct number of lines and sections |
| Header Format | ✅ PASS | Timestamp and filename present |
| Weights1 Count | ✅ PASS | 1,856 values (29×64) |
| Weights1 Range | ✅ PASS | All in [-2.0, 2.0] |
| Weights1 NaN/Inf | ✅ PASS | No invalid values |
| Bias1 Count | ✅ PASS | 64 values |
| Bias1 Range | ✅ PASS | All in [-2.0, 2.0] |
| Weights2 Count | ✅ PASS | 64 values |
| Weights2 Range | ✅ PASS | All in [-20.0, 20.0] |
| Bias2 Value | ✅ PASS | Valid value |
| Metadata | ✅ PASS | Complete training state |
| **Overall** | ✅ **COHERENT** | File is valid and loadable |

---

## Conclusion

The model file is **structurally coherent** and can be loaded successfully. All weight values are within expected ranges and contain no invalid (NaN/Inf) values.

However, there are **two concerns**:
1. **Low weight variance** - May limit network expressiveness
2. **Epsilon inconsistency** - Should be fixed in code

The model appears to be functional and has achieved good performance (373K best score), but the low variance suggests the network may benefit from further training or parameter adjustments.

