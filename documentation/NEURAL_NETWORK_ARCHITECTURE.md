# Neural Network Architecture Explanation

**Generated**: December 4, 2024

## Overview

The Tetris AI uses a **3-layer feedforward neural network** for Q-learning:
- **Input Layer**: 27 neurons (game state features)
- **Hidden Layer**: 64 neurons (processing)
- **Output Layer**: 1 neuron (Q-value prediction)

## Layer Notation

### W1 (Weights1)
- **Full Name**: `weights1`
- **Type**: 2D array `[INPUT_SIZE][HIDDEN_SIZE]`
- **Size**: 27 × 64 = **1,728 weights**
- **Purpose**: Connections from **Input Layer → Hidden Layer**
- **Meaning**: How much each input feature contributes to each hidden neuron

**Example**:
- `weights1[5][10]` = weight connecting input feature #5 to hidden neuron #10
- If `weights1[5][10] = 0.5`, then input feature #5 has a positive influence on hidden neuron #10

---

### B1 (Bias1)
- **Full Name**: `bias1`
- **Type**: 1D array `[HIDDEN_SIZE]`
- **Size**: **64 biases** (one per hidden neuron)
- **Purpose**: Bias terms for the **Hidden Layer**
- **Meaning**: Constant offset added to each hidden neuron's activation

**Example**:
- `bias1[10] = -0.2` means hidden neuron #10 has a negative bias
- This shifts the activation threshold for that neuron

---

### W2 (Weights2)
- **Full Name**: `weights2`
- **Type**: 2D array `[HIDDEN_SIZE][OUTPUT_SIZE]`
- **Size**: 64 × 1 = **64 weights**
- **Purpose**: Connections from **Hidden Layer → Output Layer**
- **Meaning**: How much each hidden neuron contributes to the final Q-value

**Example**:
- `weights2[10][0] = 0.3` means hidden neuron #10 has a positive influence on the Q-value
- Higher weight = more important hidden neuron for Q-value prediction

---

### B2 (Bias2)
- **Full Name**: `bias2`
- **Type**: 1D array `[OUTPUT_SIZE]`
- **Size**: **1 bias** (one for output neuron)
- **Purpose**: Bias term for the **Output Layer**
- **Meaning**: Constant offset added to the final Q-value

**Example**:
- `bias2[0] = 5.0` means the Q-value has a base offset of +5.0
- This shifts all Q-values by a constant amount

---

## Network Flow

```
Input Layer (27 features)
    ↓ [W1: 27×64 weights]
Hidden Layer (64 neurons) + [B1: 64 biases]
    ↓ [W2: 64×1 weights]
Output Layer (1 neuron) + [B2: 1 bias]
    ↓
Q-value (single number)
```

## Mathematical Representation

### Forward Pass:

1. **Hidden Layer Activation**:
   ```
   hidden[i] = leaky_relu(
       bias1[i] + 
       sum(input[j] * weights1[j][i] for j in 0..26)
   )
   ```

2. **Output Layer (Q-value)**:
   ```
   q_value = bias2[0] + 
             sum(hidden[i] * weights2[i][0] for i in 0..63)
   ```

## Parameter Counts

| Layer | Weights | Biases | Total Parameters |
|-------|---------|--------|------------------|
| **W1** | 1,728 | - | 1,728 |
| **B1** | - | 64 | 64 |
| **W2** | 64 | - | 64 |
| **B2** | - | 1 | 1 |
| **Total** | 1,792 | 65 | **1,857 parameters** |

## What Each Layer Does

### Input Layer (27 features)
- Receives game state features:
  - 10 column heights
  - 3 board quality metrics (max height, holes, bumpiness)
  - 7 current piece type (one-hot)
  - 7 next piece type (one-hot)

### Hidden Layer (64 neurons)
- **Processes** input features through non-linear activation (Leaky ReLU)
- **Learns** complex patterns and relationships
- **Extracts** meaningful features from raw game state

### Output Layer (1 neuron)
- **Produces** Q-value (expected future reward)
- **Used** to select best move (highest Q-value)

## Training Process

During training, all parameters (W1, B1, W2, B2) are updated using **backpropagation**:

1. **Forward pass**: Calculate Q-value from current state
2. **Calculate error**: Compare predicted Q-value to target Q-value
3. **Backward pass**: Update all weights and biases to reduce error
4. **Gradient clipping**: Prevent weight explosion
5. **Weight clipping**: Keep weights in valid range [-100, 100]

## Monitoring

The display shows statistics for each layer:

```
W1: m=1.041 Sat=0.12% Var=3.431  (Weights1: mean, saturation, variance)
B1: Sat=1.56% Var=6.656          (Bias1: saturation, variance)
W2: m=-0.773 Sat=1.56% Var=4.549  (Weights2: mean, saturation, variance)
B2: Sat=0.0% Var=0.000            (Bias2: saturation, variance - always 0 for single value)
```

## Key Points

1. **W1**: Input → Hidden connections (largest layer, 1,728 weights)
2. **B1**: Hidden layer biases (64 values)
3. **W2**: Hidden → Output connections (64 weights)
4. **B2**: Output bias (1 value, always shows 0% saturation)

## Summary

- **W1** = Input-to-Hidden weights (27×64 = 1,728)
- **B1** = Hidden layer biases (64)
- **W2** = Hidden-to-Output weights (64×1 = 64)
- **B2** = Output bias (1)

Together, these **1,857 parameters** learn to predict Q-values for Tetris moves!



