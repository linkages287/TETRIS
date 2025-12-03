============================================================
Neural Network Model Analysis
============================================================

Analyzing: tetris_model.txt


============================================================
Analysis: Input-to-Hidden Weights (weights1)
============================================================
Shape: Expected (29, 64), Got (29, 64)

Statistics:
  Min:    -0.873488
  Max:    0.928993
  Mean:   -0.028674
  Std:    0.321910
  Median: 0.002908
✓ All values within expected range [-2.0, 2.0]

Most common values:
  0.113001: 2 times (0.11%)
  0.109570: 2 times (0.11%)
  0.928993: 1 times (0.05%)
  0.928645: 1 times (0.05%)
  0.926891: 1 times (0.05%)

Variance: 0.103626

============================================================
Analysis: Hidden Layer Biases (bias1)
============================================================
Shape: Expected (1, 64), Got (1, 64)

Statistics:
  Min:    -0.329148
  Max:    1.971090
  Mean:   0.863999
  Std:    0.587386
  Median: 0.998128
✓ All values within expected range [-2.0, 2.0]

Most common values:
  1.971090: 1 times (1.56%)
  1.970960: 1 times (1.56%)
  1.970840: 1 times (1.56%)
  1.970750: 1 times (1.56%)
  1.970620: 1 times (1.56%)

Variance: 0.345022

============================================================
Analysis: Hidden-to-Output Weights (weights2)
============================================================
Shape: Expected (64, 1), Got (64, 1)

Statistics:
  Min:    -3.827720
  Max:    5.349470
  Mean:   0.384910
  Std:    3.707270
  Median: -1.586110
✓ All values within expected range [-20.0, 20.0]

Most common values:
  5.349470: 1 times (1.56%)
  5.126990: 1 times (1.56%)
  5.086500: 1 times (1.56%)
  5.067620: 1 times (1.56%)
  5.066820: 1 times (1.56%)

Variance: 13.743852
⚠️  WARNING: Very high variance - weights may be unstable

============================================================
Analysis: Output Layer Bias (bias2)
============================================================
Shape: Expected (1, 1), Got (1, 1)

Statistics:
  Min:    9.573860
  Max:    9.573860
  Mean:   9.573860
  Std:    0.000000
  Median: 9.573860
✓ All values within expected range [-20.0, 20.0]

Most common values:
  9.573860: 1 times (100.00%)
  ⚠️  WARNING: High saturation! 100.00% of values are identical

Variance: 0.000000
⚠️  WARNING: Very low variance - weights may be too uniform

============================================================
Training Metadata Analysis
============================================================

Metadata values:
  FILENAME: tetris_model.txt
  EPSILON: 0.05
  EPSILON_MIN: 0.05
  EPSILON_DECAY: 0.995
  LEARNING_RATE: 0.002
  GAMMA: 0.99
  TRAINING_EPISODES: 18965
  TOTAL_GAMES: 450
  BEST_SCORE: 23212
  AVERAGE_SCORE: 3487.62
  PREVIOUS_AVG_SCORE: 3487.62
✓ Epsilon is valid: 0.05 >= 0.05
✓ Episodes per game: 42.14
✓ Score consistency: Best=23212, Avg=3487.62

✓ Metadata appears consistent

============================================================
Overall Assessment
============================================================
✓ Model appears to be consistent and valid
