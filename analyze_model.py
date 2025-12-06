#!/usr/bin/env python3
"""
Model Analysis Tool
Analyzes the neural network model file and provides comprehensive statistics.
"""

import sys
import numpy as np
from collections import Counter

# Network architecture constants
INPUT_SIZE = 27
HIDDEN_SIZE = 64
OUTPUT_SIZE = 1

def load_model(filename):
    """Load model from file and extract weights/biases."""
    weights1 = []
    bias1 = []
    weights2 = []
    bias2 = []
    metadata = {}
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Parse header/metadata
    data_start = 0
    in_metadata = False
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('# Training State Metadata'):
            in_metadata = True
            continue
        if in_metadata and line.startswith('#'):
            continue
        if in_metadata and line:
            # Parse metadata line (format: KEY VALUE)
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0]
                try:
                    if key == 'EPSILON':
                        metadata['EPSILON'] = float(parts[1])
                    elif key == 'EPSILON_MIN':
                        metadata['EPSILON_MIN'] = float(parts[1])
                    elif key == 'EPSILON_DECAY':
                        metadata['EPSILON_DECAY'] = float(parts[1])
                    elif key == 'LEARNING_RATE':
                        metadata['LEARNING_RATE'] = float(parts[1])
                    elif key == 'GAMMA':
                        metadata['GAMMA'] = float(parts[1])
                    elif key == 'TRAINING_EPISODES':
                        metadata['TRAINING_EPISODES'] = int(parts[1])
                    elif key == 'TOTAL_GAMES':
                        metadata['TOTAL_GAMES'] = int(parts[1])
                    elif key == 'BEST_SCORE':
                        metadata['BEST_SCORE'] = int(parts[1])
                    elif key == 'AVERAGE_SCORE':
                        metadata['AVERAGE_SCORE'] = float(parts[1])
                except:
                    pass
        elif line and not line.startswith('#'):
            if data_start == 0:
                data_start = i
            if in_metadata:
                in_metadata = False
    
    # Parse weights and biases
    current_line = data_start
    
    # Load weights1 (INPUT_SIZE rows, HIDDEN_SIZE columns each)
    for i in range(INPUT_SIZE):
        if current_line >= len(lines):
            break
        line = lines[current_line].strip()
        if line and not line.startswith('#'):
            weights = [float(x) for x in line.split() if x]
            if len(weights) == HIDDEN_SIZE:
                weights1.append(weights)
            current_line += 1
    
    # Load bias1 (HIDDEN_SIZE values)
    if current_line < len(lines):
        line = lines[current_line].strip()
        if line and not line.startswith('#'):
            bias1 = [float(x) for x in line.split() if x]
            current_line += 1
    
    # Load weights2 (HIDDEN_SIZE rows, OUTPUT_SIZE columns each)
    for i in range(HIDDEN_SIZE):
        if current_line >= len(lines):
            break
        line = lines[current_line].strip()
        if line and not line.startswith('#'):
            weights = [float(x) for x in line.split() if x]
            if len(weights) == OUTPUT_SIZE:
                weights2.append(weights)
            current_line += 1
    
    # Load bias2 (OUTPUT_SIZE values)
    if current_line < len(lines):
        line = lines[current_line].strip()
        if line and not line.startswith('#'):
            bias2 = [float(x) for x in line.split() if x]
    
    return weights1, bias1, weights2, bias2, metadata

def analyze_array(arr, name):
    """Analyze an array and return statistics."""
    arr_flat = np.array(arr).flatten()
    arr_flat = arr_flat[np.isfinite(arr_flat)]  # Remove NaN/Inf
    
    if len(arr_flat) == 0:
        return {
            'count': 0,
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'nan_count': 0,
            'inf_count': 0,
            'zero_count': 0,
            'saturation': 0.0
        }
    
    # Count special values
    arr_all = np.array(arr).flatten()
    nan_count = np.sum(np.isnan(arr_all))
    inf_count = np.sum(np.isinf(arr_all))
    zero_count = np.sum(arr_flat == 0.0)
    
    # Calculate saturation (percentage of most common value)
    if len(arr_flat) > 0:
        rounded = np.round(arr_flat, decimals=6)
        counter = Counter(rounded)
        most_common_count = counter.most_common(1)[0][1] if counter else 0
        saturation = (most_common_count / len(arr_flat)) * 100.0
    else:
        saturation = 0.0
    
    return {
        'count': len(arr_flat),
        'mean': float(np.mean(arr_flat)),
        'std': float(np.std(arr_flat)),
        'min': float(np.min(arr_flat)),
        'max': float(np.max(arr_flat)),
        'nan_count': int(nan_count),
        'inf_count': int(inf_count),
        'zero_count': int(zero_count),
        'saturation': saturation
    }

def print_analysis(filename):
    """Print comprehensive model analysis."""
    print("=" * 80)
    print(f"MODEL ANALYSIS: {filename}")
    print("=" * 80)
    print()
    
    try:
        weights1, bias1, weights2, bias2, metadata = load_model(filename)
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        return
    
    # Check if model loaded correctly
    if len(weights1) != INPUT_SIZE:
        print(f"WARNING: Expected {INPUT_SIZE} rows in weights1, got {len(weights1)}")
    if len(bias1) != HIDDEN_SIZE:
        print(f"WARNING: Expected {HIDDEN_SIZE} values in bias1, got {len(bias1)}")
    if len(weights2) != HIDDEN_SIZE:
        print(f"WARNING: Expected {HIDDEN_SIZE} rows in weights2, got {len(weights2)}")
    if len(bias2) != OUTPUT_SIZE:
        print(f"WARNING: Expected {OUTPUT_SIZE} values in bias2, got {len(bias2)}")
    print()
    
    # Print metadata
    print("TRAINING STATE METADATA:")
    print("-" * 80)
    if metadata:
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    else:
        print("  No metadata found")
    print()
    
    # Analyze weights1
    print("WEIGHTS1 (Input -> Hidden):")
    print("-" * 80)
    w1_stats = analyze_array(weights1, "weights1")
    print(f"  Count: {w1_stats['count']}")
    print(f"  Mean:  {w1_stats['mean']:.6f}")
    print(f"  Std:   {w1_stats['std']:.6f}")
    print(f"  Min:   {w1_stats['min']:.6f}")
    print(f"  Max:   {w1_stats['max']:.6f}")
    print(f"  NaN:   {w1_stats['nan_count']}")
    print(f"  Inf:   {w1_stats['inf_count']}")
    print(f"  Zeros: {w1_stats['zero_count']} ({w1_stats['zero_count']/w1_stats['count']*100:.1f}%)")
    print(f"  Saturation: {w1_stats['saturation']:.2f}%")
    
    # Check for issues
    if w1_stats['nan_count'] > 0 or w1_stats['inf_count'] > 0:
        print(f"  ⚠️  ISSUE: Contains NaN/Inf values!")
    if w1_stats['saturation'] > 50.0:
        print(f"  ⚠️  WARNING: High saturation ({w1_stats['saturation']:.1f}%) - weights may be stuck")
    if abs(w1_stats['mean']) > 1.0:
        print(f"  ⚠️  WARNING: Large mean value ({w1_stats['mean']:.3f})")
    if w1_stats['std'] < 0.01:
        print(f"  ⚠️  WARNING: Very low variance ({w1_stats['std']:.6f}) - weights may be too uniform")
    print()
    
    # Analyze bias1
    print("BIAS1 (Hidden Layer):")
    print("-" * 80)
    b1_stats = analyze_array(bias1, "bias1")
    print(f"  Count: {b1_stats['count']}")
    print(f"  Mean:  {b1_stats['mean']:.6f}")
    print(f"  Std:   {b1_stats['std']:.6f}")
    print(f"  Min:   {b1_stats['min']:.6f}")
    print(f"  Max:   {b1_stats['max']:.6f}")
    print(f"  NaN:   {b1_stats['nan_count']}")
    print(f"  Inf:   {b1_stats['inf_count']}")
    print(f"  Zeros: {b1_stats['zero_count']} ({b1_stats['zero_count']/b1_stats['count']*100:.1f}%)")
    print(f"  Saturation: {b1_stats['saturation']:.2f}%")
    
    if b1_stats['nan_count'] > 0 or b1_stats['inf_count'] > 0:
        print(f"  ⚠️  ISSUE: Contains NaN/Inf values!")
    if b1_stats['saturation'] > 50.0:
        print(f"  ⚠️  WARNING: High saturation ({b1_stats['saturation']:.1f}%)")
    print()
    
    # Analyze weights2
    print("WEIGHTS2 (Hidden -> Output):")
    print("-" * 80)
    w2_stats = analyze_array(weights2, "weights2")
    print(f"  Count: {w2_stats['count']}")
    print(f"  Mean:  {w2_stats['mean']:.6f}")
    print(f"  Std:   {w2_stats['std']:.6f}")
    print(f"  Min:   {w2_stats['min']:.6f}")
    print(f"  Max:   {w2_stats['max']:.6f}")
    print(f"  NaN:   {w2_stats['nan_count']}")
    print(f"  Inf:   {w2_stats['inf_count']}")
    print(f"  Zeros: {w2_stats['zero_count']} ({w2_stats['zero_count']/w2_stats['count']*100:.1f}%)")
    print(f"  Saturation: {w2_stats['saturation']:.2f}%")
    
    if w2_stats['nan_count'] > 0 or w2_stats['inf_count'] > 0:
        print(f"  ⚠️  ISSUE: Contains NaN/Inf values!")
    if w2_stats['saturation'] > 50.0:
        print(f"  ⚠️  WARNING: High saturation ({w2_stats['saturation']:.1f}%)")
    print()
    
    # Analyze bias2
    print("BIAS2 (Output Layer):")
    print("-" * 80)
    b2_stats = analyze_array(bias2, "bias2")
    print(f"  Count: {b2_stats['count']}")
    print(f"  Mean:  {b2_stats['mean']:.6f}")
    print(f"  Std:   {b2_stats['std']:.6f}")
    print(f"  Min:   {b2_stats['min']:.6f}")
    print(f"  Max:   {b2_stats['max']:.6f}")
    print(f"  NaN:   {b2_stats['nan_count']}")
    print(f"  Inf:   {b2_stats['inf_count']}")
    print()
    
    # Overall assessment
    print("OVERALL ASSESSMENT:")
    print("-" * 80)
    issues = []
    if w1_stats['nan_count'] > 0 or w1_stats['inf_count'] > 0:
        issues.append("NaN/Inf values in weights1")
    if b1_stats['nan_count'] > 0 or b1_stats['inf_count'] > 0:
        issues.append("NaN/Inf values in bias1")
    if w2_stats['nan_count'] > 0 or w2_stats['inf_count'] > 0:
        issues.append("NaN/Inf values in weights2")
    if b2_stats['nan_count'] > 0 or b2_stats['inf_count'] > 0:
        issues.append("NaN/Inf values in bias2")
    
    if w1_stats['saturation'] > 50.0:
        issues.append("High saturation in weights1")
    if b1_stats['saturation'] > 50.0:
        issues.append("High saturation in bias1")
    if w2_stats['saturation'] > 50.0:
        issues.append("High saturation in weights2")
    
    if abs(w1_stats['mean']) > 1.0:
        issues.append("Large mean in weights1")
    if abs(w2_stats['mean']) > 1.0:
        issues.append("Large mean in weights2")
    
    if issues:
        print("  ⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  ✅ Model appears healthy")
    
    print()
    print("=" * 80)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "tetris_model.txt"
    
    print_analysis(filename)
