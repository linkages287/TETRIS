#!/usr/bin/env python3
"""
Model Analysis Script
Analyzes the neural network model file for training consistency and data integrity.
"""

import sys
import re
import numpy as np
from collections import Counter

def parse_model_file(filename):
    """Parse the model file and extract weights, biases, and metadata."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Skip header comments
    data_start = 0
    metadata_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith('#'):
            continue
        if line.strip().startswith('FILENAME') or line.strip().startswith('EPSILON'):
            metadata_start = i
            break
        if data_start == 0 and line.strip():
            data_start = i
    
    # Parse weights and biases
    weights1 = []
    bias1 = []
    weights2 = []
    bias2 = []
    
    current_section = 'weights1'
    expected_weights1_rows = 29
    expected_hidden_size = 64
    
    for i in range(data_start, metadata_start if metadata_start else len(lines)):
        line = lines[i].strip()
        if not line:
            continue
        
        # Try to parse as numbers
        try:
            values = [float(x) for x in line.split()]
            
            if current_section == 'weights1':
                if len(weights1) < expected_weights1_rows:
                    weights1.append(values)
                else:
                    # Check if this is bias1 (should be 64 values)
                    if len(values) == expected_hidden_size:
                        current_section = 'bias1'
                        bias1 = values
                    else:
                        # Might be weights2 (64 rows of 1 value each)
                        current_section = 'weights2'
                        weights2.append(values)
            elif current_section == 'bias1':
                # After bias1, should be weights2
                if len(values) == 1:
                    # Single value - might be start of weights2 or bias2
                    if len(weights2) == 0:
                        current_section = 'weights2'
                        weights2.append(values)
                    else:
                        # Could be bias2
                        if len(weights2) < expected_hidden_size:
                            weights2.append(values)
                        else:
                            current_section = 'bias2'
                            bias2 = values
                else:
                    weights2.append(values)
            elif current_section == 'weights2':
                if len(weights2) < expected_hidden_size:
                    weights2.append(values)
                else:
                    current_section = 'bias2'
                    bias2 = values
            elif current_section == 'bias2':
                # Should only be one value
                if len(bias2) == 0:
                    bias2 = values
        except ValueError:
            continue
    
    # Parse metadata
    metadata = {}
    if metadata_start:
        for i in range(metadata_start, len(lines)):
            line = lines[i].strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0]
                try:
                    value = float(parts[1]) if '.' in parts[1] or 'e' in parts[1].lower() else int(parts[1])
                    metadata[key] = value
                except ValueError:
                    metadata[key] = ' '.join(parts[1:])
    
    return weights1, bias1, weights2, bias2, metadata

def analyze_weights(weights, name, expected_shape, expected_range=None):
    """Analyze a weight matrix for anomalies."""
    print(f"\n{'='*60}")
    print(f"Analysis: {name}")
    print(f"{'='*60}")
    
    if not weights:
        print(f"❌ ERROR: {name} is empty!")
        return False
    
    # Convert to numpy array
    try:
        arr = np.array(weights)
        actual_shape = arr.shape
        print(f"Shape: Expected {expected_shape}, Got {actual_shape}")
        
        if actual_shape != expected_shape:
            print(f"⚠️  WARNING: Shape mismatch!")
            return False
        
        # Basic statistics
        print(f"\nStatistics:")
        print(f"  Min:    {np.min(arr):.6f}")
        print(f"  Max:    {np.max(arr):.6f}")
        print(f"  Mean:   {np.mean(arr):.6f}")
        print(f"  Std:    {np.std(arr):.6f}")
        print(f"  Median: {np.median(arr):.6f}")
        
        # Check for NaN and Inf
        nan_count = np.isnan(arr).sum()
        inf_count = np.isinf(arr).sum()
        if nan_count > 0:
            print(f"❌ ERROR: Found {nan_count} NaN values!")
            return False
        if inf_count > 0:
            print(f"❌ ERROR: Found {inf_count} Inf values!")
            return False
        
        # Check range
        if expected_range:
            min_val, max_val = expected_range
            out_of_range = np.sum((arr < min_val) | (arr > max_val))
            if out_of_range > 0:
                print(f"⚠️  WARNING: {out_of_range} values out of expected range [{min_val}, {max_val}]")
                print(f"  Actual range: [{np.min(arr):.6f}, {np.max(arr):.6f}]")
            else:
                print(f"✓ All values within expected range [{min_val}, {max_val}]")
        
        # Check for saturation (many identical values)
        unique_values, counts = np.unique(arr, return_counts=True)
        most_common = sorted(zip(counts, unique_values), reverse=True)[:5]
        print(f"\nMost common values:")
        for count, value in most_common:
            percentage = (count / arr.size) * 100
            print(f"  {value:.6f}: {count} times ({percentage:.2f}%)")
            if percentage > 50:
                print(f"  ⚠️  WARNING: High saturation! {percentage:.2f}% of values are identical")
        
        # Check variance
        variance = np.var(arr)
        print(f"\nVariance: {variance:.6f}")
        if variance < 0.01:
            print(f"⚠️  WARNING: Very low variance - weights may be too uniform")
        elif variance > 10:
            print(f"⚠️  WARNING: Very high variance - weights may be unstable")
        
        # Check for zero or near-zero values
        near_zero = np.sum(np.abs(arr) < 1e-10)
        if near_zero > 0:
            print(f"⚠️  WARNING: {near_zero} values are near-zero (< 1e-10)")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR analyzing {name}: {e}")
        return False

def analyze_metadata(metadata):
    """Analyze training metadata for consistency."""
    print(f"\n{'='*60}")
    print(f"Training Metadata Analysis")
    print(f"{'='*60}")
    
    if not metadata:
        print("❌ ERROR: No metadata found!")
        return False
    
    print("\nMetadata values:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    
    # Check consistency
    issues = []
    
    # Check epsilon
    if 'EPSILON' in metadata and 'EPSILON_MIN' in metadata:
        epsilon = metadata['EPSILON']
        epsilon_min = metadata['EPSILON_MIN']
        if epsilon < epsilon_min:
            issues.append(f"⚠️  EPSILON ({epsilon}) < EPSILON_MIN ({epsilon_min})")
        else:
            print(f"✓ Epsilon is valid: {epsilon} >= {epsilon_min}")
    
    # Check training progress
    if 'TRAINING_EPISODES' in metadata and 'TOTAL_GAMES' in metadata:
        episodes = metadata['TRAINING_EPISODES']
        games = metadata['TOTAL_GAMES']
        if episodes > 0 and games > 0:
            episodes_per_game = episodes / games
            print(f"✓ Episodes per game: {episodes_per_game:.2f}")
            if episodes_per_game < 1:
                issues.append(f"⚠️  Very few episodes per game ({episodes_per_game:.2f})")
    
    # Check scores
    if 'BEST_SCORE' in metadata and 'AVERAGE_SCORE' in metadata:
        best = metadata['BEST_SCORE']
        avg = metadata['AVERAGE_SCORE']
        if avg > best:
            issues.append(f"⚠️  AVERAGE_SCORE ({avg}) > BEST_SCORE ({best})")
        else:
            print(f"✓ Score consistency: Best={best}, Avg={avg:.2f}")
    
    if issues:
        print("\n⚠️  Issues found:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✓ Metadata appears consistent")
    
    return len(issues) == 0

def main():
    filename = "tetris_model.txt"
    
    print("="*60)
    print("Neural Network Model Analysis")
    print("="*60)
    print(f"\nAnalyzing: {filename}\n")
    
    try:
        weights1, bias1, weights2, bias2, metadata = parse_model_file(filename)
        
        # Analyze each component
        all_ok = True
        
        all_ok &= analyze_weights(weights1, "Input-to-Hidden Weights (weights1)", 
                                 (29, 64), expected_range=(-2.0, 2.0))
        
        all_ok &= analyze_weights([bias1], "Hidden Layer Biases (bias1)", 
                                 (1, 64), expected_range=(-2.0, 2.0))
        
        all_ok &= analyze_weights(weights2, "Hidden-to-Output Weights (weights2)", 
                                 (64, 1), expected_range=(-20.0, 20.0))
        
        all_ok &= analyze_weights([bias2], "Output Layer Bias (bias2)", 
                                 (1, 1), expected_range=(-20.0, 20.0))
        
        all_ok &= analyze_metadata(metadata)
        
        # Overall assessment
        print(f"\n{'='*60}")
        print("Overall Assessment")
        print(f"{'='*60}")
        
        if all_ok:
            print("✓ Model appears to be consistent and valid")
        else:
            print("⚠️  Model has some issues that should be addressed")
            print("\nRecommendations:")
            print("  1. Check for weight saturation (many identical values)")
            print("  2. Verify weights are within expected ranges")
            print("  3. Check metadata consistency")
            print("  4. Consider retraining if issues are severe")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())

