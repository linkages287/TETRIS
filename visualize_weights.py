#!/usr/bin/env python3
"""
Neural Network Weight Visualizer for Tetris AI
Reads tetris_model.txt and displays weight matrices graphically
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import os
import time
import sys

# Network architecture constants (matching rl_agent.h)
INPUT_SIZE = 29
HIDDEN_SIZE = 64
OUTPUT_SIZE = 1

class WeightVisualizer:
    def __init__(self, model_file="tetris_model.txt"):
        self.model_file = model_file
        self.weights1 = None  # Input to Hidden (29 x 64)
        self.bias1 = None     # Hidden bias (64)
        self.weights2 = None  # Hidden to Output (64 x 1)
        self.bias2 = None     # Output bias (1)
        self.last_load_time = None  # Timestamp of last successful load
        self.load_count = 0  # Number of times model was loaded
        
        # Training state metadata
        self.metadata = {
            'epsilon': None,
            'epsilon_min': None,
            'epsilon_decay': None,
            'learning_rate': None,
            'gamma': None,
            'training_episodes': None,
            'total_games': None,
            'best_score': None,
            'average_score': None,
            'previous_avg_score': None
        }
        
    def load_model(self):
        """Load weights from tetris_model.txt"""
        if not os.path.exists(self.model_file):
            print(f"Error: {self.model_file} not found!")
            return False
        
        try:
            with open(self.model_file, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            
            # Parse weights1 (29 rows, 64 columns each)
            self.weights1 = np.zeros((INPUT_SIZE, HIDDEN_SIZE))
            for i in range(INPUT_SIZE):
                values = list(map(float, lines[i].split()))
                if len(values) != HIDDEN_SIZE:
                    print(f"Error: Expected {HIDDEN_SIZE} values for weights1 row {i}, got {len(values)}")
                    return False
                self.weights1[i] = values
            
            # Parse bias1 (1 row, 64 values)
            bias1_line = lines[INPUT_SIZE]
            bias1_values = list(map(float, bias1_line.split()))
            if len(bias1_values) != HIDDEN_SIZE:
                print(f"Error: Expected {HIDDEN_SIZE} values for bias1, got {len(bias1_values)}")
                return False
            self.bias1 = np.array(bias1_values)
            
            # Parse weights2 (64 rows, 1 column each)
            self.weights2 = np.zeros((HIDDEN_SIZE, OUTPUT_SIZE))
            start_idx = INPUT_SIZE + 1
            for i in range(HIDDEN_SIZE):
                values = list(map(float, lines[start_idx + i].split()))
                if len(values) != OUTPUT_SIZE:
                    print(f"Error: Expected {OUTPUT_SIZE} values for weights2 row {i}, got {len(values)}")
                    return False
                self.weights2[i] = values
            
            # Parse bias2 (1 row, 1 value)
            bias2_line = lines[start_idx + HIDDEN_SIZE]
            bias2_values = list(map(float, bias2_line.split()))
            if len(bias2_values) != OUTPUT_SIZE:
                print(f"Error: Expected {OUTPUT_SIZE} values for bias2, got {len(bias2_values)}")
                return False
            self.bias2 = np.array(bias2_values)
            
            # Try to load metadata (if present) - starts after bias2 line
            metadata_start = start_idx + HIDDEN_SIZE + 1
            if metadata_start < len(lines):
                self._load_metadata(lines, metadata_start)
            
            # Update timestamp and counter
            self.last_load_time = time.time()
            self.load_count += 1
            # Don't print here - let monitor_changes handle the display
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def _load_metadata(self, lines, start_idx):
        """Load training state metadata from model file"""
        # Reset metadata
        for key in self.metadata:
            self.metadata[key] = None
        
        # Look for metadata section
        in_metadata = False
        for i in range(start_idx, len(lines)):
            line = lines[i].strip()
            if not line:
                continue
            
            if line == "# Training State Metadata":
                in_metadata = True
                continue
            
            if in_metadata:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0]
                    try:
                        value = float(parts[1]) if '.' in parts[1] or 'e' in parts[1].lower() else int(parts[1])
                        if key in self.metadata:
                            self.metadata[key] = value
                    except ValueError:
                        pass
    
    def get_statistics(self):
        """Calculate weight statistics"""
        stats = {
            'weights1': {
                'mean': np.mean(self.weights1),
                'std': np.std(self.weights1),
                'min': np.min(self.weights1),
                'max': np.max(self.weights1),
                'abs_mean': np.mean(np.abs(self.weights1))
            },
            'weights2': {
                'mean': np.mean(self.weights2),
                'std': np.std(self.weights2),
                'min': np.min(self.weights2),
                'max': np.max(self.weights2),
                'abs_mean': np.mean(np.abs(self.weights2))
            },
            'bias1': {
                'mean': np.mean(self.bias1),
                'std': np.std(self.bias1),
                'min': np.min(self.bias1),
                'max': np.max(self.bias1)
            },
            'bias2': {
                'value': self.bias2[0]
            }
        }
        return stats
    
    def visualize_all(self, save_file=None):
        """Create comprehensive visualization of all weights"""
        if self.weights1 is None:
            print("Error: No weights loaded. Call load_model() first.")
            return
        
        stats = self.get_statistics()
        
        # Create figure with subplots - adjust size and spacing
        fig = plt.figure(figsize=(16, 13))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3, top=0.90, bottom=0.10)
        
        # Custom colormap: blue (negative) -> white (zero) -> red (positive)
        colors = ['#0000FF', '#FFFFFF', '#FF0000']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
        
        # 1. Weights1 Heatmap (Input -> Hidden)
        ax1 = fig.add_subplot(gs[0, :])
        im1 = ax1.imshow(self.weights1, aspect='auto', cmap=cmap, 
                        vmin=-max(abs(self.weights1.min()), abs(self.weights1.max())),
                        vmax=max(abs(self.weights1.min()), abs(self.weights1.max())))
        ax1.set_title(f'Weights1: Input → Hidden Layer ({INPUT_SIZE}×{HIDDEN_SIZE})\n'
                     f'Mean: {stats["weights1"]["mean"]:.4f}, Std: {stats["weights1"]["std"]:.4f}, '
                     f'Range: [{stats["weights1"]["min"]:.3f}, {stats["weights1"]["max"]:.3f}]',
                     fontsize=12, fontweight='bold')
        ax1.set_xlabel('Hidden Neurons (64)', fontsize=10)
        ax1.set_ylabel('Input Features (29)', fontsize=10)
        plt.colorbar(im1, ax=ax1, label='Weight Value')
        
        # 2. Weights2 Heatmap (Hidden -> Output)
        ax2 = fig.add_subplot(gs[1, :])
        im2 = ax2.imshow(self.weights2, aspect='auto', cmap=cmap,
                        vmin=-max(abs(self.weights2.min()), abs(self.weights2.max())),
                        vmax=max(abs(self.weights2.min()), abs(self.weights2.max())))
        ax2.set_title(f'Weights2: Hidden → Output ({HIDDEN_SIZE}×{OUTPUT_SIZE})\n'
                     f'Mean: {stats["weights2"]["mean"]:.4f} | Std: {stats["weights2"]["std"]:.4f} | '
                     f'Range: [{stats["weights2"]["min"]:.3f}, {stats["weights2"]["max"]:.3f}]',
                     fontsize=11, fontweight='bold', pad=10)
        ax2.set_xlabel('Output Neurons (1)', fontsize=10)
        ax2.set_ylabel('Hidden Neurons (64)', fontsize=10)
        plt.colorbar(im2, ax=ax2, label='Weight Value')
        
        # 3. Bias1 Distribution
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.hist(self.bias1, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        ax3.axvline(stats['bias1']['mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {stats["bias1"]["mean"]:.4f}')
        ax3.set_title(f'Bias1 Distribution\n'
                     f'Mean: {stats["bias1"]["mean"]:.4f} | Std: {stats["bias1"]["std"]:.4f}',
                     fontsize=10, fontweight='bold', pad=8)
        ax3.set_xlabel('Bias Value', fontsize=9)
        ax3.set_ylabel('Frequency', fontsize=9)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Bias2 Value
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.bar([0], [self.bias2[0]], color='coral', edgecolor='black', linewidth=2)
        ax4.set_title(f'Bias2 Value\n{self.bias2[0]:.6f}', 
                     fontsize=10, fontweight='bold', pad=8)
        ax4.set_ylabel('Bias Value', fontsize=9)
        ax4.set_xticks([])
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Weight Magnitude Distribution
        ax5 = fig.add_subplot(gs[2, 2])
        all_weights = np.concatenate([self.weights1.flatten(), self.weights2.flatten()])
        ax5.hist(np.abs(all_weights), bins=50, edgecolor='black', alpha=0.7, color='lightgreen')
        ax5.axvline(np.mean(np.abs(all_weights)), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(np.abs(all_weights)):.4f}')
        ax5.set_title('Weight Magnitude Distribution', fontsize=10, fontweight='bold', pad=8)
        ax5.set_xlabel('|Weight| Value', fontsize=9)
        ax5.set_ylabel('Frequency', fontsize=9)
        ax5.set_yscale('log')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Overall title with timestamp - make it more prominent
        # Adjust figure to make room for titles
        fig.subplots_adjust(top=0.92, bottom=0.08)
        
        if self.last_load_time:
            load_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.last_load_time))
            elapsed = time.time() - self.last_load_time
            title = 'Neural Network Weight Visualization - Tetris AI'
            
            # Main title
            fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
            
            # Subtitle line 1: Timestamp (removed emoji for font compatibility)
            subtitle1 = f'Last Load: {load_time_str} ({elapsed:.1f}s ago) | Load #{self.load_count}'
            fig.text(0.5, 0.95, subtitle1, ha='center', fontsize=10, 
                    style='italic', color='blue', weight='bold')
            
            # Subtitle line 2: Metadata (if available)
            if self.metadata['epsilon'] is not None:
                metadata_str = (f"Epsilon: {self.metadata['epsilon']:.3f} | "
                              f"Episodes: {self.metadata['training_episodes']} | "
                              f"Games: {self.metadata['total_games']} | "
                              f"Best Score: {self.metadata['best_score']}")
                fig.text(0.5, 0.92, metadata_str, ha='center', fontsize=9, 
                        style='normal', color='darkblue')
        else:
            title = 'Neural Network Weight Visualization - Tetris AI'
            fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
            fig.text(0.5, 0.95, 'No weights loaded yet', ha='center', fontsize=10, 
                    style='italic', color='red')
        
        if save_file:
            plt.savefig(save_file, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_file}")
            plt.close()  # Close figure to free memory
        else:
            # Only show if not saving (interactive mode)
            try:
                plt.show()
            except Exception as e:
                print(f"Note: Could not display plot interactively: {e}")
                print("Use --save option to save visualization to file")
                plt.close()
    
    def visualize_weights1_detail(self, save_file=None):
        """Detailed visualization of Weights1"""
        if self.weights1 is None:
            print("Error: No weights loaded.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Heatmap
        colors = ['#0000FF', '#FFFFFF', '#FF0000']
        cmap = LinearSegmentedColormap.from_list('custom', colors, N=100)
        vmax = max(abs(self.weights1.min()), abs(self.weights1.max()))
        
        im = ax1.imshow(self.weights1, aspect='auto', cmap=cmap, vmin=-vmax, vmax=vmax)
        ax1.set_title(f'Weights1: Input → Hidden ({INPUT_SIZE}×{HIDDEN_SIZE})', 
                     fontsize=13, fontweight='bold', pad=10)
        ax1.set_xlabel('Hidden Neurons', fontsize=12)
        ax1.set_ylabel('Input Features', fontsize=12)
        plt.colorbar(im, ax=ax1, label='Weight Value')
        
        # Feature labels (if you want to add them)
        feature_names = [f'F{i}' for i in range(INPUT_SIZE)]
        ax1.set_yticks(range(INPUT_SIZE))
        ax1.set_yticklabels(feature_names, fontsize=8)
        
        # Weight distribution per input feature
        weight_means = np.mean(np.abs(self.weights1), axis=1)
        ax2.barh(range(INPUT_SIZE), weight_means, color='steelblue', edgecolor='black')
        ax2.set_title('Average |Weight| per Input Feature', 
                     fontsize=13, fontweight='bold', pad=10)
        ax2.set_xlabel('Average |Weight|', fontsize=12)
        ax2.set_ylabel('Input Features', fontsize=12)
        ax2.set_yticks(range(INPUT_SIZE))
        ax2.set_yticklabels(feature_names, fontsize=8)
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add timestamp to detailed view - make it more prominent
        # Adjust figure to make room for titles
        fig.subplots_adjust(top=0.90, bottom=0.10)
        
        if self.last_load_time:
            load_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.last_load_time))
            elapsed = time.time() - self.last_load_time
            title = 'Weights1 Detailed View'
            subtitle = f'Last Load: {load_time_str} ({elapsed:.1f}s ago) | Load #{self.load_count}'
            
            fig.suptitle(title, fontsize=14, fontweight='bold', y=0.97)
            fig.text(0.5, 0.94, subtitle, ha='center', fontsize=10, 
                    style='italic', color='blue', weight='bold')
            
            # Add metadata if available
            if self.metadata['epsilon'] is not None:
                metadata_str = (f"Epsilon: {self.metadata['epsilon']:.3f} | "
                              f"Episodes: {self.metadata['training_episodes']} | "
                              f"Games: {self.metadata['total_games']} | "
                              f"Best: {self.metadata['best_score']}")
                fig.text(0.5, 0.91, metadata_str, ha='center', fontsize=9, 
                        style='normal', color='darkblue')
        
        plt.tight_layout()
        
        if save_file:
            plt.savefig(save_file, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_file}")
            plt.close()  # Close figure to free memory
        else:
            # Only show if not saving (interactive mode)
            try:
                plt.show()
            except Exception as e:
                print(f"Note: Could not display plot interactively: {e}")
                print("Use --save option to save visualization to file")
                plt.close()
    
    def monitor_changes(self, interval=2.0):
        """Monitor model file for changes and update visualization"""
        print(f"Monitoring {self.model_file} for changes (every {interval}s)")
        print("Press Ctrl+C to stop")
        print("-" * 60)
        
        previous_weights1 = None
        previous_weights2 = None
        previous_mtime = None
        check_count = 0
        
        try:
            while True:
                check_count += 1
                current_time = time.time()
                time_str = time.strftime('%H:%M:%S', time.localtime(current_time))
                
                if os.path.exists(self.model_file):
                    # Check if file was modified
                    mtime = os.path.getmtime(self.model_file)
                    file_age = current_time - mtime
                    
                    # Show status every 10 checks or if file changed
                    if check_count % 10 == 0 or (previous_mtime is not None and mtime != previous_mtime):
                        status = "✓ UPDATED" if (previous_mtime is not None and mtime != previous_mtime) else "○ Checking..."
                        print(f"[{time_str}] {status} | File age: {file_age:.1f}s | Checks: {check_count}")
                    
                    # Load if file was modified or first time
                    if previous_mtime is None or mtime != previous_mtime:
                        load_start_time = time.time()
                        if self.load_model():
                            load_duration = time.time() - load_start_time
                            load_time_str = time.strftime('%H:%M:%S', time.localtime(self.last_load_time))
                            
                            # Show clear timestamp when load completes
                            print(f"\n{'='*60}")
                            print(f"[{load_time_str}] ✅ NEW WEIGHTS LOADED! (Load #{self.load_count})")
                            print(f"    Load duration: {load_duration*1000:.1f}ms")
                            print(f"    Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.last_load_time))}")
                            print(f"{'='*60}")
                            
                            # Check if weights changed
                            if previous_weights1 is not None:
                                diff1 = np.abs(self.weights1 - previous_weights1)
                                diff2 = np.abs(self.weights2 - previous_weights2) if previous_weights2 is not None else None
                                
                                max_diff1 = np.max(diff1)
                                mean_diff1 = np.mean(diff1)
                                
                                if max_diff1 > 0.0001:  # Significant change
                                    print(f"[{load_time_str}] ⚡ WEIGHTS UPDATED!")
                                    print(f"  Weights1 - Max change: {max_diff1:.6f}, Mean change: {mean_diff1:.6f}")
                                    if diff2 is not None:
                                        max_diff2 = np.max(diff2)
                                        mean_diff2 = np.mean(diff2)
                                        print(f"  Weights2 - Max change: {max_diff2:.6f}, Mean change: {mean_diff2:.6f}")
                                    print("-" * 60)
                                    
                                    # Update visualization with new timestamp
                                    self.visualize_all()
                                else:
                                    print(f"[{load_time_str}] File changed but weights unchanged (noise < 0.0001)")
                                    # Still update visualization to show new timestamp
                                    self.visualize_all()
                            else:
                                # First load
                                print(f"[{load_time_str}] 📊 Initial model loaded - displaying visualization")
                                self.visualize_all()
                            
                            previous_weights1 = self.weights1.copy()
                            previous_weights2 = self.weights2.copy() if self.weights2 is not None else None
                            previous_mtime = mtime
                else:
                    if check_count % 10 == 0:
                        print(f"[{time_str}] ⚠ File not found: {self.model_file}")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            elapsed = time.time() - (current_time - check_count * interval)
            print(f"\n[{time.strftime('%H:%M:%S')}] Monitoring stopped after {check_count} checks")
            print(f"Total loads: {self.load_count}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize Tetris AI neural network weights')
    parser.add_argument('--file', '-f', default='tetris_model.txt',
                       help='Path to model file (default: tetris_model.txt)')
    parser.add_argument('--save', '-s', type=str,
                       help='Save visualization to file (e.g., weights.png)')
    parser.add_argument('--monitor', '-m', action='store_true',
                       help='Monitor model file for changes')
    parser.add_argument('--interval', '-i', type=float, default=2.0,
                       help='Monitoring interval in seconds (default: 2.0)')
    parser.add_argument('--detail', '-d', action='store_true',
                       help='Show detailed Weights1 visualization')
    
    args = parser.parse_args()
    
    visualizer = WeightVisualizer(args.file)
    
    if args.monitor:
        if visualizer.load_model():
            visualizer.visualize_all()
            visualizer.monitor_changes(args.interval)
    else:
        if visualizer.load_model():
            if args.detail:
                visualizer.visualize_weights1_detail(args.save)
            else:
                visualizer.visualize_all(args.save)


if __name__ == '__main__':
    main()

