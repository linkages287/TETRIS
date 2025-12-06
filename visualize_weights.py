#!/usr/bin/env python3
"""
Neural Network Weight Visualizer for Tetris AI
Reads tetris_model.txt and displays weight matrices graphically
"""

import numpy as np
import matplotlib
# Set backend before importing pyplot to avoid non-responsive issues
# Try TkAgg first (most stable), then fallback to Agg (non-interactive)
try:
    matplotlib.use('TkAgg')
except:
    try:
        matplotlib.use('Agg')
        print("Warning: Using non-interactive backend (Agg). Dynamic updates may not work.")
    except:
        pass  # Use default backend

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import os
import time
import sys

# Network architecture constants (matching rl_agent.h)
INPUT_SIZE = 27  # ZERO-BASED REDESIGN: 10 heights + 3 board_quality + 7 current + 7 next + 2 game_state
HIDDEN_SIZE = 64
OUTPUT_SIZE = 1

class WeightVisualizer:
    def __init__(self, model_file="tetris_model.txt"):
        self.model_file = model_file
        self.weights1 = None  # Input to Hidden (27 x 64)
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
        
        # For dynamic updates - store figure and axes references
        self.fig = None
        self.axes = None
        self.images = None  # Store image objects for heatmaps
        self.histograms = None  # Store histogram data for updates
        self.text_objects = []  # Store text objects to clear them properly
        self.hover_annotations = {}  # Store hover annotation objects
        self.colorbars = {}  # Store colorbar objects
        
    def load_model(self):
        """Load weights from tetris_model.txt"""
        if not os.path.exists(self.model_file):
            print(f"Error: {self.model_file} not found!")
            return False
        
        try:
            with open(self.model_file, 'r') as f:
                all_lines = f.readlines()
            
            # Filter out comment lines (starting with #) and empty lines
            lines = []
            for line in all_lines:
                stripped = line.strip()
                if stripped and not stripped.startswith('#'):
                    lines.append(stripped)
            
            if len(lines) < INPUT_SIZE + 1:
                print(f"Error: Not enough data lines in file (expected at least {INPUT_SIZE + 1}, got {len(lines)})")
                return False
            
            # Parse weights1 (27 rows, 64 columns each)
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
    
    def visualize_all(self, save_file=None, update_existing=False):
        """Create comprehensive visualization of all weights
        
        Args:
            save_file: If provided, save to file instead of displaying
            update_existing: If True, update existing figure instead of creating new one
        """
        if self.weights1 is None:
            print("Error: No weights loaded. Call load_model() first.")
            return
        
        stats = self.get_statistics()
        
        # If updating existing figure, reuse it
        if update_existing and self.fig is not None and plt.fignum_exists(self.fig.number):
            fig = self.fig
            # Clear all old text objects
            for text_obj in self.text_objects:
                try:
                    text_obj.remove()
                except:
                    pass
            self.text_objects = []
            
            # Clear existing plots but keep figure structure
            for ax in self.axes.values():
                ax.clear()
            
            # Remove old colorbars
            for cb in self.colorbars.values():
                try:
                    cb.remove()
                except:
                    pass
            self.colorbars = {}
        else:
            # Create new figure
            fig = plt.figure(figsize=(16, 13))
            self.fig = fig
            self.axes = {}
            self.images = {}
            self.histograms = {}
            self.text_objects = []
            self.hover_annotations = {}
            self.colorbars = {}
            
            # Set up hover tooltip handler
            def on_hover(event):
                if event.inaxes is None:
                    return
                
                # Find which axis we're hovering over
                ax = event.inaxes
                if ax not in [self.axes.get('ax1'), self.axes.get('ax2')]:
                    return
                
                # Get image data
                if ax == self.axes.get('ax1') and 'im1' in self.images:
                    im = self.images['im1']
                    data = self.weights1
                    label = "Weights1"
                elif ax == self.axes.get('ax2') and 'im2' in self.images:
                    im = self.images['im2']
                    data = self.weights2
                    label = "Weights2"
                else:
                    return
                
                # Convert mouse position to data coordinates
                if hasattr(im, 'get_extent'):
                    extent = im.get_extent()
                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()
                    
                    # Get data indices
                    x = int(event.xdata) if event.xdata is not None else -1
                    y = int(event.ydata) if event.ydata is not None else -1
                    
                    if 0 <= y < data.shape[0] and 0 <= x < data.shape[1]:
                        value = data[y, x]
                        
                        # Create or update annotation
                        if ax not in self.hover_annotations:
                            self.hover_annotations[ax] = ax.annotate(
                                '', xy=(0, 0), xytext=(10, 10),
                                textcoords='offset points',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                                arrowprops=dict(arrowstyle='->')
                            )
                            self.hover_annotations[ax].set_visible(False)
                        
                        ann = self.hover_annotations[ax]
                        ann.xy = (event.xdata, event.ydata)
                        ann.set_text(f'{label}\nRow: {y}, Col: {x}\nValue: {value:.6f}')
                        ann.set_visible(True)
                        fig.canvas.draw_idle()
                else:
                    # Hide annotation if outside data bounds
                    if ax in self.hover_annotations:
                        self.hover_annotations[ax].set_visible(False)
                        fig.canvas.draw_idle()
            
            def on_leave(event):
                # Hide all annotations when mouse leaves axes
                for ann in self.hover_annotations.values():
                    ann.set_visible(False)
                if self.fig is not None:
                    self.fig.canvas.draw_idle()
            
            # Connect event handlers
            fig.canvas.mpl_connect('motion_notify_event', on_hover)
            fig.canvas.mpl_connect('axes_leave_event', on_leave)
        
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3, top=0.90, bottom=0.10)
        
        # Custom colormap: blue (negative) -> white (zero) -> red (positive)
        colors = ['#0000FF', '#FFFFFF', '#FF0000']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
        
        # 1. Weights1 Heatmap (Input -> Hidden)
        if 'ax1' not in self.axes:
            self.axes['ax1'] = fig.add_subplot(gs[0, :])
        ax1 = self.axes['ax1']
        
        vmin1 = -max(abs(self.weights1.min()), abs(self.weights1.max()))
        vmax1 = max(abs(self.weights1.min()), abs(self.weights1.max()))
        
        # Always recalculate color range dynamically based on current data
        vmin1 = -max(abs(self.weights1.min()), abs(self.weights1.max()))
        vmax1 = max(abs(self.weights1.min()), abs(self.weights1.max()))
        
        if 'im1' not in self.images or not update_existing:
            im1 = ax1.imshow(self.weights1, aspect='auto', cmap=cmap, vmin=vmin1, vmax=vmax1)
            self.images['im1'] = im1
            cb1 = plt.colorbar(im1, ax=ax1, label='Weight Value')
            self.colorbars['cb1'] = cb1
        else:
            # Update existing image data with new color range
            self.images['im1'].set_data(self.weights1)
            self.images['im1'].set_clim(vmin1, vmax1)  # Update color scale dynamically
            # Update colorbar
            if 'cb1' in self.colorbars:
                self.colorbars['cb1'].update_normal(self.images['im1'])
        
        title1 = ax1.set_title(f'Weights1: Input → Hidden Layer ({INPUT_SIZE}×{HIDDEN_SIZE})\n'
                     f'Mean: {stats["weights1"]["mean"]:.4f}, Std: {stats["weights1"]["std"]:.4f}, '
                     f'Range: [{stats["weights1"]["min"]:.3f}, {stats["weights1"]["max"]:.3f}]',
                     fontsize=12, fontweight='bold')
        self.text_objects.append(title1)
        ax1.set_xlabel('Hidden Neurons (64)', fontsize=10)
        ax1.set_ylabel('Input Features (27)', fontsize=10)
        
        # Add y-axis labels for input features (grouped by type)
        feature_labels = []
        # 0-9: Column heights
        for i in range(10):
            feature_labels.append(f'H{i}')
        # 10-13: Height statistics
        feature_labels.extend(['HMax', 'HMin', 'HMean', 'HStd'])
        # 14-23: Holes per column
        for i in range(10):
            feature_labels.append(f'HL{i}')
        # 24-32: Height differences
        for i in range(9):
            feature_labels.append(f'HD{i}')
        # 33-39: Current piece
        feature_labels.extend(['CP0', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6'])
        # 40-46: Next piece
        feature_labels.extend(['NP0', 'NP1', 'NP2', 'NP3', 'NP4', 'NP5', 'NP6'])
        # 47: Lines cleared
        feature_labels.append('Lines')
        # 48: Level
        feature_labels.append('Level')
        
        # Set y-axis ticks and labels (show every 5th label to avoid crowding)
        tick_positions = list(range(0, INPUT_SIZE, max(1, INPUT_SIZE // 20)))  # ~20 labels max
        if tick_positions[-1] != INPUT_SIZE - 1:
            tick_positions.append(INPUT_SIZE - 1)  # Always show last label
        tick_labels = [feature_labels[i] if i < len(feature_labels) else f'F{i}' for i in tick_positions]
        ax1.set_yticks(tick_positions)
        ax1.set_yticklabels(tick_labels, fontsize=7)
        
        # 2. Weights2 Heatmap (Hidden -> Output)
        if 'ax2' not in self.axes:
            self.axes['ax2'] = fig.add_subplot(gs[1, :])
        ax2 = self.axes['ax2']
        
        # Always recalculate color range dynamically based on current data
        vmin2 = -max(abs(self.weights2.min()), abs(self.weights2.max()))
        vmax2 = max(abs(self.weights2.min()), abs(self.weights2.max()))
        
        if 'im2' not in self.images or not update_existing:
            im2 = ax2.imshow(self.weights2, aspect='auto', cmap=cmap, vmin=vmin2, vmax=vmax2)
            self.images['im2'] = im2
            cb2 = plt.colorbar(im2, ax=ax2, label='Weight Value')
            self.colorbars['cb2'] = cb2
        else:
            # Update existing image data with new color range
            self.images['im2'].set_data(self.weights2)
            self.images['im2'].set_clim(vmin2, vmax2)  # Update color scale dynamically
            # Update colorbar
            if 'cb2' in self.colorbars:
                self.colorbars['cb2'].update_normal(self.images['im2'])
        
        title2 = ax2.set_title(f'Weights2: Hidden → Output ({HIDDEN_SIZE}×{OUTPUT_SIZE})\n'
                     f'Mean: {stats["weights2"]["mean"]:.4f} | Std: {stats["weights2"]["std"]:.4f} | '
                     f'Range: [{stats["weights2"]["min"]:.3f}, {stats["weights2"]["max"]:.3f}]',
                     fontsize=11, fontweight='bold', pad=10)
        self.text_objects.append(title2)
        ax2.set_xlabel('Output Neurons (1)', fontsize=10)
        ax2.set_ylabel('Hidden Neurons (64)', fontsize=10)
        
        # 3. Bias1 Distribution
        if 'ax3' not in self.axes:
            self.axes['ax3'] = fig.add_subplot(gs[2, 0])
        ax3 = self.axes['ax3']
        
        # Clear and redraw histogram
        ax3.clear()
        n, bins, patches = ax3.hist(self.bias1, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        self.histograms['bias1'] = (n, bins, patches)
        ax3.axvline(stats['bias1']['mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {stats["bias1"]["mean"]:.4f}')
        title3 = ax3.set_title(f'Bias1 Distribution\n'
                     f'Mean: {stats["bias1"]["mean"]:.4f} | Std: {stats["bias1"]["std"]:.4f}',
                     fontsize=10, fontweight='bold', pad=8)
        self.text_objects.append(title3)
        ax3.set_xlabel('Bias Value', fontsize=9)
        ax3.set_ylabel('Frequency', fontsize=9)
        leg3 = ax3.legend()
        if leg3:
            self.text_objects.append(leg3)
        ax3.grid(True, alpha=0.3)
        
        # 4. Bias2 Value
        if 'ax4' not in self.axes:
            self.axes['ax4'] = fig.add_subplot(gs[2, 1])
        ax4 = self.axes['ax4']
        
        # Clear and redraw bar
        ax4.clear()
        ax4.bar([0], [self.bias2[0]], color='coral', edgecolor='black', linewidth=2)
        title4 = ax4.set_title(f'Bias2 Value\n{self.bias2[0]:.6f}', 
                     fontsize=10, fontweight='bold', pad=8)
        self.text_objects.append(title4)
        ax4.set_ylabel('Bias Value', fontsize=9)
        ax4.set_xticks([])
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Weight Magnitude Distribution
        if 'ax5' not in self.axes:
            self.axes['ax5'] = fig.add_subplot(gs[2, 2])
        ax5 = self.axes['ax5']
        
        all_weights = np.concatenate([self.weights1.flatten(), self.weights2.flatten()])
        
        # Clear and redraw histogram
        ax5.clear()
        n, bins, patches = ax5.hist(np.abs(all_weights), bins=50, edgecolor='black', alpha=0.7, color='lightgreen')
        self.histograms['all_weights'] = (n, bins, patches)
        ax5.axvline(np.mean(np.abs(all_weights)), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(np.abs(all_weights)):.4f}')
        title5 = ax5.set_title('Weight Magnitude Distribution', fontsize=10, fontweight='bold', pad=8)
        self.text_objects.append(title5)
        ax5.set_xlabel('|Weight| Value', fontsize=9)
        ax5.set_ylabel('Frequency', fontsize=9)
        ax5.set_yscale('log')
        leg5 = ax5.legend()
        if leg5:
            self.text_objects.append(leg5)
        ax5.grid(True, alpha=0.3)
        
        # Overall title with timestamp - make it more prominent
        # Adjust figure to make room for titles
        fig.subplots_adjust(top=0.92, bottom=0.08)
        
        if self.last_load_time:
            load_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.last_load_time))
            elapsed = time.time() - self.last_load_time
            title = 'Neural Network Weight Visualization - Tetris AI'
            
            # Main title - clear old and add new
            if update_existing:
                # Remove old suptitle if exists
                if hasattr(fig, '_suptitle') and fig._suptitle is not None:
                    try:
                        fig._suptitle.remove()
                    except:
                        pass
            
            suptitle = fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
            self.text_objects.append(suptitle)
            
            # Subtitle line 1: Clear timestamp showing when data was loaded
            if elapsed < 60:
                time_ago = f"{elapsed:.1f}s ago"
            elif elapsed < 3600:
                time_ago = f"{elapsed/60:.1f}min ago"
            else:
                time_ago = f"{elapsed/3600:.1f}hr ago"
            
            subtitle1 = f'Data loaded: {load_time_str} ({time_ago}) | Load #{self.load_count}'
            text1 = fig.text(0.5, 0.95, subtitle1, ha='center', fontsize=11, 
                    style='normal', color='green', weight='bold')
            self.text_objects.append(text1)
            
            # Subtitle line 2: Metadata (if available)
            if self.metadata['epsilon'] is not None:
                metadata_str = (f"Epsilon: {self.metadata['epsilon']:.3f} | "
                              f"Episodes: {self.metadata['training_episodes']} | "
                              f"Games: {self.metadata['total_games']} | "
                              f"Best Score: {self.metadata['best_score']}")
                text2 = fig.text(0.5, 0.92, metadata_str, ha='center', fontsize=9, 
                        style='normal', color='darkblue')
                self.text_objects.append(text2)
        else:
            title = 'Neural Network Weight Visualization - Tetris AI'
            suptitle = fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
            self.text_objects.append(suptitle)
            text3 = fig.text(0.5, 0.95, 'No weights loaded yet', ha='center', fontsize=10, 
                    style='italic', color='red')
            self.text_objects.append(text3)
        
        if save_file:
            plt.savefig(save_file, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_file}")
            if not update_existing:
                plt.close()  # Close figure to free memory only if not updating
        else:
            # Interactive mode - show or update
            if update_existing:
                # Update existing figure dynamically
                try:
                    if fig.canvas is not None:
                        fig.canvas.draw_idle()  # Use draw_idle instead of draw for better responsiveness
                        fig.canvas.flush_events()
                    plt.pause(0.01)  # Small pause to allow GUI to update
                except Exception as e:
                    # If update fails, try to recreate
                    print(f"Warning: Could not update plot: {e}")
                    raise  # Re-raise to trigger recreation
            else:
                # Show new figure
                try:
                    plt.show(block=False)  # Non-blocking for dynamic updates
                    plt.pause(0.1)  # Small pause to ensure display
                except Exception as e:
                    print(f"Note: Could not display plot interactively: {e}")
                    print("Falling back to static display...")
                    try:
                        plt.show(block=True)  # Try blocking mode as fallback
                    except:
                        print("Use --save option to save visualization to file")
                        if not update_existing:
                            try:
                                plt.close()
                            except:
                                pass
    
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
        
        # ZERO-BASED REDESIGN: Minimal essential features
        feature_names = []
        # 0-9: Column heights
        for i in range(10):
            feature_names.append(f'H{i}')
        # 10-12: Board quality
        feature_names.extend(['MaxH', 'Holes', 'Bump'])
        # 13-19: Current piece
        feature_names.extend(['CP0', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6'])
        # 20-26: Next piece
        feature_names.extend(['NP0', 'NP1', 'NP2', 'NP3', 'NP4', 'NP5', 'NP6'])
        # 27-28: Game state (but we only have 27 features, so this is wrong)
        # Actually: 0-9 heights, 10-12 quality, 13-19 current, 20-26 next = 27 total
        # Wait, let me recount: 10 + 3 + 7 + 7 = 27, but we need lines and level
        # So: 10 heights + 3 quality + 7 current + 7 next = 27, but missing lines/level
        # Let me fix: 10 + 3 + 7 + 7 + 2 = 29... but INPUT_SIZE is 27
        # Actually the count is: 10 heights + 3 quality + 7 current + 7 next = 27
        # But we also have lines and level... let me check the code
        
        ax1.set_yticks(range(INPUT_SIZE))
        ax1.set_yticklabels(feature_names, fontsize=7)
        
        # Weight distribution per input feature
        weight_means = np.mean(np.abs(self.weights1), axis=1)
        ax2.barh(range(INPUT_SIZE), weight_means, color='steelblue', edgecolor='black')
        ax2.set_title('Average |Weight| per Input Feature', 
                     fontsize=13, fontweight='bold', pad=10)
        ax2.set_xlabel('Average |Weight|', fontsize=12)
        ax2.set_ylabel('Input Features', fontsize=12)
        ax2.set_yticks(range(INPUT_SIZE))
        ax2.set_yticklabels(feature_names, fontsize=7)
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
                plt.show(block=False)  # Non-blocking
                plt.pause(0.1)
            except Exception as e:
                print(f"Note: Could not display plot interactively: {e}")
                print("Use --save option to save visualization to file")
                plt.close()
    
    def monitor_changes(self, interval=2.0):
        """Monitor model file for changes and update visualization dynamically"""
        print(f"Monitoring {self.model_file} for changes (every {interval}s)")
        print("Press Ctrl+C to stop")
        print("-" * 60)
        
        # Try to enable interactive mode, but handle if it fails
        try:
            plt.ion()
            interactive_available = True
        except Exception as e:
            print(f"Warning: Interactive mode not available: {e}")
            print("Will use static updates instead")
            interactive_available = False
        
        previous_weights1 = None
        previous_weights2 = None
        previous_mtime = None
        check_count = 0
        first_load = True
        
        try:
            while True:
                check_count += 1
                current_time = time.time()
                time_str = time.strftime('%H:%M:%S', time.localtime(current_time))
                
                if os.path.exists(self.model_file):
                    # Check if file was modified
                    mtime = os.path.getmtime(self.model_file)
                    file_age = current_time - mtime
                    
                    # Load if file was modified or first time
                    # Use strict comparison to detect any change (0.1s tolerance for file system precision)
                    if previous_mtime is None or abs(mtime - previous_mtime) > 0.1:
                        load_start_time = time.time()
                        if self.load_model():
                            load_duration = time.time() - load_start_time
                            load_time_str = time.strftime('%H:%M:%S', time.localtime(self.last_load_time))
                            full_timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.last_load_time))
                            
                            # Show timestamp when data is loaded
                            print(f"[{load_time_str}] ✅ Data loaded at {full_timestamp} (Load #{self.load_count})")
                            
                            # Update visualization dynamically
                            try:
                                if first_load:
                                    # First time: create new visualization
                                    self.visualize_all(update_existing=False)
                                    first_load = False
                                    print(f"[{load_time_str}] 📊 Visualization window opened")
                                else:
                                    # Subsequent updates: refresh existing plots
                                    if interactive_available and self.fig is not None and plt.fignum_exists(self.fig.number):
                                        self.visualize_all(update_existing=True)
                                    else:
                                        # Window was closed or interactive mode failed - recreate
                                        print(f"[{load_time_str}] Recreating visualization window...")
                                        self.visualize_all(update_existing=False)
                            except Exception as viz_error:
                                print(f"[{load_time_str}] ⚠ Error updating visualization: {viz_error}")
                                # Try to recreate on next update
                                first_load = True
                                if self.fig is not None:
                                    try:
                                        plt.close(self.fig)
                                    except:
                                        pass
                                    self.fig = None
                            
                            previous_weights1 = self.weights1.copy()
                            previous_weights2 = self.weights2.copy() if self.weights2 is not None else None
                            previous_mtime = mtime
                    else:
                        # File not changed - show status every 20 checks
                        if check_count % 20 == 0:
                            print(f"[{time_str}] Waiting for changes... (File age: {file_age:.1f}s)")
                else:
                    if check_count % 10 == 0:
                        print(f"[{time_str}] ⚠ File not found: {self.model_file}")
                
                # Use shorter sleep with event processing to avoid blocking
                # This prevents matplotlib from becoming non-responsive
                sleep_chunks = max(1, int(interval * 10))
                chunk_time = interval / sleep_chunks
                for _ in range(sleep_chunks):
                    try:
                        # Process GUI events during sleep to keep matplotlib responsive
                        if interactive_available and plt.isinteractive():
                            plt.pause(0.01)
                    except:
                        pass
                    time.sleep(chunk_time)
                
        except KeyboardInterrupt:
            print(f"\n[{time.strftime('%H:%M:%S')}] Monitoring stopped after {check_count} checks")
            print(f"Total loads: {self.load_count}")
            try:
                plt.ioff()  # Turn off interactive mode
                if self.fig is not None:
                    plt.close(self.fig)
            except:
                pass
        except Exception as e:
            print(f"\nError during monitoring: {e}")
            import traceback
            traceback.print_exc()
            try:
                plt.ioff()
                if self.fig is not None:
                    plt.close(self.fig)
            except:
                pass


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
        # For monitoring, let monitor_changes handle everything (no double load)
        visualizer.monitor_changes(args.interval)
    else:
        # For single visualization, load and display once
        if visualizer.load_model():
            if args.detail:
                visualizer.visualize_weights1_detail(args.save)
            else:
                visualizer.visualize_all(args.save)


if __name__ == '__main__':
    main()

