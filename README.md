# Terminal Tetris Game (C++)

A classic Tetris game implemented in C++ for the terminal using the `ncurses` library.

## Features

- All 7 classic Tetromino pieces (I, O, T, S, Z, J, L)
- Piece rotation with wall-kick mechanics
- Line clearing with scoring
- Progressive difficulty (speed increases with level)
- Next piece preview
- Score tracking
- Pause functionality
- Colorful terminal graphics

## Requirements

- C++ compiler with C++11 support (g++, clang++, etc.)
- `ncurses` development library

### Installing ncurses

**Ubuntu/Debian:**
```bash
sudo apt-get install libncurses5-dev libncursesw5-dev
```

**Fedora/RHEL:**
```bash
sudo dnf install ncurses-devel
```

**macOS:**
```bash
brew install ncurses
```

**Arch Linux:**
```bash
sudo pacman -S ncurses
```

## Building

### Using Makefile (Recommended)

```bash
make
```

This will create the `tetris` executable.

### Manual Compilation

```bash
g++ -Wall -Wextra -std=c++11 -O2 -o tetris tetris.cpp -lncurses
```

## How to Run

After building:

```bash
./tetris
```

Or use the Makefile:

```bash
make run
```

## Controls

- **← →** : Move piece left/right
- **↑** : Rotate piece clockwise
- **↓** : Soft drop (piece falls faster)
- **Space** : Hard drop (piece drops instantly)
- **P** : Pause/Unpause game
- **Q** : Quit game

## Game Rules

- Clear lines by filling them completely with blocks
- Score points for clearing lines:
  - 1 line: 100 × level
  - 2 lines: 300 × level
  - 3 lines: 500 × level
  - 4 lines (Tetris): 800 × level
- Game speed increases every 10 lines cleared
- Game ends when pieces reach the top of the board

## Scoring

- Line clears: Points based on number of lines (see above)
- Soft drop: +1 point per cell
- Hard drop: +2 points per cell

## Building Options

- `make` or `make all` - Build the game
- `make clean` - Remove build artifacts
- `make run` - Build and run the game
- `make install` - Make the executable executable (chmod +x)

## AI Training Mode

The game includes a Reinforcement Learning (RL) agent that can learn to play Tetris using Q-learning with a neural network.

### Training Controls

- **A** : Toggle AI on/off
- **T** : Toggle training mode on/off

### Training Features

- **Automatic Training**: The game starts in training mode by default
- **Continuous Learning**: Games automatically restart when they end, allowing continuous training
- **Experience Replay**: The AI learns from past experiences stored in a replay buffer
- **Model Persistence**: The trained model is automatically saved and can be loaded on next run

## Neural Network Data Storage

### How the Network Saves Data

The neural network saves its learned parameters (weights and biases) to a text file named `tetris_model.txt` in the same directory as the executable.

#### File Format

The model file (`tetris_model.txt`) stores the network parameters in a plain text format:

1. **Input-to-Hidden Layer Weights** (29 × 64 values)
   - 29 rows (one per input feature)
   - 64 columns (one per hidden neuron)
   - Each weight is stored as a floating-point number

2. **Hidden Layer Biases** (64 values)
   - One bias value per hidden neuron

3. **Hidden-to-Output Layer Weights** (64 × 1 values)
   - 64 rows (one per hidden neuron)
   - 1 column (for the single Q-value output)

4. **Output Layer Bias** (1 value)
   - Single bias value for the output neuron

#### Save Process

The model is saved automatically in the following situations:

1. **Periodic Saves**: Every 100 training episodes during active training
2. **On Exit**: When you quit the game (press 'Q') while in training mode
3. **Manual Save**: Can be triggered programmatically via `agent.saveModel()`

#### Load Process

When the game starts:

1. The RL agent attempts to load `tetris_model.txt` from the current directory
2. If the file exists and is valid, the network loads the saved weights
3. If loading succeeds, the exploration rate (epsilon) is set to minimum (0.01), using the trained model
4. If the file doesn't exist or is invalid, the network starts with random weights

#### File Structure Example

```
-0.0956122 -0.0939957 0.100369 ... (64 weights for input feature 1)
0.142612 -0.0101783 -0.032986 ... (64 weights for input feature 2)
... (27 more rows for remaining input features)
0.0874638 -0.135598 ... (64 hidden layer biases)
-0.0521539 0.0723576 ... (64 weights from hidden to output)
0.142612 (1 output bias)
```

#### Network Architecture

- **Input Layer**: 29 neurons (game state features)
  - 10 column heights
  - 1 holes count
  - 1 bumpiness metric
  - 1 aggregate height
  - 7 current piece type (one-hot encoded)
  - 7 next piece type (one-hot encoded)
  - 1 lines cleared
  - 1 level

- **Hidden Layer**: 64 neurons with ReLU activation

- **Output Layer**: 1 neuron (Q-value prediction)

#### Data Persistence Benefits

- **Resume Training**: Continue training from where you left off
- **Share Models**: Transfer trained models between systems
- **Backup**: Keep copies of your best-performing models
- **Experimentation**: Test different training strategies without losing progress

#### File Location

The model file is saved in the same directory where you run the `tetris` executable:

```
/home/linkages/cursor/TETRIS/tetris_model.txt
```

#### Model File Management

- **Backup**: Copy `tetris_model.txt` to save a checkpoint
- **Reset**: Delete `tetris_model.txt` to start training from scratch
- **Restore**: Replace `tetris_model.txt` with a backup to restore a previous model

#### Training Statistics

While training, the game displays:
- **Games**: Total number of games played
- **Episodes**: Number of training episodes (batches processed)
- **Best**: Highest score achieved
- **Avg**: Average score (exponential moving average)
- **Epsilon**: Exploration rate (decreases as training progresses)
- **Buffer**: Size of experience replay buffer

## Improving Network Performance

### Understanding Training Metrics

Monitor these key metrics to assess training progress:

1. **Average Score (Avg)**: The most important metric
   - Should gradually increase over time
   - Indicates the network is learning better strategies
   - A consistently rising average means the model is improving

2. **Best Score**: Peak performance achieved
   - Shows the network's potential
   - Should increase periodically as training continues

3. **Epsilon**: Exploration vs Exploitation balance
   - Starts at 1.0 (100% exploration)
   - Decays to 0.01 (1% exploration, 99% exploitation)
   - Lower epsilon means the network relies more on learned knowledge

4. **Buffer Size**: Experience replay capacity
   - Larger buffer = more diverse training data
   - Current size: 10,000 experiences

### Hyperparameter Tuning

To improve results, you can adjust these parameters in `rl_agent.cpp` (in the `RLAgent` constructor):

#### Learning Rate (`learning_rate`)
- **Current**: 0.001
- **Purpose**: Controls how much the network updates from each training step
- **Adjustments**:
  - **Too high** (>0.01): Network may overshoot optimal values, unstable training
  - **Too low** (<0.0001): Network learns very slowly
  - **Recommended range**: 0.0005 - 0.005
  - **For faster learning**: Try 0.002 - 0.003
  - **For more stable learning**: Try 0.0005 - 0.001

#### Discount Factor (`gamma`)
- **Current**: 0.95
- **Purpose**: How much the network values future rewards vs immediate rewards
- **Adjustments**:
  - **Higher** (0.98-0.99): Network thinks more long-term, better for strategic play
  - **Lower** (0.90-0.93): Network focuses on immediate rewards
  - **Recommended**: 0.95-0.99 for Tetris (long-term planning is important)

#### Epsilon Decay (`epsilon_decay`)
- **Current**: 0.9995 (optimized for better learning)
- **Purpose**: How quickly exploration decreases
- **Why 0.9995**: Slower decay prevents premature convergence to suboptimal strategies
  - With 0.995: Epsilon reaches minimum in ~900 episodes (too fast)
  - With 0.9995: Epsilon reaches minimum in ~9000 episodes (allows more exploration)
- **Adjustments**:
  - **Higher** (0.999-0.9999): Very slow decay, extensive exploration
  - **Lower** (0.995-0.998): Faster decay, quicker transition to exploitation
  - **Recommended**: 0.999-0.9995 for Tetris (complex strategy game needs exploration)

#### Epsilon Minimum (`epsilon_min`)
- **Current**: 0.05 (optimized for continued learning)
- **Purpose**: Minimum exploration rate (keeps some randomness)
- **Why 0.05**: Higher minimum ensures network continues exploring even after initial learning
  - Prevents getting stuck in local optima
  - Allows discovery of new strategies as game progresses
- **Adjustments**:
  - **Higher** (0.10-0.20): More exploration, better for complex games
  - **Lower** (0.01-0.02): More exploitation, relies heavily on learned knowledge
  - **Recommended**: 0.05-0.10 for games requiring strategic thinking

#### Epsilon Decay Delay
- **Current**: Decay starts after 100 training episodes
- **Purpose**: Ensures network has learned some basics before reducing exploration
- **Why it matters**: Prevents epsilon from decaying before the network has acquired useful knowledge

#### Batch Size (`BATCH_SIZE`)
- **Current**: 32
- **Purpose**: Number of experiences used per training step
- **Adjustments**:
  - **Larger** (64-128): More stable gradients, slower training
  - **Smaller** (16-24): Faster training, less stable
  - **Recommended**: 32-64

#### Replay Buffer Size (`BUFFER_SIZE`)
- **Current**: 10,000
- **Purpose**: Maximum number of stored experiences
- **Adjustments**:
  - **Larger** (20,000-50,000): More diverse training data, better generalization
  - **Smaller** (5,000): Faster updates, less memory
  - **Recommended**: 10,000-20,000

### Reward Shaping

The reward function significantly impacts learning. Current rewards in `tetris.cpp`:

```cpp
reward += score_diff * 0.1;        // Score-based reward
reward += lines_diff * 10.0;       // Line clearing reward
if (game.game_over) {
    reward -= 100.0;                // Game over penalty
}
```

**Improvement strategies**:

1. **Increase line clearing reward**: Change `10.0` to `20.0` or `30.0` to emphasize line clearing
2. **Add height penalty**: Penalize high stacks to encourage keeping the board low
3. **Add hole penalty**: Strongly penalize creating holes
4. **Add bumpiness penalty**: Encourage smooth board surfaces
5. **Reward survival time**: Give small rewards for each piece placed successfully

**Example improved reward function**:
```cpp
double reward = 0.0;
int score_diff = game.score - game.last_score;
int lines_diff = game.lines_cleared - game.last_lines;

reward += score_diff * 0.1;
reward += lines_diff * 20.0;  // Increased line reward

// Height penalty (encourage low stacks)
int max_height = 0;
for (int x = 0; x < game.WIDTH; x++) {
    int h = game.getColumnHeight(x, game.board);
    if (h > max_height) max_height = h;
}
reward -= max_height * 0.5;

// Holes penalty
int holes = game.countHoles(game.board);
reward -= holes * 2.0;

if (game.game_over) {
    reward -= 200.0;  // Stronger game over penalty
}
```

### Training Strategies

#### 1. **Long Training Sessions**
- Let the game train for extended periods (hours or days)
- The network improves gradually over thousands of games
- Monitor average score trends over time

#### 2. **Progressive Training**
- Start with easier settings, then increase difficulty
- Train with slower fall speed initially, then speed up
- This helps the network learn fundamentals before advanced play

#### 3. **Checkpoint Management**
- Save model backups at different stages
- Compare performance of different training runs
- Keep the best-performing model

#### 4. **Restart from Scratch**
- If training plateaus, try resetting the model
- Sometimes a fresh start with different hyperparameters works better
- Delete `tetris_model.txt` to start over

### Monitoring Progress

#### Signs of Good Training:
- ✅ Average score steadily increasing
- ✅ Best score periodically breaking records
- ✅ Epsilon decreasing (network using learned knowledge)
- ✅ Buffer filling up (diverse experiences collected)
- ✅ Network surviving longer in games

#### Signs of Poor Training:
- ❌ Average score not improving after 1000+ games
- ❌ Score stuck at very low values
- ❌ Network making obviously bad moves
- ❌ Training unstable (scores vary wildly)

#### When to Adjust Hyperparameters:
- **If learning is too slow**: Increase learning rate (0.002-0.003)
- **If training is unstable**: Decrease learning rate (0.0005)
- **If network stops exploring too early** (most common issue):
  - Increase epsilon_decay to 0.999-0.9995 for slower decay
  - Increase epsilon_min to 0.10-0.15 for more exploration
  - Network relies on poor initial knowledge before learning enough
- **If network is too random for too long**: Decrease epsilon_decay (0.998) or epsilon_min (0.02)
- **If network doesn't plan ahead**: Increase gamma (0.98)
- **If network converges to poor strategies**: 
  - Reset model (delete tetris_model.txt) and use slower epsilon decay
  - Increase epsilon_min to maintain exploration

### Network Architecture Adjustments

You can modify the network structure in `rl_agent.h`:

#### Hidden Layer Size (`HIDDEN_SIZE`)
- **Current**: 64 neurons
- **Adjustments**:
  - **Larger** (128-256): More capacity, slower training, better for complex strategies
  - **Smaller** (32-48): Faster training, less capacity
  - **Recommended**: 64-128 for Tetris

#### Input Features (`INPUT_SIZE`)
- **Current**: 29 features
- The state representation is already comprehensive
- Adding more features (e.g., well depth, T-spin potential) could help

### Best Practices

1. **Patience**: Training takes time. Let it run for hours or days
2. **Monitor Trends**: Watch average score over time, not individual games
3. **Save Regularly**: The model auto-saves every 100 episodes
4. **Experiment**: Try different hyperparameter combinations
5. **Document Changes**: Note what works and what doesn't
6. **Compare Models**: Test different trained models to find the best

### Expected Performance

- **Initial**: Random play, scores 0-100
- **After 100 games**: Basic play, scores 100-500
- **After 1,000 games**: Improved strategy, scores 500-2000
- **After 10,000 games**: Good play, scores 2000-10000
- **After 100,000 games**: Expert play, scores 10000+

Remember: Results vary based on hyperparameters, reward shaping, and training duration. Continuous training and experimentation are key to achieving the best results!

## Notes

- The game requires a terminal that supports colors and Unicode characters
- Minimum terminal size: 40x30 characters recommended
- For best experience, use a terminal with true color support
- Training mode uses faster game speed (50ms delay) for efficient learning
- The model file is human-readable text, making it easy to inspect or modify if needed

Enjoy playing Tetris!
