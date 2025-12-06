# findBestMove Algorithm Flowchart

## Visual Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    findBestMove(game, training)                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │ current_piece == null? │
                    └─────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
               YES                          NO
                │                           │
                ▼                           ▼
        ┌───────────────┐         ┌──────────────────────┐
        │ Return invalid│         │ Initialize:          │
        │ move (0,0,-∞) │         │ best_move = (-∞)     │
        └───────────────┘         │ piece = current_piece│
                                  └──────────────────────┘
                                            │
                                            ▼
                                  ┌─────────────────────┐
                                  │ EPSILON-GREEDY      │
                                  │ Decision            │
                                  │                     │
                                  │ explore = random()  │
                                  │   < epsilon?        │
                                  └─────────────────────┘
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    │                                               │
                   YES (Explore)                                    NO (Exploit)
                    │                                               │
                    ▼                                               ▼
        ┌──────────────────────┐         ┌──────────────────────────────────┐
        │ Random Exploration   │         │ OPTIMIZATION SETUP                │
        │                      │         │                                   │
        │ rot = rand() % 4     │         │ • Pre-cache: next_piece,          │
        │ x = rand() % WIDTH   │         │   lines_cleared, level            │
        │                      │         │ • Generate x_positions:           │
        │ Return {rot, x, 0.0} │         │   [center, center±1, center±2...]│
        └──────────────────────┘         └──────────────────────────────────┘
                                                    │
                                                    ▼
                                    ┌───────────────────────────────┐
                                    │ FOR EACH ROTATION (0 to 3)    │
                                    │   AND move_evaluations < 300  │
                                    └───────────────────────────────┘
                                                    │
                                                    ▼
                                    ┌───────────────────────────────┐
                                    │ Calculate Piece Bounds        │
                                    │ • Get blocks for rotation     │
                                    │ • Find min_x, max_x          │
                                    └───────────────────────────────┘
                                                    │
                                                    ▼
                                    ┌───────────────────────────────┐
                                    │ FOR EACH X POSITION           │
                                    │   (center → outward order)    │
                                    │   AND move_evaluations < 300  │
                                    └───────────────────────────────┘
                                                    │
                                                    ▼
                                    ┌───────────────────────────────┐
                                    │ BOUNDS CHECK                 │
                                    │ x + min_x < -2 OR            │
                                    │ x + max_x >= WIDTH+2?        │
                                    └───────────────────────────────┘
                                                    │
                        ┌───────────────────────────┴───────────────────────────┐
                        │                                                       │
                       YES                                                      NO
                        │                                                       │
                        ▼                                                       ▼
                ┌───────────────┐         ┌──────────────────────────────────┐
                │ SKIP (continue│         │ Set piece position:              │
                │   to next x)  │         │ piece.x = x, piece.y = 0         │
                └───────────────┘         │ move_evaluations++               │
                                          └──────────────────────────────────┘
                                                      │
                                                      ▼
                                          ┌──────────────────────────┐
                                          │ EARLY COLLISION CHECK    │
                                          │ checkCollision(piece)?    │
                                          └──────────────────────────┘
                                                      │
                        ┌───────────────────────────┴───────────────────────────┐
                        │                                                       │
                       YES                                                      NO
                        │                                                       │
                        ▼                                                       ▼
                ┌───────────────┐         ┌──────────────────────────────────┐
                │ SKIP (continue│         │ SIMULATE DROP                    │
                │   to next x)  │         │                                  │
                └───────────────┘         │ drop_y = 0                       │
                                          │ while (!collision && drop_y < max)│
                                          │   drop_y++                        │
                                          └──────────────────────────────────┘
                                                      │
                                                      ▼
                                          ┌──────────────────────────┐
                                          │ DROP VALIDATION          │
                                          │ drop_y >= max_drop?      │
                                          └──────────────────────────┘
                                                      │
                        ┌───────────────────────────┴───────────────────────────┐
                        │                                                       │
                       YES                                                      NO
                        │                                                       │
                        ▼                                                       ▼
                ┌───────────────┐         ┌──────────────────────────────────┐
                │ SKIP (continue│         │ SIMULATE BOARD STATE              │
                │   to next x)  │         │                                   │
                └───────────────┘         │ sim_board =                        │
                                          │   simulatePlacePiece(piece, drop_y)│
                                          │ lines_cleared =                     │
                                          │   simulateClearLines(sim_board)    │
                                          └──────────────────────────────────┘
                                                      │
                                                      ▼
                                          ┌──────────────────────────┐
                                          │ HEURISTIC FILTER          │
                                          │ holes = countHoles()     │
                                          │ holes > 15 AND           │
                                          │ lines < 50?              │
                                          └──────────────────────────┘
                                                      │
                        ┌───────────────────────────┴───────────────────────────┐
                        │                                                       │
                       YES                                                      NO
                        │                                                       │
                        ▼                                                       ▼
                ┌───────────────┐         ┌──────────────────────────────────┐
                │ SKIP (continue│         │ EXTRACT STATE FEATURES           │
                │   to next x)  │         │                                   │
                └───────────────┘         │ next_state = extractStateFromBoard│
                                          │   (29 features: heights, holes,   │
                                          │    bumpiness, aggregate, pieces,  │
                                          │    lines, level)                  │
                                          └──────────────────────────────────┘
                                                      │
                                                      ▼
                                          ┌──────────────────────────┐
                                          │ NEURAL NETWORK EVALUATION│
                                          │ q_value =                │
                                          │   q_network.forward(      │
                                          │     next_state)          │
                                          └──────────────────────────┘
                                                      │
                                                      ▼
                                          ┌──────────────────────────┐
                                          │ UPDATE BEST MOVE         │
                                          │ q_value > best_move.q?   │
                                          └──────────────────────────┘
                                                      │
                        ┌───────────────────────────┴───────────────────────────┐
                        │                                                       │
                       YES                                                      NO
                        │                                                       │
                        ▼                                                       ▼
        ┌──────────────────────────────────┐         ┌──────────────────────┐
        │ best_move = {rot, x, q_value}   │         │ Continue to next x  │
        │                                  │         │   position          │
        │ CHECK EARLY TERMINATION          │         └──────────────────────┘
        │ q_value > 100.0 AND             │                    │
        │ evaluations > 10?                │                    │
        └──────────────────────────────────┘                    │
                    │                                            │
        ┌───────────┴───────────┐                              │
        │                       │                              │
       YES                      NO                             │
        │                       │                              │
        ▼                       ▼                              │
┌───────────────┐   ┌──────────────────────┐                 │
│ BREAK (stop   │   │ Continue searching   │                 │
│  searching)   │   │   more positions     │                 │
└───────────────┘   └──────────────────────┘                 │
        │                       │                              │
        └───────────┬───────────┴──────────────────────────────┘
                    │
                    ▼
        ┌──────────────────────────┐
        │ CHECK ROTATION TERMINATION│
        │ Found very good move AND  │
        │ evaluations > 20?         │
        └──────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
       YES                      NO
        │                       │
        ▼                       ▼
┌───────────────┐   ┌──────────────────────┐
│ BREAK (stop   │   │ Try next rotation    │
│  rotations)    │   │                      │
└───────────────┘   └──────────────────────┘
        │                       │
        └───────────┬───────────┘
                    │
                    ▼
        ┌──────────────────────────┐
        │ RETURN best_move         │
        │ {rotation, x, q_value}   │
        └──────────────────────────┘
```

## Algorithm Steps Breakdown

### Phase 1: Initialization & Exploration Check
1. **Null Check**: If no current piece, return invalid move
2. **Epsilon-Greedy**: Random exploration with probability `epsilon` (training mode only)
   - If exploring: return random rotation and x position

### Phase 2: Optimization Setup (Exploitation Mode)
1. **Pre-compute constants**: Cache `next_piece`, `lines_cleared`, `level`
2. **Generate move order**: Create x-position list ordered center-outward
   - Example: [5, 4, 6, 3, 7, 2, 8, ...] for WIDTH=10

### Phase 3: Nested Search Loop
**Outer Loop**: For each rotation (0 to 3)
- Calculate piece bounds (min_x, max_x) for this rotation

**Inner Loop**: For each x position (center-outward)
1. **Bounds Check**: Skip if piece completely off-board
2. **Early Collision**: Skip if immediate collision
3. **Simulate Drop**: Calculate final landing position
4. **Drop Validation**: Skip if drop calculation failed
5. **Simulate Board**: Place piece and clear lines
6. **Heuristic Filter**: Skip if too many holes early game
7. **Extract State**: Create 29-feature state vector
8. **Evaluate**: Get Q-value from neural network
9. **Update Best**: Track move with highest Q-value
10. **Early Termination**: Stop if excellent move found

### Phase 4: Early Termination Checks
- **Position-level**: If Q-value > 100.0 and evaluated ≥10 moves, stop searching positions
- **Rotation-level**: If excellent move found and evaluated ≥20 moves, stop trying rotations

### Phase 5: Return Result
Return the move with highest Q-value: `{rotation, x, q_value}`

## Key Optimizations

1. **Move Ordering**: Center positions tried first (more likely to be good)
2. **Early Termination**: Stops when excellent move found (saves ~50-80% evaluations)
3. **Heuristic Filtering**: Skips obviously bad moves (too many holes)
4. **Bounds Optimization**: Calculates piece width to filter invalid positions
5. **Pre-computation**: Caches constants to avoid repeated access

## Performance Characteristics

- **Best Case**: ~10-20 evaluations (excellent move found early)
- **Average Case**: ~50-100 evaluations (good move found)
- **Worst Case**: ~300 evaluations (safety limit reached)
- **Time Complexity**: O(rotations × positions × state_extraction)
- **Space Complexity**: O(1) - only stores best move

## State Features Extracted (29 total)

1. **10** Column heights (normalized by /20.0)
2. **1** Holes count (normalized by /100.0)
3. **1** Bumpiness metric (normalized by /20.0)
4. **1** Aggregate height (normalized by /50.0)
5. **7** Current piece type (one-hot, all 0s after placement)
6. **7** Next piece type (one-hot encoded)
7. **1** Lines cleared (normalized by /100.0)
8. **1** Level (normalized by /20.0)

## Decision Points

The algorithm makes decisions at several key points:
- **Exploration vs Exploitation**: Based on epsilon probability
- **Move Validity**: Multiple filters (bounds, collision, drop, holes)
- **Early Termination**: Based on Q-value threshold and evaluation count
- **Best Move Selection**: Greedy selection of highest Q-value



