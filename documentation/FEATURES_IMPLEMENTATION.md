# Implementazione Features - Guida Completa

**Data:** 2025-12-01  
**File Principale:** `rl_agent.cpp`  
**Funzione Principale:** `RLAgent::extractState()`

---

## 📊 STRUTTURA FEATURES (29 totali)

Il modello usa **29 features** come input alla rete neurale (`INPUT_SIZE = 29`):

```
INPUT_SIZE = 29 features:
├─ Column Heights: 10 features (indici 0-9)
├─ Holes: 1 feature (indice 10)
├─ Bumpiness: 1 feature (indice 11)
├─ Aggregate Height: 1 feature (indice 12)
├─ Current Piece Type: 7 features (indici 13-19)
├─ Next Piece Type: 7 features (indici 20-26)
├─ Lines Cleared: 1 feature (indice 27)
└─ Level: 1 feature (indice 28)
```

---

## 📍 DOVE SONO IMPLEMENTATE

### 1. **Funzione Principale: `extractState()`**

**File:** `rl_agent.cpp`  
**Linee:** 675-722  
**Prototipo:** `std::vector<double> RLAgent::extractState(const TetrisGame& game)`

Questa funzione estrae tutte le 29 features dallo stato corrente del gioco.

### 2. **Funzione Simulazione: `findBestMove()`**

**File:** `rl_agent.cpp`  
**Linee:** 773-850 (circa)  
**Contesto:** Simula lo stato futuro per calcolare Q-values

Questa funzione ricrea manualmente le features per lo stato simulato dopo aver piazzato un pezzo.

---

## 🔍 DETTAGLIO FEATURES

### **1. Column Heights (10 features, indici 0-9)**

**Implementazione:**
```cpp
// rl_agent.cpp:684-686
for (int x = 0; x < game.WIDTH; x++) {
    state[idx++] = game.getColumnHeight(x, game.board) / 20.0;  // Normalize
}
```

**Calcolo:**
- **Funzione helper:** `TetrisGame::getColumnHeight(int x, const std::vector<std::vector<int>>& board)`
- **File:** `tetris.cpp:401-408`
- **Normalizzazione:** `/20.0` (altezza massima = 20)
- **Range:** 0.0 - 1.0 (altezza 0-20)

**Logica:**
```cpp
// tetris.cpp:401-408
int TetrisGame::getColumnHeight(int x, const std::vector<std::vector<int>>& sim_board) const {
    for (int y = 0; y < HEIGHT; y++) {
        if (sim_board[y][x] != 0) {
            return HEIGHT - y;  // Altezza dalla base (0 = vuota, 20 = piena)
        }
    }
    return 0;  // Colonna vuota
}
```

**Simulazione (findBestMove):**
```cpp
// rl_agent.cpp:777-785
for (int x = 0; x < game.WIDTH; x++) {
    int height = 0;
    for (int y = 0; y < game.HEIGHT; y++) {
        if (sim_board[y][x] != 0) {
            height = game.HEIGHT - y;
            break;
        }
    }
    next_state[idx++] = height / 20.0;
}
```

---

### **2. Holes Count (1 feature, indice 10)**

**Implementazione:**
```cpp
// rl_agent.cpp:688-689
state[idx++] = game.countHoles(game.board) / 100.0;
```

**Calcolo:**
- **Funzione helper:** `TetrisGame::countHoles(const std::vector<std::vector<int>>& board)`
- **File:** `tetris.cpp:410-422`
- **Normalizzazione:** `/100.0`
- **Range:** 0.0 - ~0.2 (tipicamente 0-20 holes)

**Logica:**
```cpp
// tetris.cpp:410-422
int TetrisGame::countHoles(const std::vector<std::vector<int>>& sim_board) const {
    int holes = 0;
    for (int x = 0; x < WIDTH; x++) {
        bool block_found = false;
        for (int y = 0; y < HEIGHT; y++) {
            if (sim_board[y][x] != 0) {
                block_found = true;
            } else if (block_found) {
                holes++;  // Empty cell below a block = hole
            }
        }
    }
    return holes;
}
```

**Simulazione (findBestMove):**
```cpp
// rl_agent.cpp:788-800
int holes = 0;
for (int x = 0; x < game.WIDTH; x++) {
    bool block_found = false;
    for (int y = 0; y < game.HEIGHT; y++) {
        if (sim_board[y][x] != 0) {
            block_found = true;
        } else if (block_found) {
            holes++;
        }
    }
}
next_state[idx++] = holes / 100.0;
```

---

### **3. Bumpiness (1 feature, indice 11)** ⚠️ MODIFICATA RECENTEMENTE

**Implementazione:**
```cpp
// rl_agent.cpp:691-693
// Bumpiness - improved normalization for better learning (from /50.0 to /20.0)
// Larger normalized values (0.0-2.5) allow weights to have more impact
state[idx++] = game.calculateBumpiness(game.board) / 20.0;
```

**Calcolo:**
- **Funzione helper:** `TetrisGame::calculateBumpiness(const std::vector<std::vector<int>>& board)`
- **File:** `tetris.cpp:425-432`
- **Normalizzazione:** `/20.0` (modificata da `/50.0`)
- **Range:** 0.0 - ~2.5 (bumpiness tipicamente 0-50)

**Logica:**
```cpp
// tetris.cpp:425-432
int TetrisGame::calculateBumpiness(const std::vector<std::vector<int>>& sim_board) const {
    int bumpiness = 0;
    for (int x = 0; x < WIDTH - 1; x++) {
        int h1 = getColumnHeight(x, sim_board);
        int h2 = getColumnHeight(x + 1, sim_board);
        bumpiness += abs(h1 - h2);  // Somma differenze altezze adiacenti
    }
    return bumpiness;
}
```

**Simulazione (findBestMove):**
```cpp
// rl_agent.cpp:802-812
int bumpiness = 0;
for (int x = 0; x < game.WIDTH - 1; x++) {
    int h1 = 0, h2 = 0;
    for (int y = 0; y < game.HEIGHT; y++) {
        if (sim_board[y][x] != 0 && h1 == 0) h1 = game.HEIGHT - y;
        if (sim_board[y][x+1] != 0 && h2 == 0) h2 = game.HEIGHT - y;
    }
    bumpiness += abs(h1 - h2);
}
next_state[idx++] = bumpiness / 20.0;
```

**Feature Importance:**
- **Learning rate multiplier:** 2.0x (indice 11)
- **Ragione:** Feature sottosviluppata, necessita più apprendimento

---

### **4. Aggregate Height (1 feature, indice 12)** ⚠️ MODIFICATA RECENTEMENTE

**Implementazione:**
```cpp
// rl_agent.cpp:695-697
// Aggregate height - improved normalization for better learning (from /200.0 to /50.0)
// Larger normalized values (0.0-4.0) allow weights to have more impact
state[idx++] = game.getAggregateHeight(game.board) / 50.0;
```

**Calcolo:**
- **Funzione helper:** `TetrisGame::getAggregateHeight(const std::vector<std::vector<int>>& board)`
- **Normalizzazione:** `/50.0` (modificata da `/200.0`)
- **Range:** 0.0 - ~4.0 (aggregate height tipicamente 0-200)

**Logica:**
```cpp
// tetris.cpp:435-441
int TetrisGame::getAggregateHeight(const std::vector<std::vector<int>>& sim_board) const {
    int height = 0;
    for (int x = 0; x < WIDTH; x++) {
        height += getColumnHeight(x, sim_board);  // Somma tutte le altezze
    }
    return height;
}
```

**Simulazione (findBestMove):**
```cpp
// rl_agent.cpp:814-824
int aggregate_height = 0;
for (int x = 0; x < game.WIDTH; x++) {
    for (int y = 0; y < game.HEIGHT; y++) {
        if (sim_board[y][x] != 0) {
            aggregate_height += game.HEIGHT - y;
            break;
        }
    }
}
next_state[idx++] = aggregate_height / 50.0;
```

**Feature Importance:**
- **Learning rate multiplier:** 2.0x (indice 12)
- **Ragione:** Feature sottosviluppata, necessita più apprendimento

---

### **5. Current Piece Type (7 features, indici 13-19)**

**Implementazione:**
```cpp
// rl_agent.cpp:699-702
// Current piece type (7 features)
for (int i = 0; i < 7; i++) {
    state[idx++] = (game.current_piece->type == i) ? 1.0 : 0.0;
}
```

**Tipi di pezzi:**
- 0: I (line)
- 1: O (square)
- 2: T (T-shape)
- 3: S (S-shape)
- 4: Z (Z-shape)
- 5: J (J-shape)
- 6: L (L-shape)

**Range:** 0.0 o 1.0 (one-hot encoding)

**Simulazione (findBestMove):**
```cpp
// rl_agent.cpp:826-829
// Current piece type (7 features) - none since piece is placed
for (int i = 0; i < 7; i++) {
    next_state[idx++] = 0.0;  // Pezzo già piazzato
}
```

---

### **6. Next Piece Type (7 features, indici 20-26)**

**Implementazione:**
```cpp
// rl_agent.cpp:704-713
// Next piece type (7 features)
if (game.next_piece) {
    for (int i = 0; i < 7; i++) {
        state[idx++] = (game.next_piece->type == i) ? 1.0 : 0.0;
    }
} else {
    for (int i = 0; i < 7; i++) {
        state[idx++] = 0.0;
    }
}
```

**Range:** 0.0 o 1.0 (one-hot encoding)

**Simulazione (findBestMove):**
```cpp
// rl_agent.cpp:831-838
// Next piece type (7 features) - use actual next piece
if (game.next_piece) {
    for (int i = 0; i < 7; i++) {
        next_state[idx++] = (game.next_piece->type == i) ? 1.0 : 0.0;
    }
} else {
    for (int i = 0; i < 7; i++) {
        next_state[idx++] = 0.0;
    }
}
```

---

### **7. Lines Cleared (1 feature, indice 27)**

**Implementazione:**
```cpp
// rl_agent.cpp:715-716
// Lines cleared
state[idx++] = game.lines_cleared / 100.0;
```

**Normalizzazione:** `/100.0`  
**Range:** 0.0 - ~2.0 (tipicamente 0-200 lines)

**Simulazione (findBestMove):**
```cpp
// rl_agent.cpp:840-841
// Lines cleared (includes newly cleared lines)
next_state[idx++] = (game.lines_cleared + lines_cleared) / 100.0;
```

---

### **8. Level (1 feature, indice 28)**

**Implementazione:**
```cpp
// rl_agent.cpp:718-719
// Level
state[idx++] = game.level / 20.0;
```

**Normalizzazione:** `/20.0`  
**Range:** 0.0 - ~1.0 (tipicamente 0-20 levels)

**Simulazione (findBestMove):**
```cpp
// rl_agent.cpp:843-844
// Level (same as current)
next_state[idx++] = game.level / 20.0;
```

---

## 🔧 FUNZIONI HELPER (game_classes.h)

Le funzioni helper sono dichiarate in `game_classes.h` e implementate in `tetris.cpp`:

```cpp
// game_classes.h:67-70
int getColumnHeight(int x, const std::vector<std::vector<int>>& board) const;
int countHoles(const std::vector<std::vector<int>>& board) const;
int calculateBumpiness(const std::vector<std::vector<int>>& board) const;
int getAggregateHeight(const std::vector<std::vector<int>>& board) const;
```

**Implementazioni:**
- `getColumnHeight()`: `tetris.cpp:401-408`
- `countHoles()`: `tetris.cpp:410-422`
- `calculateBumpiness()`: `tetris.cpp:425-432`
- `getAggregateHeight()`: `tetris.cpp:435-441`

---

## 📊 TABELLA RIEPILOGATIVA

| Indice | Feature | Normalizzazione | Range Tipico | LR Multiplier |
|--------|---------|-----------------|--------------|---------------|
| 0-9 | Column Heights | `/20.0` | 0.0-1.0 | 1.0x |
| 10 | Holes | `/100.0` | 0.0-0.2 | 1.0x |
| 11 | Bumpiness | `/20.0` ⚠️ | 0.0-2.5 | **2.0x** |
| 12 | Aggregate Height | `/50.0` ⚠️ | 0.0-4.0 | **2.0x** |
| 13-19 | Current Piece | One-hot | 0.0/1.0 | 1.0x |
| 20-26 | Next Piece | One-hot | 0.0/1.0 | 1.0x |
| 27 | Lines Cleared | `/100.0` | 0.0-2.0 | 1.0x |
| 28 | Level | `/20.0` | 0.0-1.0 | 1.0x |

⚠️ = Modificata recentemente per migliorare l'apprendimento

---

## 🎯 UTILIZZO NELLE REWARDS

Le features sono anche usate per calcolare le rewards durante il training:

**File:** `tetris.cpp:1246-1253`

```cpp
int aggregate_height = game.getAggregateHeight(game.board);
reward -= aggregate_height * 0.2;  // Penalty per altezza

int holes = game.countHoles(game.board);
reward -= holes * 1.5;  // Penalty per holes

int bumpiness = game.calculateBumpiness(game.board);
reward -= bumpiness * 0.1;  // Penalty per superficie irregolare
```

---

## 🔍 FEATURE IMPORTANCE WEIGHTING

Le features **Bumpiness (11)** e **Aggregate Height (12)** hanno un learning rate moltiplicato per 2.0x:

**File:** `rl_agent.cpp:216-236`

```cpp
// Feature importance indices: bumpiness (11) and aggregate height (12)
const int BUMPINESS_IDX = 11;
const int AGGREGATE_HEIGHT_IDX = 12;
const double FEATURE_IMPORTANCE_MULTIPLIER = 2.0;

// Durante l'update dei pesi:
double feature_lr_multiplier = (j == BUMPINESS_IDX || j == AGGREGATE_HEIGHT_IDX) 
                              ? FEATURE_IMPORTANCE_MULTIPLIER : 1.0;
weights1[j][i] += learning_rate * feature_lr_multiplier * weight_gradient;
```

---

## 📝 NOTE IMPORTANTI

1. **Normalizzazione modificata recentemente:**
   - Bumpiness: `/50.0` → `/20.0` (valori 2.5x più grandi)
   - Aggregate Height: `/200.0` → `/50.0` (valori 4x più grandi)

2. **Feature importance weighting:**
   - Bumpiness e Aggregate Height hanno LR 2x per apprendere più velocemente

3. **Simulazione in findBestMove:**
   - Le features vengono ricalcolate manualmente per ogni mossa simulata
   - Non usa `extractState()` per performance (evita copie di oggetti)

4. **Consistenza:**
   - Le normalizzazioni devono essere identiche in `extractState()` e `findBestMove()`
   - Modifiche recenti applicate in entrambe le funzioni

---

## 🔗 RIFERIMENTI

- **Header:** `rl_agent.h:30` - `INPUT_SIZE = 29`
- **Estrazione stato:** `rl_agent.cpp:675-722` - `extractState()`
- **Simulazione stato:** `rl_agent.cpp:773-850` - `findBestMove()`
- **Helper functions:** `game_classes.h:67-70`
- **Implementazioni:** `tetris.cpp:410-432+`

