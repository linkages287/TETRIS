# Analisi Problemi Algoritmo di Apprendimento
## Report Diagnostico e Soluzioni Proposte

**Data:** 2025-12-01  
**Modello:** `tetris_model.txt`

---

## 🔴 PROBLEMI CRITICI IDENTIFICATI

### 1. **Prestazioni in Degrado**
- **Average Score**: 2,794 → **2,233** (-20.1% ⚠️)
- **Best Score**: 33,062 (stagnante, non migliora)
- **Ratio Average/Best**: 8.5% → **6.8%** (peggioramento)

**Implicazione**: Il modello sta **peggiorando** invece di migliorare. Questo indica:
- Instabilità nell'apprendimento
- Possibile overfitting o catastrophic forgetting
- Reward shaping non ottimale

### 2. **Bias2 Completamente Saturato**
- **Valore**: 19.95 (100% saturazione)
- **Range**: [-20.0, 20.0] (vicino al limite superiore)
- **Varianza**: 0.0 (nessuna variazione)

**Implicazione**: 
- Bias2 non può più apprendere (bloccato al limite)
- Limita la capacità del modello di adattarsi
- Riduce la flessibilità della rete neurale

### 3. **Alta Varianza in Weights2**
- **Varianza**: 17.45 (molto alta)
- **Range**: [-3.72, 4.30]
- **Std**: 2.34

**Implicazione**:
- Instabilità nell'apprendimento
- Pesi che oscillano troppo
- Possibile learning rate troppo alto

### 4. **Epsilon Management**
- **Epsilon corrente**: 0.1 (non più al minimo)
- **Epsilon min**: 0.1
- **Stato**: Il modello è tornato a esplorazione moderata

**Implicazione**:
- Il sistema ha rilevato il peggioramento e ha aumentato epsilon
- Ma questo potrebbe non essere sufficiente
- Potrebbe servire più esplorazione strutturata

---

## 🔍 ANALISI CAUSE ROOT

### A. **Reward Shaping Subottimale**

**Problema attuale:**
```cpp
reward += lines_diff * 5.0;        // Troppo basso per line clearing
reward += score_diff * 0.1;        // Troppo basso per score
reward += 1.0;                     // Survival bonus troppo piccolo
reward -= 50.0;                    // Game over penalty troppo bassa
reward -= aggregate_height * 0.05; // Penalty troppo debole
reward -= holes * 0.3;             // Penalty troppo debole
```

**Problemi:**
1. Le ricompense sono troppo piccole rispetto alle penalità
2. Il modello non è sufficientemente incentivato a fare line clears
3. Le penalità per stati cattivi sono troppo deboli
4. Il survival bonus è troppo piccolo per guidare l'apprendimento

### B. **Q-Value Clipping Troppo Restrittivo**

**Problema attuale:**
```cpp
next_q = std::max(-100.0, std::min(100.0, next_q));
target = std::max(-100.0, std::min(100.0, target));
```

**Problemi:**
1. Range [-100, 100] potrebbe essere troppo stretto
2. Limita la capacità del modello di apprendere valori Q estremi
3. Può causare saturazione prematura

### C. **Learning Rate Instabile**

**Problema attuale:**
- Learning rate: 0.003 (relativamente alto)
- Adaptive learning rate con logica complessa
- Può causare oscillazioni

**Problemi:**
1. Learning rate troppo alto causa instabilità (varianza alta)
2. Adaptive logic potrebbe non essere ottimale
3. Non c'è learning rate decay strutturato

### D. **Bias2 Update Logic**

**Problema attuale:**
- Bias2 ha learning rate aumentato (5x)
- Gradient clipping a [-5.0, 5.0]
- Weight decay aumentato (5x)
- Ma ancora saturato al limite

**Problemi:**
1. Le modifiche precedenti non hanno risolto la saturazione
2. Il clipping range [-20, 20] potrebbe essere troppo stretto
3. Il learning rate aumentato potrebbe causare oscillazioni

### E. **Experience Replay Buffer**

**Problema attuale:**
- Buffer size: 10,000
- Batch size: 32
- Sampling: 80% recent, 20% old

**Problemi:**
1. Bias verso esperienze recenti potrebbe causare catastrophic forgetting
2. Rimozione di game-over experiences potrebbe rimuovere informazioni importanti
3. Buffer potrebbe essere troppo piccolo per stabilità

---

## 💡 SOLUZIONI PROPOSTE

### 1. **Migliorare Reward Shaping** ⭐ PRIORITÀ ALTA

**Obiettivo**: Guidare meglio l'apprendimento con ricompense più bilanciate

**Modifiche proposte:**
```cpp
// Primary rewards (aumentati significativamente)
reward += lines_diff * 20.0;       // Aumentato da 5.0 (4x)
reward += score_diff * 0.5;        // Aumentato da 0.1 (5x)

// Survival bonus (aumentato)
if (!game.game_over) {
    reward += 2.0;                 // Aumentato da 1.0 (2x)
}

// Game over penalty (aumentata)
if (game.game_over) {
    reward -= 200.0;               // Aumentato da 50.0 (4x)
}

// State quality penalties (aumentate)
reward -= aggregate_height * 0.2;  // Aumentato da 0.05 (4x)
reward -= holes * 1.5;              // Aumentato da 0.3 (5x)
reward -= bumpiness * 0.1;          // Aumentato da 0.02 (5x)

// Bonus per board bassa (aumentato)
if (max_height < 10) {
    reward += 2.0;                 // Aumentato da 0.5 (4x)
}
```

**Razionalizzazione**:
- Line clearing è l'obiettivo principale → reward molto alto
- Score incrementale → reward moderato ma significativo
- Survival → reward costante per guidare l'apprendimento
- Penalties → sufficientemente alte da evitare stati cattivi

### 2. **Risolvere Bias2 Saturation** ⭐ PRIORITÀ ALTA

**Obiettivo**: Permettere a bias2 di apprendere liberamente

**Modifiche proposte:**
```cpp
// Aumentare clipping range per bias2
const double BIAS2_MIN = -50.0;  // Aumentato da -20.0
const double BIAS2_MAX = 50.0;   // Aumentato da 20.0

// Ridurre learning rate per bias2 (più stabile)
double bias2_learning_rate = learning_rate * 2.0;  // Ridotto da 5.0

// Ridurre gradient clipping per bias2
if (std::abs(bias2_gradient) > 3.0) {  // Ridotto da 5.0
    bias2_gradient = (bias2_gradient > 0) ? 3.0 : -3.0;
}

// Aggiungere momentum per bias2 (stabilità)
static double bias2_momentum = 0.0;
bias2_gradient = 0.9 * bias2_momentum + 0.1 * bias2_gradient;
bias2_momentum = bias2_gradient;
```

**Razionalizzazione**:
- Range più ampio permette più spazio per apprendere
- Learning rate più basso riduce oscillazioni
- Momentum aggiunge stabilità
- Gradient clipping più conservativo previene salti estremi

### 3. **Stabilizzare Learning Rate** ⭐ PRIORITÀ MEDIA

**Obiettivo**: Ridurre instabilità e varianza

**Modifiche proposte:**
```cpp
// Ridurre learning rate base
learning_rate = 0.0015;  // Ridotto da 0.003 (50%)

// Implementare learning rate decay strutturato
double lr_decay = 1.0 / (1.0 + training_episodes / 10000.0);
double current_lr = learning_rate * lr_decay;

// Usare learning rate più conservativo per weights2
double weights2_lr = current_lr * 0.5;  // Metà del learning rate normale
```

**Razionalizzazione**:
- Learning rate più basso riduce varianza
- Decay strutturato permette apprendimento più stabile
- Learning rate ridotto per weights2 previene oscillazioni

### 4. **Migliorare Q-Value Clipping** ⭐ PRIORITÀ MEDIA

**Obiettivo**: Permettere valori Q più estremi senza instabilità

**Modifiche proposte:**
```cpp
// Aumentare range di clipping
next_q = std::max(-500.0, std::min(500.0, next_q));  // Aumentato da 100.0
target = std::max(-500.0, std::min(500.0, target));  // Aumentato da 100.0

// Oppure: clipping dinamico basato su reward massimo
double max_reward = 200.0;  // Basato su reward massimo possibile
double clip_range = max_reward * 10.0;  // 10x il reward massimo
next_q = std::max(-clip_range, std::min(clip_range, next_q));
```

**Razionalizzazione**:
- Range più ampio permette valori Q più realistici
- Previene saturazione prematura
- Mantiene stabilità con clipping appropriato

### 5. **Migliorare Experience Replay** ⭐ PRIORITÀ BASSA

**Obiettivo**: Migliorare stabilità dell'apprendimento

**Modifiche proposte:**
```cpp
// Aumentare buffer size
static const int BUFFER_SIZE = 20000;  // Aumentato da 10000

// Migliorare sampling strategy
// 60% recent, 40% old (più bilanciato)
int recent_size = std::min((int)replay_buffer.size() / 2, ...);
int old_size = replay_buffer.size() - recent_size;

// Prioritized experience replay (opzionale)
// Sample esperienze con errore più alto più frequentemente
```

**Razionalizzazione**:
- Buffer più grande = più stabilità
- Sampling più bilanciato previene catastrophic forgetting
- Prioritized replay accelera apprendimento

### 6. **Aggiungere Gradient Normalization** ⭐ PRIORITÀ MEDIA

**Obiettivo**: Prevenire gradient explosion e instabilità

**Modifiche proposte:**
```cpp
// Normalizzare gradienti prima di applicarli
double gradient_norm = 0.0;
for (auto& row : weights2) {
    gradient_norm += row[0] * row[0];
}
gradient_norm = std::sqrt(gradient_norm);

if (gradient_norm > 5.0) {  // Threshold
    double scale = 5.0 / gradient_norm;
    // Applicare scale a tutti i gradienti
}
```

**Razionalizzazione**:
- Previene gradient explosion
- Mantiene stabilità durante l'apprendimento
- Standard practice in deep learning

---

## 📊 PRIORITÀ DI IMPLEMENTAZIONE

1. **⭐⭐⭐ CRITICO**: Migliorare Reward Shaping
2. **⭐⭐⭐ CRITICO**: Risolvere Bias2 Saturation
3. **⭐⭐ ALTO**: Stabilizzare Learning Rate
4. **⭐⭐ ALTO**: Migliorare Q-Value Clipping
5. **⭐ MEDIO**: Aggiungere Gradient Normalization
6. **⭐ BASSO**: Migliorare Experience Replay

---

## 🎯 RISULTATI ATTESI

Dopo l'implementazione delle soluzioni:

1. **Average Score**: Dovrebbe aumentare da 2,233 a >3,000
2. **Consistenza**: Ratio Average/Best dovrebbe migliorare da 6.8% a >15%
3. **Bias2**: Dovrebbe uscire dalla saturazione (varianza > 0)
4. **Stabilità**: Varianza weights2 dovrebbe ridursi da 17.45 a <10
5. **Apprendimento**: Il modello dovrebbe mostrare trend positivo invece di negativo

---

## ⚠️ RISCHI E MITIGAZIONI

**Rischio 1**: Modifiche troppo aggressive potrebbero destabilizzare ulteriormente
- **Mitigazione**: Implementare modifiche gradualmente, testare dopo ogni cambio

**Rischio 2**: Reward shaping troppo alto potrebbe causare overfitting
- **Mitigazione**: Monitorare performance su validation set (se disponibile)

**Rischio 3**: Learning rate troppo basso potrebbe rallentare apprendimento
- **Mitigazione**: Usare learning rate decay invece di valore fisso basso

---

## 📝 NOTE FINALI

Il modello mostra segni di instabilità e peggioramento. Le modifiche proposte sono mirate a:
1. Guidare meglio l'apprendimento (reward shaping)
2. Risolvere limitazioni strutturali (bias2 saturation)
3. Stabilizzare l'apprendimento (learning rate, gradient normalization)
4. Permettere più flessibilità (Q-value clipping)

**Raccomandazione**: Implementare le modifiche in ordine di priorità, testando dopo ogni gruppo di modifiche.



