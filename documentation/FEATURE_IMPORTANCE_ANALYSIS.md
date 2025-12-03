# Analisi Feature Importance - Bumpiness e Aggregate Height
## Problema: Feature Poco Addestrate

**Data:** 2025-12-01  
**Feature Problematiche:** Bumpiness (indice 11), Aggregate Height (indice 12)

---

## 🔍 ANALISI DEL PROBLEMA

### Normalizzazione Attuale

**Bumpiness:**
- Valore raw: 0-50 (tipico)
- Normalizzazione: `/50.0`
- Valore normalizzato: 0.0-1.0
- Posizione nello stato: indice 11

**Aggregate Height:**
- Valore raw: 0-200 (tipico)
- Normalizzazione: `/200.0`
- Valore normalizzato: 0.0-1.0
- Posizione nello stato: indice 12

### Problema Identificato

1. **Valori normalizzati troppo piccoli**: Entrambe le feature hanno valori tipici 0.0-1.0
2. **Pesi iniziali piccoli**: Con He initialization, i pesi sono inizializzati con `stddev = sqrt(2.0 / INPUT_SIZE)`
3. **Impatto ridotto**: Per avere lo stesso impatto di altre feature, i pesi devono essere molto grandi
4. **Gradient piccolo**: Con valori normalizzati piccoli, i gradienti sono proporzionalmente piccoli
5. **Update lenti**: I pesi per queste feature si aggiornano più lentamente

### Confronto con Altre Feature

| Feature | Normalizzazione | Range Tipico | Impatto |
|---------|----------------|--------------|---------|
| **Column Heights** | `/20.0` | 0.0-1.0 | ✓ Buono (10 feature) |
| **Holes** | `/100.0` | 0.0-1.0 | ✓ Buono |
| **Bumpiness** | `/50.0` | 0.0-1.0 | ⚠️ **Troppo piccolo** |
| **Aggregate Height** | `/200.0` | 0.0-1.0 | ⚠️ **Troppo piccolo** |
| **Piece Types** | One-hot | 0.0-1.0 | ✓ Buono (binario) |
| **Lines Cleared** | `/100.0` | 0.0-0.2 | ⚠️ Piccolo ma ok |
| **Level** | `/20.0` | 0.0-1.0 | ✓ Buono |

---

## 💡 SOLUZIONI PROPOSTE

### Soluzione 1: Migliorare Normalizzazione ⭐ PRIORITÀ ALTA

**Obiettivo**: Rendere i valori normalizzati più grandi per aumentare l'impatto

**Modifiche:**
```cpp
// Prima
state[idx++] = game.calculateBumpiness(game.board) / 50.0;      // 0.0-1.0
state[idx++] = game.getAggregateHeight(game.board) / 200.0;     // 0.0-1.0

// Dopo
state[idx++] = game.calculateBumpiness(game.board) / 20.0;      // 0.0-2.5 (più grande)
state[idx++] = game.getAggregateHeight(game.board) / 50.0;     // 0.0-4.0 (più grande)
```

**Vantaggi:**
- Valori normalizzati più grandi → gradienti più grandi
- Pesi possono essere più piccoli ma avere stesso impatto
- Più facile per la rete apprendere queste feature

**Svantaggi:**
- Potrebbe causare instabilità se valori troppo grandi
- Richiede aggiustamento dei pesi esistenti

### Soluzione 2: Feature Importance Weighting ⭐ PRIORITÀ ALTA

**Obiettivo**: Aumentare il learning rate per i pesi di queste feature specifiche

**Modifiche:**
```cpp
// Durante l'update, aumentare learning rate per feature importanti
const int BUMPINESS_IDX = 11;
const int AGGREGATE_HEIGHT_IDX = 12;
const double FEATURE_IMPORTANCE_MULTIPLIER = 2.0;  // 2x learning rate

// Per bumpiness
for (int i = 0; i < HIDDEN_SIZE; i++) {
    double weight_gradient = hidden_gradient * relu_derivative * input[BUMPINESS_IDX];
    double lr_multiplier = (j == BUMPINESS_IDX) ? FEATURE_IMPORTANCE_MULTIPLIER : 1.0;
    weights1[BUMPINESS_IDX][i] += learning_rate * lr_multiplier * weight_gradient;
}

// Stesso per aggregate height
```

**Vantaggi:**
- Apprendimento più veloce per queste feature
- Non richiede cambiare normalizzazione
- Mantiene compatibilità con modello esistente

**Svantaggi:**
- Richiede modifiche al codice di update
- Potrebbe causare instabilità se troppo aggressivo

### Soluzione 3: Inizializzazione Migliorata ⭐ PRIORITÀ MEDIA

**Obiettivo**: Inizializzare i pesi per queste feature con valori più grandi

**Modifiche:**
```cpp
// Durante inizializzazione
const int BUMPINESS_IDX = 11;
const int AGGREGATE_HEIGHT_IDX = 12;
const double FEATURE_INIT_MULTIPLIER = 2.0;  // 2x inizializzazione

for (int j = 0; j < INPUT_SIZE; j++) {
    double init_multiplier = (j == BUMPINESS_IDX || j == AGGREGATE_HEIGHT_IDX) 
                             ? FEATURE_INIT_MULTIPLIER : 1.0;
    weights1[j][i] = dist1(gen) * init_multiplier;
}
```

**Vantaggi:**
- Pesi iniziali più grandi per queste feature
- Maggiore impatto iniziale
- Facile da implementare

**Svantaggi:**
- Solo aiuta all'inizio, non durante l'apprendimento
- Potrebbe causare instabilità iniziale

### Soluzione 4: Normalizzazione Logaritmica ⭐ PRIORITÀ BASSA

**Obiettivo**: Usare normalizzazione logaritmica per valori più distribuiti

**Modifiche:**
```cpp
// Normalizzazione logaritmica
double bumpiness_raw = game.calculateBumpiness(game.board);
state[idx++] = std::log1p(bumpiness_raw) / std::log1p(50.0);  // 0.0-1.0 ma più distribuito

double aggregate_height_raw = game.getAggregateHeight(game.board);
state[idx++] = std::log1p(aggregate_height_raw) / std::log1p(200.0);
```

**Vantaggi:**
- Distribuzione più uniforme dei valori
- Migliore per apprendimento

**Svantaggi:**
- Più complesso
- Richiede aggiustamenti

---

## 🎯 RACCOMANDAZIONE

**Implementare Soluzione 1 + Soluzione 2** (combinazione):

1. **Migliorare normalizzazione** per valori più grandi
2. **Aggiungere feature importance weighting** durante l'update

Questo approccio combinato:
- ✓ Aumenta l'impatto delle feature (normalizzazione)
- ✓ Accelera l'apprendimento (weighting)
- ✓ Mantiene stabilità (non troppo aggressivo)
- ✓ Compatibile con modello esistente

---

## 📊 RISULTATI ATTESI

Dopo l'implementazione:

1. **Pesi per bumpiness**: Dovrebbero aggiornarsi più velocemente
2. **Pesi per aggregate height**: Dovrebbero aggiornarsi più velocemente
3. **Varianza pesi**: Dovrebbe aumentare per queste feature (segno di apprendimento)
4. **Performance**: Il modello dovrebbe usare meglio queste feature nelle decisioni

---

## ⚠️ NOTE

- Le modifiche alla normalizzazione richiedono retraining o aggiustamento pesi esistenti
- Il feature importance weighting può essere aggiunto senza modificare il modello esistente
- Monitorare la stabilità dopo le modifiche



