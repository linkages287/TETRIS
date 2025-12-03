# Analisi Funzione di Attivazione - Leaky ReLU vs Alternative

**Data:** 2025-12-01  
**Stato Attuale:** Leaky ReLU con leak factor 0.01

---

## 🔍 SITUAZIONE ATTUALE

**Funzione Attuale:** Leaky ReLU  
**Implementazione:** `rl_agent.cpp:60-64`

```cpp
double NeuralNetwork::leaky_relu(double x) const {
    // Leaky ReLU: allows small negative gradients to flow through
    // Prevents "dead neurons" that output 0 for all inputs
    return std::max(0.01 * x, x);  // Leak factor: 0.01 (1% of negative values)
}
```

**Derivata:** `rl_agent.cpp:212`
```cpp
double relu_derivative = (hidden_pre_activation[i] > 0) ? 1.0 : 0.01;
```

**Utilizzo:**
- Hidden layer: Leaky ReLU
- Output layer: Nessuna attivazione (lineare)

---

## ⚠️ PROBLEMI POTENZIALI CON LEAKY RELU

1. **Gradient troppo piccolo per valori negativi:**
   - Leak factor 0.01 significa che i gradient negativi sono molto piccoli
   - Potrebbe rallentare l'apprendimento per neuroni con input negativi

2. **Non è una funzione standard:**
   - ReLU standard è più comune e testato
   - Leaky ReLU è spesso usato con leak factor 0.2-0.3, non 0.01

3. **Complessità non necessaria:**
   - Per Q-Learning, ReLU standard potrebbe essere sufficiente
   - Leaky ReLU è utile solo se ci sono molti "dead neurons"

---

## 🔄 ALTERNATIVE DISPONIBILI

### **1. ReLU Standard** ✓ CONSIGLIATO

**Formula:** `f(x) = max(0, x)`

**Vantaggi:**
- ✓ Più semplice e standard
- ✓ Computazionalmente più veloce
- ✓ Derivata: 1 per x>0, 0 per x<0
- ✓ Funziona bene per la maggior parte dei casi
- ✓ Già implementato nel codice (`relu()`)

**Svantaggi:**
- ⚠️ Può causare "dead neurons" (ma raro con buona inizializzazione)

**Implementazione:**
```cpp
double NeuralNetwork::relu(double x) const {
    return std::max(0.0, x);
}
// Derivata: (x > 0) ? 1.0 : 0.0
```

---

### **2. Tanh (Hyperbolic Tangent)**

**Formula:** `f(x) = tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))`

**Vantaggi:**
- ✓ Output range: [-1, 1] (normalizzato)
- ✓ Derivata sempre positiva
- ✓ Nessun "dead neuron"

**Svantaggi:**
- ⚠️ Saturata per valori estremi
- ⚠️ Computazionalmente più costosa
- ⚠️ Gradient può essere piccolo per valori estremi

**Implementazione:**
```cpp
double NeuralNetwork::tanh(double x) const {
    return std::tanh(x);
}
// Derivata: 1 - tanh(x)^2
```

---

### **3. ELU (Exponential Linear Unit)**

**Formula:** `f(x) = x if x > 0, else α(e^x - 1)`

**Vantaggi:**
- ✓ Smooth per valori negativi
- ✓ Nessun "dead neuron"
- ✓ Migliore di Leaky ReLU in molti casi

**Svantaggi:**
- ⚠️ Computazionalmente più costosa (usa exp)
- ⚠️ Meno standard di ReLU

**Implementazione:**
```cpp
double NeuralNetwork::elu(double x, double alpha = 1.0) const {
    return (x > 0) ? x : alpha * (std::exp(x) - 1.0);
}
// Derivata: (x > 0) ? 1.0 : alpha * exp(x)
```

---

### **4. Swish (Self-Gated Activation)**

**Formula:** `f(x) = x * sigmoid(x)`

**Vantaggi:**
- ✓ Smooth e differenziabile ovunque
- ✓ Performance migliori in molti casi
- ✓ Nessun "dead neuron"

**Svantaggi:**
- ⚠️ Computazionalmente costosa (usa sigmoid)
- ⚠️ Meno standard

**Implementazione:**
```cpp
double NeuralNetwork::swish(double x) const {
    return x / (1.0 + std::exp(-x));  // x * sigmoid(x)
}
// Derivata: swish(x) + sigmoid(x) * (1 - swish(x))
```

---

## 💡 RACCOMANDAZIONE

### **Opzione 1: ReLU Standard** ⭐ CONSIGLIATO

**Motivi:**
1. Già implementato nel codice
2. Standard per reti neurali
3. Più semplice e veloce
4. Funziona bene per Q-Learning
5. Con buona inizializzazione (He initialization), i "dead neurons" sono rari

**Modifiche necessarie:**
- Cambiare `leaky_relu()` → `relu()` in `forward()` e `update()`
- Cambiare derivata da `0.01` → `0.0` per valori negativi

---

### **Opzione 2: Leaky ReLU con leak factor maggiore**

**Se vuoi mantenere Leaky ReLU:**
- Aumentare leak factor da `0.01` a `0.2` o `0.3`
- Più standard e permette gradient più grandi

**Modifiche:**
```cpp
return std::max(0.2 * x, x);  // Leak factor: 0.2 (20% invece di 1%)
// Derivata: (x > 0) ? 1.0 : 0.2
```

---

## 📊 CONFRONTO RAPIDO

| Funzione | Velocità | Dead Neurons | Standard | Consigliato |
|----------|----------|--------------|----------|-------------|
| **ReLU** | ⭐⭐⭐⭐⭐ | Raro | ⭐⭐⭐⭐⭐ | ✅ Sì |
| **Leaky ReLU (0.01)** | ⭐⭐⭐⭐ | No | ⭐⭐⭐ | ⚠️ Attuale |
| **Leaky ReLU (0.2)** | ⭐⭐⭐⭐ | No | ⭐⭐⭐⭐ | ✅ Meglio |
| **Tanh** | ⭐⭐⭐ | No | ⭐⭐⭐⭐ | ⚠️ Per output |
| **ELU** | ⭐⭐⭐ | No | ⭐⭐⭐ | ⚠️ Complesso |
| **Swish** | ⭐⭐ | No | ⭐⭐ | ❌ Troppo complesso |

---

## 🔧 MODIFICHE NECESSARIE PER RELU STANDARD

**File:** `rl_agent.cpp`

1. **Forward pass (linea 74):**
```cpp
// Da:
hidden[i] = leaky_relu(sum);

// A:
hidden[i] = relu(sum);
```

2. **Update pass (linea 96):**
```cpp
// Da:
hidden[i] = leaky_relu(sum);

// A:
hidden[i] = relu(sum);
```

3. **Derivata (linea 212):**
```cpp
// Da:
double relu_derivative = (hidden_pre_activation[i] > 0) ? 1.0 : 0.01;

// A:
double relu_derivative = (hidden_pre_activation[i] > 0) ? 1.0 : 0.0;
```

4. **Commenti (linee 19, 67, 74, 96):**
```cpp
// Da: "use Leaky ReLU to prevent dead neurons"
// A: "use ReLU activation"
```

---

## ✅ CONCLUSIONE

**Raccomandazione:** Cambiare a **ReLU Standard**

**Motivi:**
- Più semplice e standard
- Già implementato
- Funziona bene per Q-Learning
- Con He initialization, i dead neurons sono rari
- Più veloce computazionalmente

**Se preferisci mantenere Leaky ReLU:**
- Aumentare leak factor a 0.2 o 0.3 (più standard)

