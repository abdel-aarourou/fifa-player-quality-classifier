# FIFA 19 Player Quality Classifier

Deep neural network for classifying FIFA 19 football players into four quality tiers based on their technical attributes.

**Course:** Aprendizaje Automático II — Grado en Ciencia de Datos e Inteligencia Artificial, UPM  
**Authors:** Abdelaziz Aarourou Uarda, Mario Martín Muñoz, Rodrigo Pinto Aguilera, Daniel Barahona Moreno

---

## Problem

Given 22 technical attributes of a football player (Crossing, Finishing, Reactions, etc.), predict their overall quality tier:

| Class | Overall Score Range |
|---|---|
| Poor | [46, 62] |
| Intermediate | [63, 66] |
| Good | [67, 71] |
| Excellent | [72, 94] |

The dataset comes from [FIFA 19 on Kaggle](https://www.kaggle.com/datasets/javagarm/fifa-19-complete-player-dataset) and contains 16,134 instances after preprocessing (goalkeepers removed, low-correlation attributes dropped).

---

## Final Model (Step 32)

| Parameter | Value |
|---|---|
| Architecture | Dense(64, ReLU) → Dense(64, ReLU) → Dense(4, Softmax) |
| Regularization | L1 = 0.0001 on all hidden layers |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Batch size | 32 |
| Epochs | 300 |
| Loss | Categorical Crossentropy |

### Results

| Metric | Dev Set | Test Set |
|---|---|---|
| Accuracy | 88.2% | 86.74% |
| Error | 11.8% | 13.26% |
| Bias (vs. 10% Bayesian) | 1.7% | 3.26% |
| Variance | 0.2% | — |

Classification report on the final test set:

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Poor | 0.942 | 0.916 | 0.929 |
| Intermediate | 0.797 | 0.814 | 0.806 |
| Good | 0.807 | 0.855 | 0.830 |
| Excellent | 0.943 | 0.884 | 0.913 |
| **Weighted avg** | **0.870** | **0.867** | **0.868** |

---

## Design Process

37 configurations were evaluated across 9 phases:

1. **Baseline** — Logistic regression (31.5% error → high bias, confirmed nonlinearity needed)
2. **Hidden layers + epochs** — 2×64 ReLU, 300 epochs → 11% error
3. **L2 regularization** — Marginal improvement; L2=0.001 over-regularized
4. **Dropout** — 0.2 rate caused underfitting on a small network
5. **Capacity increase** — Larger networks overfit; required regularization
6. **Optimizers** — Adam with LR=0.0001 gave the cleanest convergence
7. **Training adjustments** — Batch Normalization incompatible with this config; Early Stopping efficient but suboptimal
8. **Initializers & activations** — HeNormal and ELU showed no improvement over defaults
9. **L1 regularization** — Best result: L1=0.0001 leverages the pre-selection of attributes by correlation, achieving 0.2% variance

Key finding: L1 outperformed L2 and Dropout individually, likely because its sparsifying effect is consistent with the attribute pre-selection by correlation threshold.

---

## Repository Structure

```
.
├── fifa19_player_quality_classifier.ipynb    # Full experiment notebook (37 steps)
├── FootballPlayerPreparedCleanAttributes.csv # 16,134 × 22 scaled input features
├── FootballPlayerOneHotEncodedClasses.csv    # One-hot encoded labels (4 classes)
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
jupyter notebook practica2.ipynb
```

The notebook expects the two CSV files in the same directory. All 37 steps are self-contained — run them sequentially to reproduce the full design process, or jump directly to Step 32 for the final model.

---

## Data Split

| Split | Size | Instances |
|---|---|---|
| Train | 80% | 12,907 |
| Dev | 10% | 1,614 |
| Test | 10% | 1,614 |

The test set was kept completely held out until the final model was selected.
