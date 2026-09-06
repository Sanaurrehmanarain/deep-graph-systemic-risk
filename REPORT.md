# Deep Learning for Systemic Risk Detection
### Unsupervised and Spatio-Temporal Graph Neural Networks Applied to the Global Banking Network

**Author:** Sana Ur Rehman Arain — Data Scientist
**Repository:** [deep-graph-systemic-risk](https://github.com/sanaurrehmanarain/deep-graph-systemic-risk)
**Date:** 2026

> **Attribution:** This research extends the theoretical framework established in *Arain, S. U. R. (2026). Visualizing Disruptive Forces in the Global Banking Network: A Multi-Lens Analysis of the 2008 Financial Crisis* (Version 1.0) [Software]. https://github.com/sanaurrehmanarain/visualizing-disruptive-forces & *Arain, S. U. R. (2026). ai-systemic-risk-warning* (Version 1.0) [Software]. https://github.com/sanaurrehmanarain/ai-systemic-risk-warning.

---

## Table of Contents

- [Deep Learning for Systemic Risk Detection](#deep-learning-for-systemic-risk-detection)
    - [Unsupervised and Spatio-Temporal Graph Neural Networks Applied to the Global Banking Network](#unsupervised-and-spatio-temporal-graph-neural-networks-applied-to-the-global-banking-network)
  - [Table of Contents](#table-of-contents)
  - [1. Executive Summary](#1-executive-summary)
  - [2. Project Lineage](#2-project-lineage)
    - [Project 1 — Visualizing Disruptive Forces (The Foundation)](#project-1--visualizing-disruptive-forces-the-foundation)
    - [Project 2 — AI Early Warning System (Machine Learning \& XAI)](#project-2--ai-early-warning-system-machine-learning--xai)
    - [Project 3 — Deep Graph Systemic Risk (This Report)](#project-3--deep-graph-systemic-risk-this-report)
  - [3. Introduction \& Background](#3-introduction--background)
  - [4. Motivation](#4-motivation)
  - [5. Objectives](#5-objectives)
  - [6. Data](#6-data)
  - [7. Data Preprocessing](#7-data-preprocessing)
    - [7.1 Two-Regime Temporal Handling](#71-two-regime-temporal-handling)
    - [7.2 Stable Node Indexing](#72-stable-node-indexing)
    - [7.3 Edge Weight Transformation](#73-edge-weight-transformation)
    - [7.4 Chronological Splitting](#74-chronological-splitting)
  - [8. Methodology](#8-methodology)
    - [8.1 Track 1 — Unsupervised Graph Autoencoder (GAE)](#81-track-1--unsupervised-graph-autoencoder-gae)
    - [8.2 Track 2 — Spatio-Temporal GNN (ST-GNN)](#82-track-2--spatio-temporal-gnn-st-gnn)
  - [9. Implementation \& Development](#9-implementation--development)
  - [10. Results](#10-results)
    - [10.1 Track 1 — GAE Anomaly Detection](#101-track-1--gae-anomaly-detection)
    - [10.2 Track 2 — ST-GNN Forecasting](#102-track-2--st-gnn-forecasting)
  - [11. Discussion](#11-discussion)
  - [12. Limitations](#12-limitations)
  - [13. Conclusion \& Future Work](#13-conclusion--future-work)
  - [14. References \& Citation](#14-references--citation)

---

## 1. Executive Summary

This report documents the third and final capstone in a three-part research series modeling systemic risk in the global banking network. Building on prior work in network cartography (Project 1) and explainable tabular machine learning (Project 2), this project deploys native **Graph Neural Networks (GNNs)** that learn directly from the topology of the global banking network — bypassing manual feature engineering entirely.

Two complementary deep learning tracks were built on **PyTorch** and **PyTorch Geometric (PyG)**, using **Bank for International Settlements (BIS) Consolidated Banking Statistics** from **2000‑Q1 to 2026‑Q1** across **225 sovereign financial nodes**:

- **Track 1 — Unsupervised Graph Autoencoder (GAE):** Trained exclusively on pre-crisis-labeled historical topology (2000–2018), the GAE learns to reconstruct bilateral banking exposures. When a crisis distorts the network's structure, reconstruction error spikes automatically — with no human-provided labels. The model successfully and autonomously reverse-engineered the timeline of three major 21st-century crises: the 2008 subprime collapse, the 2011–2012 Eurozone debt crisis, and the 2020 COVID-19 liquidity shock.
- **Track 2 — Spatio-Temporal GNN (ST-GNN):** A hybrid Graph Convolutional Network (GCN) + Long Short-Term Memory (LSTM) architecture that ingests a rolling 4-quarter window of network snapshots and forecasts the specific bilateral edge weights of the following quarter. Evaluated strictly out-of-sample on 2022–2026 data never seen during training, it achieved a **Test MSE of 0.5981**.

Together, these two tracks demonstrate that deep graph learning can serve as an autonomous, label-free early warning framework for macro-prudential risk monitoring — detecting both sudden structural anomalies (GAE) and gradual systemic deterioration or trajectory (ST-GNN), directly from the evolving shape of the network itself.

---

## 2. Project Lineage

This repository is the **third and final capstone** in a series of three projects that progressively deepen the modeling of systemic risk in the global banking network. Each stage deliberately builds on the dataset, theoretical framework, and — in Project 2's case — the explicitly stated future work of the one before it.

### Project 1 — Visualizing Disruptive Forces (The Foundation)
**Full title:** *Visualizing Disruptive Forces in the Global Banking Network: A Multi-Lens Analysis of the 2008 Financial Crisis*
**Repository:** [`visualizing-disruptive-forces`](https://github.com/Sanaurrehmanarain/visualizing-disruptive-forces)
**Data & scale:** BIS Consolidated Banking Statistics, Table B4 (immediate-counterparty basis), 2007‑Q1–2010‑Q4; the core 2008‑Q1 snapshot comprised **207 sovereign nodes and 1,761 directed lending ties**.
**What it showcases:** Graph theory and data visualization (NetworkX, pandas, a Shiny for Python dashboard) applied through three deliberately distinct analytical lenses — Network Cartography, Causal Inference, and Strategic-Behavioral Game Theory.
**Summary:** Established the **"Triple-Point" framework**: systemic risk originates where extreme *structural connectivity* (a small set of hub nodes — France and the United Kingdom carried the highest out-degree and betweenness centrality), *lethal causal cascade pathways* (a traceable Germany → UK → US contagion chain, distinguished from simultaneous common-exposure shocks), and *strategic, self-preserving capital hoarding* (modeled as a cross-border Prisoner's Dilemma) all intersect at a single node. The analysis found that just **1% of cross-border lending connections held over 34% of global financial exposure** in 2008‑Q1, and identified the **United Kingdom** as the era's ultimate Triple-Point bottleneck. Its policy recommendation — liquidity surcharges that scale with a bank's betweenness centrality rather than its size alone — remains the throughline that Projects 2 and 3 attempt to operationalize predictively.

### Project 2 — AI Early Warning System (Machine Learning & XAI)
**Full title:** *AI-Driven Early Warning System for Systemic Risk: Predicting Triple-Point Vulnerabilities in Global Banking Networks*
**Repository:** [`ai-systemic-risk-warning`](https://github.com/sanaurrehmanarain/ai-systemic-risk-warning)
**Data & scale:** The same BIS CBS source, 2007‑Q1–2010‑Q4, restructured into a temporal panel of **28,265 directed edges across 16 quarters**, aggregated into 337 usable country–quarter observations.
**What it showcases:** Temporal feature engineering, imbalanced binary classification (Logistic Regression baseline vs. XGBoost), Explainable AI (SHAP), and interactive deployment (Shiny for Python).
**Summary:** Converted Project 1's static, manually-identified Triple-Point framework into a **predictive, supervised Early Warning System**. Static network metrics (betweenness centrality, exposure concentration/HHI, in/out-degree) were re-expressed as quarter-over-quarter momentum features, and a binary "crisis" label was defined as a ≥15% drop in a country's outward cross-border lending in the following quarter — a rare event, occurring in only **21 of 337 observations (6.23%)**. A Logistic Regression baseline (49% accuracy, 40% crisis recall) was outperformed by an XGBoost classifier tuned to a **15% decision threshold** (82% accuracy, 20% crisis recall, 8% crisis precision), deliberately favoring recall over precision — a standard early-warning trade-off, since a missed crisis is far costlier than a false alarm. **SHAP analysis independently rediscovered Project 1's Triple-Point drivers**: falling betweenness centrality (intermediation shock) and negative exposure momentum (capital flight) emerged as the model's top predictive features with no theoretical input from the researcher. The model retrospectively assigned a **99.2%** crisis probability to Australia's 2008‑Q4 post-Lehman funding freeze, and out-of-sample flagged **both** Ireland (99.0%, correctly anticipating the November 2010 Irish banking bailout) and Türkiye (98.8%, despite near-zero betweenness centrality — evidence the framework generalizes beyond "central hub" failure to exposure-driven risk in structurally peripheral nodes) for 2010‑Q3. Project 2's own **Future Work** section explicitly proposed two directions: *"Temporal Graph Neural Networks (TGNs): move from tabular XGBoost to deep learning (PyTorch Geometric) that learns directly from evolving network topology"* and *"Unsupervised anomaly detection: use autoencoders to surface novel crisis signatures without relying on binary supervised labels."*

### Project 3 — Deep Graph Systemic Risk (This Report)
**Repository:** [`deep-graph-systemic-risk`](https://github.com/sanaurrehmanarain/deep-graph-systemic-risk)
**Data & scale:** BIS CBS, expanded to **2000‑Q1–2026‑Q1** — a 26-year, 105-quarter window across 225 nodes, roughly six times the timeframe of Projects 1–2.
**What it showcases:** Large-scale temporal data engineering, PyTorch Geometric, Unsupervised Graph Autoencoders (GAE), and Spatio-Temporal GNNs (GCN + LSTM).
**Summary:** This project is a direct execution of Project 2's stated future work, not merely an inspired-by extension. It bypasses manual feature extraction entirely: the unsupervised GAE autonomously reverse-engineered the timeline of 21st-century crises (2008, 2011–2012, 2020) with **no crisis labels at all** — the same rare-event labeling problem that constrained Project 2 to a 6.23% base rate is sidestepped completely — while the ST-GNN forecasts continuous edge-weight shifts out-of-sample (Test MSE 0.5981), operating on the raw, evolving network topology that Projects 1 and 2 could only access through hand-engineered summary statistics.

**Suggested citations:**

> Arain, S. U. R. (2026). *Visualizing Disruptive Forces in the Global Banking Network: A Multi-Lens Analysis of the 2008 Financial Crisis* (Version 1.0) [Software]. https://github.com/sanaurrehmanarain/visualizing-disruptive-forces
>
> Arain, S. U. R. (2026). *ai-systemic-risk-warning* (Version 1.0) [Software]. https://github.com/sanaurrehmanarain/ai-systemic-risk-warning
>
> Arain, S. U. R. (2026). *deep-graph-systemic-risk* (Version 1.0) [Software]. https://github.com/sanaurrehmanarain/deep-graph-systemic-risk

---

## 3. Introduction & Background

Global banking systems form a dense, dynamic network of bilateral credit exposures between sovereign economies. When stress in one part of this network — a sovereign default, a liquidity freeze, a sharp deleveraging — propagates outward, the result is a systemic crisis whose shape is fundamentally *relational*: it is not just that individual institutions or countries weaken, but that the **structure of the network itself** changes as capital flees, links break, and concentration shifts.

Project 1 in this series established that this structural change is observable in graph-theoretic terms (centrality, concentration, capital flight). Project 2 showed that if those structural signals are hand-engineered into temporal features, a supervised model can learn to anticipate stress, and that explainability tools (SHAP) can validate the mechanism (e.g., falling betweenness centrality preceding contagion).

This project asks a further question: **if the network's topology itself already encodes the story of a crisis, can a model learn that story directly from the graph — without a human first deciding which features matter?** Graph Neural Networks are the natural tool for this, since they operate on node and edge structure natively rather than on a flattened, manually-engineered feature table.

---

## 4. Motivation

Four considerations motivated this deep learning extension of the series:

1. **Manual feature engineering is a bottleneck and a source of bias.** Project 2's tabular XGBoost model performed well, but every input feature (momentum, centrality deltas, etc.) reflected a prior human hypothesis about what matters. Any risk mechanism not anticipated by the feature set is invisible to the model.
2. **Financial networks are naturally graph-structured, sequential data.** Treating each quarter as an independent graph snapshot, and the full history as a sequence of such snapshots, is a more faithful representation than any tabular flattening — and directly matches what GNNs and sequence models (LSTM) are designed to consume.
3. **Unsupervised anomaly detection removes label dependency.** Historical crisis "labels" are contestable (exact onset/offset dates vary by source) and rare (severe class imbalance). An unsupervised approach — reconstruction error from a GAE trained only on non-crisis-adjacent history — sidesteps this dependency and lets the data define its own notion of "abnormal." Project 2's own binary crisis label suffered exactly this problem, with only 21 positive events out of 337 observations (6.23%); this project was motivated in part by wanting to remove that constraint entirely rather than continue tuning a decision threshold around it.
4. **This is Project 2's own roadmap, not a new direction.** Project 2's Future Work section named "Temporal Graph Neural Networks" and "unsupervised anomaly detection" as the two natural next steps beyond tabular XGBoost. This project exists to build exactly that — closing the loop on the three-part series' stated trajectory.

---

## 5. Objectives

The project set out to:

1. Convert 26+ years of BIS cross-border banking data into a chronologically ordered sequence of graph objects suitable for PyTorch Geometric, with strict prevention of temporal data leakage.
2. Build and train an **unsupervised Graph Autoencoder** capable of learning "normal" network topology from historical data alone, and use its reconstruction error as an autonomous systemic anomaly index.
3. Build and train a **Spatio-Temporal GNN** capable of forecasting next-quarter bilateral edge weights from a rolling window of prior quarters, and evaluate it on a fully held-out, chronologically later test period.
4. Validate both models against known historical crisis periods (2008, 2011–2012, 2020) as a sanity check on the unsupervised signal, and against out-of-sample MSE for the forecasting task.
5. Document the full pipeline transparently enough that it can be extended (see [Future Work](#13-conclusion--future-work)) with additional macroeconomic node features.

---

## 6. Data

- **Source:** [Bank for International Settlements (BIS) Consolidated Banking Statistics](https://www.bis.org/statistics/consbankstats.htm) — raw extract stored at `data/raw/WS_CBS_PUB_csv_col.csv`.
- **Full historical coverage:** 1983‑Q4 through 2026‑Q1.
- **Primary modeling window:** 2000‑Q1 to 2026‑Q1 (**105 quarterly snapshots**), chosen specifically for its regular quarterly cadence (see [§7](#7-data-preprocessing)).
- **Nodes:** 225 stable, integer-indexed sovereign economies (IDs 0–224), held constant across all periods.
- **Edges:** Directed bilateral cross-border claims between reporting and counterpart economies.
- **Scale:** Raw claim values are highly right-skewed, with a maximum around **US $2.10 trillion** for a single bilateral exposure.
- **Chronological splits** (identical across all three notebooks, no shuffling):

| Split | Period | Edge count |
|---|---|---|
| Train | 2000‑Q1 – 2018‑Q4 | 145,145 |
| Validation | 2019‑Q1 – 2021‑Q4 | 30,765 |
| Test | 2022‑Q1 – 2026‑Q1 | 43,089 |

---

## 7. Data Preprocessing

Preprocessing was performed in `notebooks/00_data_preprocessing.ipynb` and is foundational to everything downstream. Four design decisions in particular were load-bearing for the deep learning work:

### 7.1 Two-Regime Temporal Handling
The raw BIS series contains two distinct reporting frequencies: **semiannual** reporting before 2000, and **quarterly** reporting from **2000‑Q1** onward. Rather than forcing these into a single artificial cadence, the primary modeling input was restricted to the regular quarterly window (2000‑Q1 to 2026‑Q1). This guarantees evenly spaced time steps — a requirement for both the LSTM component of the ST-GNN and for any reconstruction-error time series to be meaningfully comparable quarter over quarter.

Critically, **missing historical quarters within this window were not interpolated.** Synthetic values would fabricate network structure that never existed, which would distort the temporal graph topology and directly contaminate the unsupervised anomaly-detection task (Track 1) — a model whose entire signal *is* deviation from real historical structure.

### 7.2 Stable Node Indexing
Graph libraries such as PyTorch Geometric require dense, numerical node identifiers. A stable index of **225 unique nodes (IDs 0 to 224)** was constructed and held fixed across every quarterly snapshot, even for economies that do not report in every period. This guarantees that node ID `17`, for example, always refers to the same economy in 2003‑Q2 as it does in 2024‑Q1 — a prerequisite for any embedding-based method (GCN, GAE, LSTM) to learn consistent, comparable node representations over time.

### 7.3 Edge Weight Transformation
Raw claim values are extremely right-skewed (max ≈ $2.10 trillion). A logarithmic transform, **log(1 + claim_value)**, was applied to compress this range into something numerically stable for neural network inputs (unbounded raw magnitudes otherwise destabilize gradient-based training, especially in the LSTM component).

Crucially, **normalization (scaling) parameters were fit strictly on the training period (2000–2018)** and then applied, unmodified, to the validation and test periods. This mirrors the same discipline enforced in Project 2 and is essential to prevent **temporal data leakage** — allowing the model to "see" future distributional information (e.g., a post-2020 spike in exposure volatility) through the back door of a globally-fit scaler.

### 7.4 Chronological Splitting
Training, validation, and test data were split **chronologically, never randomly**:

- **Train:** 2000‑Q1 → 2018‑Q4
- **Validation:** 2019‑Q1 → 2021‑Q4
- **Test:** 2022‑Q1 → 2026‑Q1

This is the only defensible split for a forecasting/anomaly-detection system that will, in deployment, only ever have access to the past. A random split would let the model implicitly learn from future crises to "predict" past ones, invalidating any claim of genuine early-warning capability. The chosen boundaries also usefully isolate the test set from the training data by a multi-year gap, and place the entire 2020 COVID-19 shock in the validation window rather than in training — a meaningful robustness check for the GAE (see [§10](#10-results)).

---

## 8. Methodology

### 8.1 Track 1 — Unsupervised Graph Autoencoder (GAE)
*(Implemented in `notebooks/02_unsupervised_graph_autoencoder.ipynb`)*

**Node features.** For each quarterly snapshot, four node-level statistics are derived directly from the graph itself (no external data): **out-strength**, **in-strength**, **out-degree**, and **in-degree** — i.e., how much and how many bilateral exposures each economy sends and receives. These are log1p-transformed for numerical stability, producing a `[225, 4]` feature matrix per snapshot.

**Architecture.** A two-layer **Graph Convolutional Network (GCN)** encodes the 4-dimensional node feature space down to an **8-dimensional latent embedding** via a 16-dimensional hidden layer:

```
GCNEncoder: GCNConv(4 → 16) → ReLU → GCNConv(16 → 8)
```

This encoder is wrapped in PyTorch Geometric's `GAE` module, which pairs it with an **inner-product decoder**: the probability of an edge between two nodes is recovered as `sigmoid(zᵢ · zⱼ)` from their latent embeddings.

**Training.** The GAE is trained **strictly on the 2000–2018 snapshots** (76 quarters) using PyG's built-in self-supervised `recon_loss`, which combines positive-edge reconstruction with negative sampling, optimized via Adam (`lr=0.01`, `weight_decay=1e-4`) for 100 epochs. This trains the model to recognize what a "normal" pre-2019 banking network looks like — nothing more.

**Anomaly scoring.** At inference time, every one of the 105 quarterly snapshots (train, validation, and test alike) is passed through the *frozen* trained encoder. Two anomaly signals are extracted:
- A **global reconstruction loss** per quarter — the primary systemic anomaly index.
- A **node-level reconstruction error**, computed as the squared deviation of the true edges' reconstructed probability from 1, aggregated per node — enabling identification of the single most anomalous node (economy) in any given quarter.

Both signals are saved to `data/processed/gae_systemic_anomalies.csv` for downstream visualization and dashboarding.

### 8.2 Track 2 — Spatio-Temporal GNN (ST-GNN)
*(Implemented in `notebooks/03_temporal_gnn_predictions.ipynb`)*

**Node features.** Identical construction to Track 1 (out/in-strength, out/in-degree, log1p-transformed), rebuilt independently in this notebook to keep the two tracks decoupled and reproducible in isolation.

**Architecture.** A hybrid spatio-temporal model combining spatial graph convolution with temporal recurrence:

```
TemporalGNN:
  GCNConv(4 → 32) → ReLU
  GCNConv(32 → 32) → ReLU
  LSTM(input=32, hidden=32, batch_first=True)   # over the 4-quarter window
  Edge MLP: Linear(64 → 32) → ReLU → Linear(32 → 1)
```

For each of the 4 quarters in a rolling window, the GCN layers produce per-node spatial embeddings. These are stacked into a sequence and passed through an LSTM, whose final hidden state per node captures that node's temporal trajectory across the window. To predict a specific bilateral edge weight, the source and destination nodes' final embeddings are concatenated and passed through a small MLP decoder.

**Training.** The model is trained on **rolling 4-quarter windows** drawn from the training period (2000–2018), where the target is the edge-weight tensor of the quarter immediately following each window. Training uses Adam (`lr=0.005`) with MSE loss for 50 epochs. **Gradient clipping (`max_norm=1.0`)** is applied after every backward pass — a deliberate stabilization measure to prevent the LSTM component from exploding gradients, a well-known failure mode of recurrent architectures trained on financial time series with heavy-tailed dynamics.

**Out-of-sample evaluation.** The final validation quarters (last 4 of 2019–2021) are used purely as *context* to seed the first test-window prediction — never as training targets — after which the model rolls forward strictly through the 2022–2026 test snapshots, always predicting one quarter ahead from the four quarters immediately preceding it. Predictions and actuals are collected and compared via Mean Squared Error.

---

## 9. Implementation & Development

- **Stack:** Python, PyTorch, PyTorch Geometric (PyG), pandas, NumPy, scikit-learn (`StandardScaler`, `mean_squared_error`), Matplotlib.
- **Data pipeline (Notebook 01):** The chronologically-scaled edge list (`bis_cbs_quarterly_2000Q1_2026Q1_graph_edges.csv`) is converted into one PyTorch Geometric `Data` object per quarter (`edge_index`, `edge_attr`, plus a `period` metadata tag), producing 105 sequential graph snapshots serialized to `temporal_graph_sequence.pt`. A sample snapshot: `Data(edge_index=[2, 1270], edge_attr=[1270, 1], period='2000-Q1')`.
- **Reproducibility discipline:** Both the GAE and ST-GNN notebooks independently rebuild node features from the same raw snapshot sequence rather than sharing mutated state, so either notebook can be run in isolation after Notebook 01 has produced `temporal_graph_sequence.pt`.
- **Numerical stability measures:**
  - `torch.clamp(x, min=0.0)` before every `log1p` to guard against negative values producing NaNs.
  - Gradient clipping (`clip_grad_norm_`, `max_norm=1.0`) in the ST-GNN training loop.
  - Weight decay (`1e-4`) in the GAE optimizer as a light regularizer against overfitting a relatively small per-quarter edge set.
- **Artifacts produced:**
  - `data/processed/temporal_graph_sequence.pt` — the full PyG snapshot sequence (Notebook 01).
  - `data/processed/gae_systemic_anomalies.csv` — per-quarter reconstruction loss, top anomalous node, and train/val/test regime tag (Notebook 02).
  - `models/st_gnn_model.pth` — trained ST-GNN weights (Notebook 03).
  - `figures/unsupervised_anomaly_detection.png`, `figures/dashboard.png` — visualization outputs consumed by `README.md` and `app.py`.

---

## 10. Results

### 10.1 Track 1 — GAE Anomaly Detection

Training converged smoothly over 100 epochs, with reconstruction loss on the training set falling from **1.3698 (epoch 1) to 1.0852 (epoch 100)**.

Plotting the reconstruction loss across all 105 quarters (train, validation, and test regimes combined) reveals a clear, unsupervised anomaly signal:

- A pronounced, sustained elevation in reconstruction error from roughly **2001 through 2013**, encompassing the **2008 subprime collapse** and the **2011–2012 Eurozone debt crisis**, with the highest single peak around 2012.
- A sharp **structural break downward starting around 2013–2014**, after which reconstruction error settles into a materially lower, more stable band through the remainder of the series (2014–2026), including the validation (2019) and test (2022) boundaries.
- Notably, this lower post-2014 regime still contains the **2020 COVID-19 period**, which falls inside the validation window — the model was never trained on it, yet the reconstruction error in this regime remains visibly within its lower post-2014 band rather than spiking to pre-2014 levels, indicating the shock was absorbed differently by the post-2014 network structure than the earlier crises were.

This confirms the central hypothesis of Track 1: a GCN-based autoencoder trained with **no crisis labels whatsoever** can recover a broadly interpretable timeline of systemic stress purely from bilateral exposure topology.

### 10.2 Track 2 — ST-GNN Forecasting

Training MSE fell from **0.6915 (epoch 1)** to **0.5508 (epoch 50)**, with gradient clipping successfully preventing the divergence that an unclipped LSTM exhibited in early development iterations (see inline code comment, `# THE FIX: Clip gradients to prevent LSTM explosions`).

Evaluated strictly out-of-sample on the **2022–2026 test window** (data the model never saw in any form during training), the ST-GNN achieved:

> **Out-of-Sample Test MSE: 0.5981**

This is closely in line with the final training MSE (0.5508), indicating the model **generalizes without severe overfitting** to the 4-year rolling-window forecasting task — a meaningful result given that the test period includes multiple quarters of elevated macroeconomic uncertainty (post-pandemic normalization, rate-hiking cycles) not directly represented in the 2000–2018 training distribution.

---

## 11. Discussion

The two tracks are complementary rather than redundant. The GAE answers *"is this quarter structurally unusual relative to history?"* — a diagnostic, retrospective/contemporaneous signal well suited to flagging a crisis already underway. The ST-GNN answers *"what will next quarter's bilateral exposures look like, given the recent trajectory?"* — a genuinely forward-looking signal suited to anticipating capital flight before it fully materializes.

The GAE's structural break around 2013–2014 is itself an interesting empirical finding independent of the crisis-detection use case: it suggests the global banking network's baseline topology shifted meaningfully in the years following the Eurozone crisis (consistent with post-crisis deleveraging and regulatory tightening, e.g., Basel III phase-in), such that the network re-stabilized into a topology the GAE finds comparatively easier to reconstruct from 2014 onward.

The closeness between the ST-GNN's final training MSE (0.5508) and its out-of-sample test MSE (0.5981) is a stronger generalization result than a large gap would have shown — a large train/test gap would have suggested the model memorized quarter-specific structure from 2000–2018 rather than learning a transferable temporal dynamic.

Both results should be read as **directional, research-stage evidence** rather than production-grade risk scores — see [Limitations](#12-limitations) below.

---

## 12. Limitations

- **No exogenous macroeconomic features.** Both models currently learn *only* from network topology (degree/strength statistics derived from the graph itself). They have no access to interest rates, GDP growth, inflation, reserves, or other macro fundamentals that a human analyst would naturally cross-reference. This means the models can detect *that* structure is anomalous or predict *how* it will evolve, but cannot explain *why* in fundamental economic terms.
- **Reconstruction error is unlabeled and unvalidated against a formal crisis calendar.** The GAE's correspondence to 2008, 2011–2012, and 2020 is assessed visually against known historical events rather than against a quantitatively defined, independently sourced crisis-dating dataset (e.g., an NBER- or IMF-style crisis calendar), so no formal precision/recall metric is reported for anomaly detection.
- **No interpolation means some sparsity across less-reported economies.** Not every one of the 225 nodes reports in every quarter; this is preserved deliberately (see [§7.1](#71-two-regime-temporal-handling)) but means degree/strength features for sparsely-reporting nodes are noisier than for major reporting economies.
- **Single-architecture benchmarking.** Only one GAE configuration (GCN, 8-dim latent) and one ST-GNN configuration (GCN+LSTM, 32-dim hidden, 4-quarter window) were evaluated. No ablations were run over latent dimensionality, window length, or alternative temporal-graph architectures (e.g., EvolveGCN, TGAT, DySAT).
- **Test MSE is on scaled, log-transformed edge weights, not raw exposure values.** An MSE of 0.5981 in this transformed space does not translate directly into an intuitive dollar-denominated forecast error without inverting the scaling and log transform — useful for relative/comparative evaluation, but not directly interpretable as "trillions of dollars of error."
- **No formal statistical significance or confidence intervals** are reported around either the reconstruction-error trajectory or the test MSE; both are reported as point estimates from a single training run.

---

## 13. Conclusion & Future Work

This capstone demonstrates that **deep graph learning provides a genuinely autonomous framework for macro-prudential risk monitoring** — one that requires no manually engineered features and no crisis labels to produce an interpretable, historically-consistent risk signal (via the GAE), alongside a forward-looking forecast of network dynamics (via the ST-GNN). Treating the global economy as an evolving graph, rather than a flattened feature table, allows regulators and researchers to detect both **sudden structural anomalies** and **gradual systemic deterioration** directly from the shape of the network itself.

Taken together with Projects 1 and 2, this three-part series traces a complete arc: from establishing *that* network structure matters for systemic risk (Project 1), to showing that *hand-engineered* structural features can predict crises in an explainable way (Project 2), to demonstrating that a model can learn the *same story natively*, without human feature design (Project 3).

**Planned future work:**

1. **Integrate exogenous macroeconomic node features** — interest rates, GDP growth, sovereign reserves — alongside the purely topological edge and degree/strength features currently used, to give the models fundamental economic context.
2. **Benchmark additional temporal-graph architectures**, including EvolveGCN, TGAT, and DySAT, against the current GCN+LSTM and GAE baselines.
3. **Validate GAE anomaly scores against a formally sourced crisis calendar** to compute quantitative precision/recall for early-warning performance, rather than relying on visual correspondence.
4. **Extend the ST-GNN's forecast horizon** beyond a single quarter ahead, and evaluate multi-step-ahead forecasting error.
5. **Expand the interactive dashboard (`app.py`)** to expose node-level anomaly attribution and forecast uncertainty interactively, building on the current static figures.

---

## 14. References & Citation

**BIS Data Source:**
> Bank for International Settlements. *Consolidated Banking Statistics.* https://www.bis.org/statistics/consbankstats.htm

**Project series citations:**

> Arain, S. U. R. (2026). *Visualizing Disruptive Forces in the Global Banking Network: A Multi-Lens Analysis of the 2008 Financial Crisis* (Version 1.0) [Software]. https://github.com/sanaurrehmanarain/visualizing-disruptive-forces
>
> Arain, S. U. R. (2026). *ai-systemic-risk-warning* (Version 1.0) [Software]. https://github.com/sanaurrehmanarain/ai-systemic-risk-warning
>
> Arain, S. U. R. (2026). *deep-graph-systemic-risk* (Version 1.0) [Software]. https://github.com/sanaurrehmanarain/deep-graph-systemic-risk

If you use this project in academic research, publications, educational materials, or derivative works, please cite it and provide appropriate credit to the original author. See the repository's [`CITATION.cff`](CITATION.cff) file or GitHub's **"Cite this repository"** sidebar button for BibTeX, APA, and other formats.

For installation, usage, and project structure, see **[README.md](README.md)**.