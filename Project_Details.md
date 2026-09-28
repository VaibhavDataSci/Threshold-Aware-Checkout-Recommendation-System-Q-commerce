# Technical Architecture & Developer Reference Manual

## Threshold-Aware Checkout Recommendation System (Quick Commerce MVP)

---

## 1. Executive Summary & Core System Intent

The **Threshold-Aware Checkout Recommendation System** is an end-to-end AI application built for quick-commerce platforms. Unlike traditional recommendation systems that optimize purely for historical user affinity or global product popularity—often suggesting high-cost items (e.g., ₹120 Basmati Rice) that overshoot checkout budgets—this system introduces **threshold awareness**.

The primary objective is to monitor a customer's cart value at checkout, detect when the total is below the **fixed ₹150 free-delivery threshold**, calculate the remaining monetary gap, and recommend personalized, low-cost items that bridge the gap precisely while matching the customer's personal preferences.

---

## 2. End-to-End System Architecture & Data Flow

```text
                                 [ USER CHECKOUT SESSION ]
                                             │
                                             ▼
                                  st.session_state (app.py)
                                             │
                                             ├──────────────────────────────────┐
                                             ▼                                  ▼
                                    calculate_cart_total()            get_initial_cart()
                                     (src/threshold.py)                 (src/threshold.py)
                                             │
                                             ▼
                                  Calculate Threshold Gap
                                  gap = max(0, 150 - total)
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
             cart_total >= ₹150?                           cart_total < ₹150?
                       │                                           │
                       ▼                                           ▼
            "Free Delivery Unlocked"                     get_top_recommendations()
          (Recommendations Disabled)                     (src/recommendation.py)
                                                                   │
                                 ┌─────────────────────────────────┴─────────────────────────────────┐
                                 ▼                                                                   ▼
                     Candidate Filtering                                                     Model Inference
                 (Exclude Cart Items)                                                   (EmbeddingRecommendationModel)
                         │                                                                           │
                         └─────────────────────────────────┬─────────────────────────────────────────┘
                                                           ▼
                                                Calculate Hybrid Score
                                       0.7 * model_score + 0.3 * threshold_score
                                                           │
                                                           ▼
                                                  Sort & Select Top 3
                                                           │
                                                           ▼
                                               Render Cards in Streamlit
```

---

## 3. Deep Learning Neural Network Architecture

The neural recommendation model (`EmbeddingRecommendationModel` in [`model/recommendation_model.py`](file:///Users/VAIBHAV/Desktop/Threshold-Aware-Checkout-Recommendation-System-Q-commerce/model/recommendation_model.py)) learns low-dimensional dense vector embeddings for users and products to output a continuous probability score $\in [0, 1]$ representing personal affinity.

```text
               User Index (int)                       Product Index (int)
                     │                                         │
                     ▼                                         ▼
            nn.Embedding (100 × 16)                   nn.Embedding (50 × 16)
                     │                                         │
                     └───────────────────┬─────────────────────┘
                                         ▼
                             Concatenate (Dim: 32)
                                         ▼
                             nn.Linear(32, 32)
                                         ▼
                                   nn.ReLU()
                                         ▼
                             nn.Linear(32, 1)
                                         ▼
                                  nn.Sigmoid()
                                         ▼
                       Relevance Score p(purchase) ∈ [0, 1]
```

### PyTorch Layer Configuration
- **User Embeddings (`user_embedding`):** Maps 100 discrete user IDs (`U001` - `U100`) to 16-dimensional continuous dense vectors.
- **Product Embeddings (`product_embedding`):** Maps 50 discrete product IDs (`P001` - `P050`) to 16-dimensional continuous dense vectors.
- **Concatenation:** Flattens user and product latent vectors into a single 32-dimensional feature vector.
- **Hidden Layer:** Fully-connected `nn.Linear(32, 32)` with Rectified Linear Unit (`ReLU`) activation.
- **Output Layer:** `nn.Linear(32, 1)` with `Sigmoid` activation mapping outputs to positive purchase probabilities.

---

## 4. Mathematical Formulation & Hybrid Ranking

### 4.1. Business Threshold & Remaining Gap
$$\text{cart\_total} = \sum_{i \in \text{Cart}} \text{price}_i$$

$$\text{remaining\_gap} = \max(0.0, 150.0 - \text{cart\_total})$$

$$\text{threshold\_status} = \begin{cases} \text{"Free Delivery Unlocked"}, & \text{if } \text{cart\_total} \ge 150.0 \\ \text{"You are ₹ gap away from free delivery."}, & \text{otherwise} \end{cases}$$

### 4.2. Threshold Proximity Score Function
For a candidate product with price $p_k$ and current remaining gap $g$:

$$S_{\text{threshold}}(p_k, g) = \begin{cases} 0.0, & \text{if } g \le 0 \\ \max\left(0.0, \min\left(1.0, 1.0 - \frac{|p_k - g|}{\max(g, 1.0)}\right)\right), & \text{if } g > 0 \end{cases}$$

- **Properties:**
  - $S_{\text{threshold}} = 1.0$ when product price exactly equals the remaining gap ($p_k = g$).
  - $S_{\text{threshold}}$ degrades smoothly as price deviates above or below the target gap.
  - Bounded strictly within $[0.0, 1.0]$.

### 4.3. Hybrid Final Score
Given predicted neural relevance score $S_{\text{model}}(u, p_k) \in [0, 1]$ and threshold proximity score $S_{\text{threshold}}(p_k, g)$:

$$S_{\text{final}}(u, p_k) = w_{\text{model}} \cdot S_{\text{model}}(u, p_k) + w_{\text{threshold}} \cdot S_{\text{threshold}}(p_k, g)$$

*Default Configuration:* $w_{\text{model}} = 0.7$, $w_{\text{threshold}} = 0.3$.

---

## 5. Synthetic Dataset & Behavioral Persona Specifications

The synthetic dataset (generated reproducibly via [`data/generate_dataset.py`](file:///Users/VAIBHAV/Desktop/Threshold-Aware-Checkout-Recommendation-System-Q-commerce/data/generate_dataset.py) with `SEED=42`) consists of:
- **100 Users:** `U001` through `U100`.
- **50 Products:** `P001` through `P050` across 9 quick-commerce categories.
- **1,832 Interactions:** Distributed as `purchase` (1,115), `add_to_cart` (463), `view` (254).

### 5.1. Product Price Distribution (Threshold Support)
To support bridging arbitrary gaps up to ₹150, the product set contains 45 items priced $\le$ ₹75:
- **₹10 – ₹20:** 13 products (e.g., Potato Chips ₹10, Lemon ₹15, Coriander ₹10)
- **₹20 – ₹30:** 9 products (e.g., Cucumber ₹20, Tomatoes ₹25, Plain Curd ₹25)
- **₹30 – ₹50:** 13 products (e.g., Fresh Milk ₹30, Sandwich Bread ₹35, Brown Bread ₹45)
- **₹50 – ₹75:** 10 products (e.g., Atta ₹55, Amul Butter ₹58, Garbage Bags ₹65)
- **₹75 – ₹100:** 3 products (e.g., Paneer ₹85, Apples ₹90, Oranges ₹80)
- **₹100+:** 2 products (e.g., Orange Juice ₹110, Basmati Rice ₹120)

### 5.2. Customer Behavioral Personas
1. `dairy_bakery` (`U001` - `U020`): 75% preference weight for Dairy & Bakery items.
2. `fruits_vegetables` (`U021` - `U040`): 75% preference weight for Fresh Produce.
3. `snacks_beverages` (`U041` - `U060`): 75% preference weight for Snacks & Drinks.
4. `staples_household` (`U061` - `U080`): 75% preference weight for Kitchen Staples & Cleaning supplies.
5. `personalcare_household` (`U081` - `U090`): 75% preference weight for Personal Hygiene & Cleaning.
6. `mixed` (`U091` - `U100`): Uniform preference distribution across all 9 categories.

---

## 6. Empirical Model Evaluation & Diagnostics

Evaluated on a 20% holdout validation dataset (446 samples; 223 positive, 223 negative):

| Evaluation Metric | Value | Technical Context |
| :--- | :---: | :--- |
| **Accuracy** | **75.34%** | Overall classification accuracy |
| **Precision** | **70.70%** | True purchase ratio among positive predictions |
| **Recall** | **86.55%** | Sensitivity in identifying true positive purchases |
| **F1 Score** | **77.82%** | Harmonic mean of Precision and Recall |
| **ROC-AUC** | **80.34%** | Area under Receiver Operating Characteristic curve |

### 6.1. Confusion Matrix
- **True Positives (TP):** 193
- **True Negatives (TN):** 143
- **False Positives (FP):** 80
- **False Negatives (FN):** 30

### 6.2. Overfitting Trajectory Analysis
- **Epoch 1:** Train Loss = `0.7011` | Val Loss = `0.6988`
- **Epoch 4:** Train Loss = `0.6773` | Val Loss = `0.7188` *(Optimal Generalization)*
- **Epoch 20:** Train Loss = `0.3166` | Val Loss = `1.2861` *(Overfitting Region)*

*Diagnostic Finding:* Training loss steadily decreases while validation loss increases past epoch 4. On small synthetic matrix factorization datasets, prolonged epochs cause embedding layers to memorize training pairs. In production, early stopping at epoch 4–5 with $L_2$ weight decay resolves this variance.

---

## 7. Developer & Source Module Scaffolding

```text
Threshold-Aware-Checkout-Recommendation-System-Q-commerce/
│
├── data/
│   ├── products.csv          # Product metadata schema: (product_id, product_name, category, price)
│   ├── interactions.csv      # Interaction schema: (user_id, product_id, interaction)
│   └── generate_dataset.py   # Dataset generator script with validation suite
│
├── model/
│   ├── recommendation_model.py # EmbeddingRecommendationModel PyTorch nn.Module
│   ├── train.py               # PyTorch training loop & saved_model.pt checkpoint generator
│   └── saved_model.pt         # Saved model weights & bidirectional ID index maps
│
├── src/
│   ├── data_loader.py         # ID encoding, 1:1 negative sampling, PyTorch Dataset wrapper
│   ├── threshold.py           # Business logic: calculate_cart_total, gap, status, get_initial_cart
│   └── recommendation.py      # Engine: load_trained_model, candidate filtering, hybrid score & ranking
│
├── tests/
│   ├── test_phase3.py         # Data preparation unit tests
│   ├── test_phase4.py         # Model architecture & forward pass tests
│   ├── test_phase5.py         # Model evaluation & classification metrics test
│   ├── test_phase6.py         # Recommendation engine & scenarios A-E tests
│   ├── test_phase7.py         # Streamlit app backend integration test
│   └── run_all_tests.py       # Master test suite runner
│
├── docs/
│   └── DEMO_GUIDE.md          # Viva presentation guide & step-by-step demo instructions
│
├── reports/
│   └── phase5_evaluation.md   # Model evaluation report & loss curves
│
├── app.py                     # Streamlit frontend user interface
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variables configuration template
└── README.md                  # Comprehensive user-facing project guide
```

---

## 8. Scalability & Future System Roadmap

While this MVP is optimized for academic clarity and fast execution, a production-grade quick-commerce recommendation platform would extend this architecture across several dimensions:

### 8.1. Candidate Retrieval: Two-Tower Architecture
For platforms with $10^6+$ products, scoring all items per user is computationally prohibitive.
- **User Tower:** Encodes real-time user demographic, location, and past 30-day session embeddings.
- **Item Tower:** Encodes product titles, categories, pricing, and visual features.
- **Vector Indexing:** Approximate Nearest Neighbor (ANN) search via **FAISS** or **HNSW** to retrieve top 100 candidate items in $< 5\text{ ms}$.

### 8.2. Ranking & Sequential Modeling: SASRec Transformer
- Replace static embedding concatenation with **Self-Attention Sequential Recommendation (SASRec)** or **BERT4Rec** to capture in-session item ordering (e.g., user added Tea $\to$ recommend Sugar/Biscuits).

### 8.3. Dynamic Business Rule Optimization
- Replace fixed ₹150 threshold with dynamic per-user threshold optimization algorithms to maximize Average Order Value (AOV) while maintaining cart conversion rates.

### 8.4. MLOps Infrastructure
- **Serving:** Asynchronous **FastAPI** microservice with **Redis** caching for user embeddings and candidate scores.
- **Tracking:** **MLFlow** experiment logging and **Prometheus** / **Grafana** performance dashboards.
- **Deployment:** Containerized **Docker** images orchestrated on **Kubernetes (k8s)** with horizontal pod autoscaling.
