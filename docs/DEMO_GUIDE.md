# Academic Demonstration & Viva Presentation Guide

**Project Name:** Threshold-Aware Checkout Recommendation System — Quick Commerce MVP  
**Domain:** Quick Commerce / E-commerce  
**Course:** Advanced Deep Learning  

---

## 1. Problem Statement & Motivation

In quick-commerce platforms (e.g., 10-minute grocery delivery services), customers frequently reach the checkout page with cart totals slightly below the platform's **free delivery threshold** (e.g., cart total = ₹120 vs. threshold = ₹150, remaining gap = ₹30).

Standard recommendation engines typically recommend top-selling or high-priced items (e.g., ₹120 Basmati Rice or ₹90 Olive Oil) based solely on global popularity or generic user interest. These recommendations fail to address two critical customer constraints at checkout:
1. **Personal Preference:** Recommending items the customer actually wants.
2. **Threshold Goal:** Recommending low-cost items that bridge the specific remaining delivery gap (e.g., ₹25–₹35 items) to unlock free delivery without overspending.

---

## 2. Proposed Solution Architecture

Our solution combines **deep learning personalization** with **threshold-aware candidate filtering and hybrid ranking**:

```text
User Selects Profile (e.g. U001)
              ↓
  Load Current Shopping Cart
              ↓
     Calculate Cart Total (e.g., ₹120.00)
              ↓
  Check ₹150 Free Delivery Threshold
              ↓
    Calculate Gap (150 - Cart Total = ₹30.00)
              ↓
 Candidate Filtering (Exclude items in cart)
              ↓
Deep Learning Model Scoring (User & Product Embeddings)
              ↓
  Threshold Relevance Scoring (Gap Proximity)
              ↓
    Hybrid Ranking: 0.7 * Model + 0.3 * Threshold
              ↓
  Display Top 3 Recommendations (e.g., Fresh Milk ₹30)
              ↓
 Customer Adds Product → Cart Updates → Unlocks Free Delivery!
```

---

## 3. Deep Learning Model Architecture

The neural network component is a PyTorch-based **Embedding Recommendation Network** (`EmbeddingRecommendationModel`):

```text
User ID (int)               Product ID (int)
     │                             │
     ▼                             ▼
User Embedding               Product Embedding
 (100 × 16)                     (50 × 16)
     │                             │
     └──────────────┬──────────────┘
                    ▼
          Concatenate (32 dims)
                    ▼
          Dense Layer (32 → 32)
                    ▼
             ReLU Activation
                    ▼
          Dense Layer (32 → 1)
                    ▼
          Sigmoid Activation
                    ▼
       Predicted Relevance Score ∈ [0, 1]
```

- **Objective:** Learn latent user and product representations to predict how relevant a given product is to a specific user based on historical purchase patterns.

---

## 4. Threshold-Aware Hybrid Ranking Strategy

The recommendation engine scores every candidate product outside the current cart using a combined formula:

$$\text{Final Score} = 0.7 \times \text{Neural Model Score} + 0.3 \times \text{Threshold Score}$$

Where the **Threshold Score** measures price proximity to the remaining gap:

$$\text{Threshold Score} = \max\left(0.0, 1.0 - \frac{|\text{product\_price} - \text{gap}|}{\max(\text{gap}, 1.0)}\right)$$

- **Intuition:** A ₹30 product when the gap is ₹30 gets $\text{Threshold Score} = 1.0$. A ₹120 product receives a low threshold score, preventing the engine from recommending excessively expensive overshoots unless the model score is exceptionally high.

---

## 5. Step-by-Step Viva Demonstration Sequence (2–3 Minutes)

Follow this exact sequence when presenting the project to the professor:

### Step 1: Scenario A — Cart Below Threshold
1. Launch Streamlit app: `streamlit run app.py`
2. Select Customer **`U001`**.
3. Point out the current cart:
   - *Toned Milk (1L)* (₹55.00) + *White Sandwich Bread* (₹35.00)
   - **Cart Total:** ₹90.00
   - **Free Delivery Threshold:** ₹150.00
   - **Remaining Gap:** ₹60.00
   - **Status Alert:** `"🚚 You are ₹60.00 away from free delivery."`
4. Show the **Top 3 AI Recommendations** (e.g., *Herbal Toothpaste* ₹60, *Fresh Milk* ₹30, *Amul Butter* ₹58) and highlight how prices target the ₹60 gap while scores reflect personal preference.

### Step 2: Scenario B — Dynamic Add-to-Cart & Recalculation
1. Click **"➕ Add to Cart"** on **Fresh Milk (500ml)** (₹30.00).
2. Show that the cart updates immediately:
   - **New Cart Total:** ₹120.00
   - **New Remaining Gap:** ₹30.00
   - **Status Alert:** `"🚚 You are ₹30.00 away from free delivery."`
3. Point out that *Fresh Milk* disappeared from recommendations, and new recommendations specifically targeting the ₹30 gap (e.g., *Burger Buns* ₹30, *Potatoes* ₹30) appeared dynamically.

### Step 3: Scenario C — Free Delivery Unlocked
1. Click **"➕ Add to Cart"** on **Burger Buns (4 pcs)** (₹30.00).
2. Point out the updated UI:
   - **New Cart Total:** ₹150.00
   - **Remaining Gap:** ₹0.00
   - **Banner:** `"🎉 FREE DELIVERY UNLOCKED!"`
   - **Recommendations:** Recommendation cards disappear, confirming checkout goal reached.

### Step 4: Scenario D — Customer Already Unlocked
1. Switch Customer dropdown to **`U100`**.
2. Show initial cart total = ₹220.00.
3. Show that free delivery is immediately displayed as unlocked with 0 unnecessary recommendations.

---

## 6. Academic Model Evaluation & Limitations

### Evaluation Results (Validation Set = 446 samples)
- **Accuracy:** `75.34%`
- **Precision:** `70.70%`
- **Recall:** `86.55%`
- **F1 Score:** `77.82%`
- **ROC-AUC:** `80.34%`

### Overfitting Discussion (Viva Answer)
- *Observed Loss Progression:* Training loss decreased from `0.7011` to `0.3166` over 20 epochs, while validation loss reached its minimum (`~0.70`) at Epoch 4 and rose to `1.2861` by Epoch 20.
- *Explanation:* This demonstrates standard overfitting on a small synthetic dataset (100 users, 50 products). The embedding model memorizes specific user-product training pairs over prolonged epochs.

### Key Limitations
1. **Synthetic Data Scale:** 100 users and 50 products built for academic demonstration.
2. **Static Session Context:** Does not model real-time sequential interaction timestamps or web clickstream logs.
3. **Price Sensitivity:** Assumes fixed ₹150 free delivery threshold.

---

## 7. Future Scope & Enhancements

1. **Two-Tower Candidate Retrieval:** Scale to 1M+ products using dual encoders and FAISS ANN search.
2. **SASRec Transformer Ranking:** Incorporate sequential transformer models for real-time user session modeling.
3. **Dynamic Threshold Optimization:** Machine learning models to optimize per-user free-delivery thresholds based on average order value (AOV).
4. **MLOps & Real-time Pipeline:** Deploy model serving on FastAPI, Redis caching, Prometheus metrics, and automated retraining pipelines.
