# Threshold-Aware Checkout Recommendation System — Quick Commerce MVP

An AI-powered, deep-learning recommendation system designed for quick-commerce checkout scenarios. The system monitors a customer's shopping cart, detects when the total is below the **₹150 free-delivery threshold**, calculates the remaining gap, and generates personalized low-cost product recommendations using a PyTorch neural embedding network to help customers unlock free delivery.

---

## 1. Problem Statement

In quick-commerce applications, customers frequently reach the checkout screen slightly below the free delivery threshold (e.g., Cart Total = ₹120, Threshold = ₹150, Gap = ₹30). Traditional recommendation engines often suggest generic top-selling or high-priced items (e.g., ₹120 Basmati Rice) without considering:
- Current cart contents
- Remaining amount needed to hit free delivery
- User personal preference and purchase history

This project addresses the problem by combining **deep learning personalization** with **threshold-aware candidate filtering and ranking**.

---

## 2. Core MVP Features

- **Cart Monitoring & Total Calculation:** Dynamically computes shopping cart value.
- **₹150 Threshold Detection:** Calculates the exact remaining gap to reach free delivery.
- **PyTorch Neural Recommendation Engine:** Learns latent user-product embeddings from interaction data.
- **Threshold-Aware Hybrid Ranking:** Combines neural relevance ($70\%$) with price proximity to the remaining gap ($30\%$).
- **Dynamic Cart Exclusions:** Automatically excludes items already in the customer's cart.
- **Interactive Streamlit Web Application:** Allows users to switch customer profiles, view carts, receive recommendations, add items, and observe real-time threshold unlocking.

---

## 3. Project Architecture & Directory Structure

```text
Threshold-Aware-Checkout-Recommendation-System-Q-commerce/
│
├── data/
│   ├── products.csv            # Metadata for 50 quick-commerce products
│   ├── interactions.csv        # 1,832 synthetic user-product interactions
│   └── generate_dataset.py     # Reproducible synthetic dataset generator (SEED=42)
│
├── model/
│   ├── recommendation_model.py # PyTorch Embedding Recommendation Architecture
│   ├── train.py                # Neural network training and checkpoint saving script
│   └── saved_model.pt          # Saved PyTorch trained model weights & metadata
│
├── src/
│   ├── data_loader.py          # Data loading, encoding, negative sampling & PyTorch Dataset
│   ├── threshold.py           # Cart total, ₹150 threshold gap & status logic
│   └── recommendation.py      # Recommendation engine & hybrid ranking logic
│
├── tests/
│   ├── test_phase3.py          # Data preparation unit tests
│   ├── test_phase4.py          # Neural model architecture tests
│   ├── test_phase5.py          # Model classification evaluation tests
│   ├── test_phase6.py          # Threshold engine scenario tests (Scenarios A-E)
│   ├── test_phase7.py          # Streamlit app integration tests
│   └── run_all_tests.py        # Master test runner script
│
├── docs/
│   └── DEMO_GUIDE.md           # Academic viva demonstration & presentation guide
│
├── reports/
│   └── phase5_evaluation.md    # Model evaluation metrics & loss report
│
├── app.py                      # Interactive Streamlit checkout application
├── requirements.txt            # Project Python dependencies
├── .env.example                # Environment variables & rules configuration template
├── .gitignore                  # Git ignore specification
└── README.md                   # Project documentation
```

---

## 4. Deep Learning Model Architecture

The deep-learning component is a PyTorch-based **Embedding Recommendation Network**:

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

---

## 5. Threshold-Aware Hybrid Ranking Strategy

Candidates outside the current cart are ranked using a hybrid formula:

$$\text{Final Score} = 0.7 \times \text{Neural Model Score} + 0.3 \times \text{Threshold Score}$$

Where:

$$\text{Threshold Score} = \max\left(0.0, 1.0 - \frac{|\text{product\_price} - \text{gap}|}{\max(\text{gap}, 1.0)}\right)$$

- **Behavior:** Prioritizes products whose prices closely match the remaining gap while honoring the user's personal preference learned by the neural model.

---

## 6. Dataset Overview

Generated programmatically via [`data/generate_dataset.py`](file:///Users/VAIBHAV/Desktop/Threshold-Aware-Checkout-Recommendation-System-Q-commerce/data/generate_dataset.py) with `SEED = 42`:
- **Users:** 100 synthetic customer profiles (`U001` to `U100`) grouped into 6 behavioral personas.
- **Products:** 50 quick-commerce items across 9 categories (Dairy, Bakery, Fruits, Vegetables, Snacks, Beverages, Staples, Personal Care, Household).
- **Prices:** ₹10.00 to ₹120.00 (Mean: ₹41.76), with 45 low-cost products ($\le$ ₹75) available to bridge checkout gaps.
- **Interactions:** 1,832 total interactions (`purchase`: 1,115, `add_to_cart`: 463, `view`: 254).

---

## 7. Model Evaluation Summary

Evaluated on a 20% holdout validation dataset (446 samples):
- **Accuracy:** `75.34%`
- **Precision:** `70.70%`
- **Recall:** `86.55%`
- **F1 Score:** `77.82%`
- **ROC-AUC:** `80.34%`
- **Loss Trajectory:** Training loss decreased from `0.7011` to `0.3166` over 20 epochs. Validation loss reached minimum (`0.6988`) at Epoch 1–4 and rose to `1.2861` at Epoch 20, indicating expected overfitting on the small synthetic dataset.

---

## 8. Installation & Quickstart

### Prerequisites
- Python 3.10+

### Setup Environment
```bash
# 1. Clone/Navigate to workspace
cd Threshold-Aware-Checkout-Recommendation-System-Q-commerce

# 2. Copy environment configuration template
cp .env.example .env

# 3. Install Python dependencies
pip install -r requirements.txt
```

---

## 9. Running the Web Application

Launch the interactive Streamlit application:

```bash
streamlit run app.py
```

The web application will open automatically in your browser at `http://localhost:8501`.

---

## 10. Running Automated Tests

Run the complete master test suite covering all project components:

```bash
python3 tests/run_all_tests.py
```

Or run individual test modules:
```bash
python3 tests/test_phase3.py  # Test Data Preparation
python3 tests/test_phase4.py  # Test Neural Model Architecture
python3 tests/test_phase5.py  # Test Model Evaluation & Metrics
python3 tests/test_phase6.py  # Test Threshold Engine (Scenarios A-E)
python3 tests/test_phase7.py  # Test Streamlit Application Integration
```

---

## 11. Demonstration Scenarios

- **Scenario A (Cart Below Threshold):** Select Customer `U001` (Cart Total = ₹90, Gap = ₹60). System displays `"🚚 You are ₹60.00 away from free delivery."` alongside Top 3 personalized recommendations.
- **Scenario B (Add Recommendation):** Click "➕ Add to Cart" on *Fresh Milk (500ml)* (₹30). Cart total updates to ₹120 (Gap = ₹30), recommendations refresh dynamically, and *Fresh Milk* disappears from recommendations.
- **Scenario C (Unlock Free Delivery):** Click "➕ Add to Cart" on *Burger Buns (4 pcs)* (₹30). Cart total reaches ₹150 (Gap = ₹0). System displays `"🎉 FREE DELIVERY UNLOCKED!"` and clears recommendation cards.
- **Scenario D (Already Unlocked User):** Select Customer `U100` (Cart Total = ₹220 $\ge$ ₹150). System displays free delivery unlocked state immediately with 0 recommendations.

---

## 12. Academic Limitations & Future Scope

### Limitations
- Academic MVP based on synthetic datasets (100 users, 50 products).
- Overfitting on small dataset when trained for 20 epochs.
- Single fixed threshold of ₹150.

### Future Scope
- **Two-Tower Retrieval:** Candidate retrieval scaling for millions of products using FAISS vector search.
- **SASRec Transformer:** Sequential transformer modeling for real-time user clickstreams.
- **Dynamic Threshold Optimization:** Personalizing free-delivery thresholds based on user order history.
