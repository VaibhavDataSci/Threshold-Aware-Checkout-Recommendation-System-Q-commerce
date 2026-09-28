# Threshold-Aware Checkout Recommendation System — Quick Commerce MVP

An AI-powered checkout recommendation system designed for quick-commerce applications. The system monitors a customer's cart, detects when the cart total is below the **₹150 free-delivery threshold**, calculates the remaining gap, and generates personalized, low-cost product recommendations using a deep learning embedding model to help customers reach free delivery.

---

## 1. Project Overview & Problem Statement

In quick-commerce platforms, customers frequently reach the checkout screen with cart totals slightly below the free delivery threshold (e.g., cart total = ₹120, threshold = ₹150, gap = ₹30). Traditional recommendation systems often suggest generic or high-priced items without accounting for:
- Current cart contents
- Remaining amount needed to hit free delivery
- User purchase history and preference

This project bridges that gap by combining **deep learning personalization** with **threshold-aware candidate filtering and ranking**.

---

## 2. Fixed Business Rule
- **Free Delivery Threshold:** Permanent fixed threshold of **₹150**.
- **Threshold Gap Calculation:** $\text{gap} = 150 - \text{cart\_total}$
- **Status:**
  - $\text{cart\_total} \ge 150 \implies \text{Free Delivery Unlocked}$
  - $\text{cart\_total} < 150 \implies \text{Remaining Amount Displayed}$

---

## 3. Project Architecture & Directory Structure

```text
Threshold-Aware-Checkout-Recommendation-System-Q-commerce/
│
├── data/
│   ├── products.csv          # Product metadata dataset
│   └── interactions.csv      # User-product interaction dataset
│
├── model/
│   ├── train.py               # Training and evaluation script
│   └── recommendation_model.py # Deep learning neural network architecture
│
├── src/
│   ├── recommendation.py      # Recommendation engine & ranking logic
│   ├── threshold.py           # Cart total & threshold calculation logic
│   └── data_loader.py         # Dataset loading and preprocessing utilities
│
├── app.py                     # Streamlit interactive application
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variables template
├── .gitignore                 # Git ignore specification
└── README.md                  # Project documentation
```

---

## 4. Synthetic Dataset Overview

The dataset is programmatically generated via [`data/generate_dataset.py`](file:///Users/VAIBHAV/Desktop/Threshold-Aware-Checkout-Recommendation-System-Q-commerce/data/generate_dataset.py) using a fixed random seed (`SEED=42`) for 100% reproducibility.

### Dataset Rationale & Characteristics
- **No External Dependency:** Academic MVP relies strictly on reproducible synthetic quick-commerce data.
- **Realistic Pricing & Threshold Support:** Products range from ₹10 to ₹120 (Mean: ₹41.76), with 45 low-cost products ($\le$ ₹75) specifically available to satisfy various threshold gaps for the ₹150 free delivery goal.
- **Behavioral Personas:** 100 synthetic users divided into 6 preference personas (Dairy/Bakery, Fruits/Vegetables, Snacks/Beverages, Staples/Household, Personal Care/Household, and Mixed) to provide learnable user-item preference patterns for deep learning.

### Statistics
- **Users:** 100 (`U001` - `U100`)
- **Products:** 50 (`P001` - `P050`) across 9 categories (Dairy, Bakery, Fruits, Vegetables, Snacks, Beverages, Staples, Personal Care, Household)
- **Interactions:** 1,832 interactions (`purchase`: 1,115, `add_to_cart`: 463, `view`: 254)

---

## 5. Deep Learning Model Summary

- **Architecture:** Embedding-based Neural Recommendation Model.
- **Inputs:** User ID & Product ID embeddings.
- **Layers:** Concatenate(User Embedding, Product Embedding) $\to$ Dense Layer (ReLU) $\to$ Dense Layer (Sigmoid).
- **Output:** Predicted relevance score $\in [0, 1]$.

---

## 6. Ranking Strategy

$$\text{Final Score} = 0.7 \times \text{Neural Relevance Score} + 0.3 \times \text{Threshold Gap Proximity Score}$$

---

## 7. How to Run

### Setup Environment & Generate Data
```bash
# Copy environment configuration
cp .env.example .env

# Install requirements
pip install -r requirements.txt

# Generate reproducible synthetic dataset
python3 data/generate_dataset.py
```

---

## 8. Development Status
- [x] **Phase 1:** Project Foundation & Directory Scaffolding
- [x] **Phase 2:** Synthetic Dataset Generation
- [x] **Phase 3:** Data Preprocessing & Tensor Preparation
- [x] **Phase 4:** Deep Learning Model Architecture & Training
- [x] **Phase 5:** Model Evaluation & Metrics
- [x] **Phase 6:** Threshold & Cart Logic Implementation
- [x] **Phase 7:** Recommendation Engine Integration
- [x] **Phase 8:** Hybrid Ranking Implementation
- [ ] **Phase 9:** Streamlit UI Development
- [ ] **Phase 10:** End-to-End System Integration
- [ ] **Phase 11:** Demonstration & Scenario Verification
- [ ] **Phase 12:** Final Polish & Documentation
