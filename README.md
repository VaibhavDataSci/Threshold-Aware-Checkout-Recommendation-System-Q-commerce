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

## 4. Deep Learning Model Summary

- **Architecture:** Embedding-based Neural Recommendation Model.
- **Inputs:** User ID & Product ID embeddings.
- **Layers:** Concatenate(User Embedding, Product Embedding) $\to$ Dense Layer (ReLU) $\to$ Dense Layer (Sigmoid).
- **Output:** Predicted relevance score $\in [0, 1]$.

---

## 5. Ranking Strategy

$$\text{Final Score} = 0.7 \times \text{Neural Relevance Score} + 0.3 \times \text{Threshold Gap Proximity Score}$$

---

## 6. How to Run

### Setup Environment
```bash
# Copy environment configuration
cp .env.example .env

# Install requirements
pip install -r requirements.txt
```

---

## 7. Development Status
- [x] **Phase 1:** Project Foundation & Directory Scaffolding
- [ ] **Phase 2:** Synthetic Dataset Generation
- [ ] **Phase 3:** Data Preprocessing & Tensor Preparation
- [ ] **Phase 4:** Deep Learning Model Architecture & Training
- [ ] **Phase 5:** Model Evaluation & Metrics
- [ ] **Phase 6:** Threshold & Cart Logic Implementation
- [ ] **Phase 7:** Recommendation Engine Integration
- [ ] **Phase 8:** Hybrid Ranking Implementation
- [ ] **Phase 9:** Streamlit UI Development
- [ ] **Phase 10:** End-to-End System Integration
- [ ] **Phase 11:** Demonstration & Scenario Verification
- [ ] **Phase 12:** Final Polish & Documentation
