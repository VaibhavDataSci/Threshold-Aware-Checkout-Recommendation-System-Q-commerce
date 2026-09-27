# Threshold-Aware Checkout Recommendation System --- MVP

## Project Source of Truth

> **Purpose:** This document is the single source of truth for the MVP.
> If any implementation decision is unclear, follow this document. Do
> not add features, technologies, or complexity outside this scope
> unless explicitly requested.

------------------------------------------------------------------------

## 1. Project Overview

**Project Name:** Threshold-Aware Checkout Recommendation System

**Domain:** Quick Commerce / E-commerce

**Course:** Advanced Deep Learning

**MVP Goal:** Build a simple AI-powered checkout recommendation system
that detects when a customer's cart is below a **₹150 free-delivery
threshold** and recommends relevant, low-cost products that can help the
customer reach or cross the threshold.

The original project concept contains cart monitoring, threshold
detection, feature extraction, recommendation, ranking, and checkout
updates. This MVP implements these core ideas in a simplified,
understandable form suitable for a semester project.

------------------------------------------------------------------------

## 2. Problem Statement

Customers may reach checkout with a cart value slightly below the
free-delivery threshold.

Example:

-   Cart Value: ₹120
-   Free Delivery Threshold: ₹150
-   Remaining Amount: ₹30

Generic recommendation systems may recommend popular products without
considering:

-   Current cart contents
-   User purchase history
-   Remaining amount required
-   Product price

This project addresses the problem by recommending products that are
both **personally relevant** and **useful for reaching the ₹150
threshold**.

------------------------------------------------------------------------

## 3. MVP Objective

The system must:

1.  Accept a user's current cart.
2.  Calculate the cart value.
3.  Compare the cart value with the ₹150 threshold.
4.  Calculate the remaining amount.
5.  Use a simple deep-learning recommendation model to generate product
    relevance scores.
6.  Filter products based on the threshold gap.
7.  Rank suitable products.
8.  Display the top recommendations.
9.  Allow the user to add a recommended product.
10. Recalculate the cart.
11. Display **Free Delivery Unlocked** when the cart reaches ₹150 or
    more.

------------------------------------------------------------------------

## 4. Fixed Business Rule

### Free Delivery Threshold

**₹150**

This value must remain **₹150 throughout the MVP**.

### Threshold Logic

``` text
gap = 150 - cart_total
```

If:

``` text
cart_total >= 150
```

then:

``` text
Free Delivery = Unlocked
```

Otherwise:

``` text
Free Delivery = Not Unlocked
```

------------------------------------------------------------------------

## 5. Example User Journey

Initial cart:

``` text
Milk       ₹40
Bread      ₹45
Eggs       ₹35
----------------
Total     ₹120
```

Threshold:

``` text
₹150
```

Gap:

``` text
₹30
```

The system displays:

> You are ₹30 away from free delivery.

Possible recommendations:

``` text
Biscuits    ₹30
Banana      ₹25
Chocolate   ₹40
```

If the user adds Biscuits:

``` text
₹120 + ₹30 = ₹150
```

The system displays:

> Free delivery unlocked!

------------------------------------------------------------------------

## 6. System Workflow

``` text
User
 ↓
Shopping Cart
 ↓
Calculate Cart Total
 ↓
Threshold Detection
 ↓
Is Cart >= ₹150?
 ├── YES → Free Delivery Unlocked
 │
 └── NO
       ↓
 Calculate Remaining Amount
       ↓
 Generate Product Candidates
       ↓
 Deep Learning Recommendation Model
       ↓
 Calculate Relevance Scores
       ↓
 Threshold-Aware Filtering
       ↓
 Rank Products
       ↓
 Display Top Recommendations
       ↓
 User Adds Product
       ↓
 Cart Updated
       ↓
 Threshold Checked Again
```

The workflow follows the original project's core methodology:

**Cart Monitoring → Threshold Detection → Feature Extraction →
Recommendation Engine → Ranking Model → Checkout Update**

------------------------------------------------------------------------

## 7. Dataset

The MVP will use a **synthetic e-commerce dataset**.

No real customer data is required.

### Products Dataset

Required fields:

``` text
product_id
product_name
category
price
```

### User Interactions Dataset

Required fields:

``` text
user_id
product_id
interaction
```

Optional:

``` text
interaction_type
```

Possible interaction types:

``` text
purchase
view
add_to_cart
```

For simplicity, the primary training signal can be based on purchases.

------------------------------------------------------------------------

## 8. Synthetic Dataset Requirements

Generate a realistic but small dataset.

Recommended scale:

``` text
Users:        50–100
Products:     30–50
Interactions: 500–2000
```

Suggested product categories:

``` text
Dairy
Bakery
Fruits
Vegetables
Snacks
Beverages
Staples
Personal Care
Household
```

Product prices should generally be realistic for a quick-commerce
application.

The dataset must contain enough variation for the recommendation model
to learn basic user-product relationships.

------------------------------------------------------------------------

## 9. Deep Learning Component

The MVP must contain a genuine neural-network recommendation component
because this is an **Advanced Deep Learning** project.

Do **not** implement the full production architecture from the original
proposal.

Use a simplified **embedding-based recommendation model**.

Conceptually:

``` text
User ID
   ↓
User Embedding
   │
   ├──────→ Neural Network → Relevance Score
   │
Product ID
   ↓
Product Embedding
```

The model learns representations of users and products and predicts how
relevant a product is to a user.

------------------------------------------------------------------------

## 10. Simplified Model Architecture

Use:

``` text
User Embedding
+
Product Embedding
        ↓
Concatenate
        ↓
Dense Layer
        ↓
ReLU
        ↓
Dense Layer
        ↓
Sigmoid
        ↓
Recommendation Score
```

Keep the model small.

Prioritize:

-   Simplicity
-   Fast training
-   Explainability
-   Successful demonstration

Do not optimize for production-scale accuracy.

------------------------------------------------------------------------

## 11. Recommendation Logic

The final recommendation must not depend only on the neural-network
score.

The system should consider:

### A. Personal Relevance

How relevant is the product according to the neural model?

### B. Threshold Relevance

How useful is the product for closing the remaining amount?

### C. Cart Relevance

Does the product make sense given the current cart?

### Basic Filtering Rules

If the cart is below ₹150:

1.  Do not recommend products already in the cart.
2.  Prefer products whose price is close to or below the remaining
    amount.
3.  Use the neural model to calculate relevance.
4.  Rank eligible products.
5.  Show the top 3.

Example:

``` text
Cart = ₹120
Gap  = ₹30
```

Products around the gap should be prioritized:

``` text
₹20
₹25
₹30
₹35
₹50
```

The system should not blindly recommend the most expensive products.

------------------------------------------------------------------------

## 12. Important Recommendation Rule

The system must **not force the user to purchase anything**.

Recommendations are suggestions.

If a highly relevant product costs more than the remaining gap, it may
still be shown if it is genuinely relevant, but products reasonably
close to the gap should receive higher threshold relevance.

------------------------------------------------------------------------

## 13. Ranking

The final ranking should combine:

``` text
Deep Learning Relevance
+
Price/Threshold Relevance
```

A simple conceptual formula can be:

``` text
final_score =
    0.7 × model_score
    +
    0.3 × threshold_score
```

The exact implementation may use a simple alternative weighting if
necessary, but the principle must remain:

> **Personalization + usefulness for crossing the threshold.**

Keep the ranking understandable.

------------------------------------------------------------------------

## 14. User Interface

Use **Streamlit**.

The MVP should have one main page.

### Customer Selection

``` text
Select User
```

### Cart

Display:

``` text
Product
Price
Quantity
```

### Cart Summary

Display:

``` text
Cart Total: ₹120
Free Delivery Threshold: ₹150
Amount Remaining: ₹30
```

### Recommendation Section

Display:

``` text
AI Recommendations

1. Biscuits — ₹30
2. Banana — ₹25
3. Chocolate — ₹40
```

Each recommendation should contain:

``` text
Product
Price
Recommendation Score
Add to Cart
```

------------------------------------------------------------------------

## 15. Required Demo Scenario

The final application must demonstrate:

``` text
User selects
     ↓
Existing cart loaded
     ↓
Cart is below ₹150
     ↓
System calculates gap
     ↓
AI recommendations appear
     ↓
User adds recommendation
     ↓
Cart total changes
     ↓
System recalculates gap
     ↓
Cart reaches ₹150+
     ↓
"Free Delivery Unlocked"
```

This is the primary demonstration for evaluation.

------------------------------------------------------------------------

## 16. Technology Stack

### Required

``` text
Python
Pandas
NumPy
PyTorch OR TensorFlow
Streamlit
Scikit-learn
```

### Optional

``` text
Matplotlib
```

Use Matplotlib only if simple training/result visualizations are useful.

------------------------------------------------------------------------

## 17. Technologies NOT Required

Do not implement the following in the MVP:

``` text
Kubernetes
Docker
Redis
PostgreSQL
Kafka
RabbitMQ
MLflow
Prometheus
Grafana
FAISS
Feature Store
API Gateway
Cloud deployment
Payment Gateway
Inventory Service
CI/CD
```

These are production-scale components and are outside the MVP scope.

------------------------------------------------------------------------

## 18. Two-Tower and SASRec Scope

The original project proposal mentions:

-   Two-Tower model for candidate generation
-   SASRec Transformer for ranking

For this MVP:

**Do not implement full Two-Tower + SASRec.**

Instead, implement the simplified embedding-based neural recommendation
model.

The simplified model should demonstrate the same fundamental idea:

> Learn user-product relevance and use that relevance to generate
> personalized recommendations.

Full Two-Tower retrieval and SASRec ranking are future enhancements, not
MVP requirements.

------------------------------------------------------------------------

## 19. Project Structure

``` text
threshold-aware-checkout/
│
├── data/
│   ├── products.csv
│   └── interactions.csv
│
├── model/
│   ├── train.py
│   └── recommendation_model.py
│
├── src/
│   ├── recommendation.py
│   ├── threshold.py
│   └── data_loader.py
│
├── app.py
│
├── requirements.txt
│
└── README.md
```

Keep the codebase simple and understandable.

Do not create unnecessary microservices or complex abstractions.

------------------------------------------------------------------------

## 20. File Responsibilities

### `data/products.csv`

Stores product information:

``` text
product_id
product_name
category
price
```

### `data/interactions.csv`

Stores user-product interactions:

``` text
user_id
product_id
interaction
```

### `model/train.py`

Responsible for:

``` text
Load dataset
↓
Prepare user/product IDs
↓
Create training samples
↓
Train neural recommendation model
↓
Evaluate model
↓
Save trained model
```

### `model/recommendation_model.py`

Contains the neural recommendation architecture.

### `src/threshold.py`

Contains:

``` text
calculate_cart_total()
calculate_threshold_gap()
is_threshold_reached()
```

### `src/recommendation.py`

Contains:

``` text
generate_candidates()
predict_relevance()
filter_products()
calculate_threshold_score()
rank_products()
get_top_recommendations()
```

### `src/data_loader.py`

Loads products, interactions, and required data.

### `app.py`

Contains the Streamlit application and connects the UI to the
recommendation system.

------------------------------------------------------------------------

## 21. Expected Model Output

For a selected user, the model may produce:

``` text
Product       Relevance Score
Chocolate       0.91
Biscuits        0.87
Banana          0.82
Juice           0.71
Rice            0.55
```

The recommendation system then applies threshold-aware filtering and
ranking.

Final recommendations might be:

``` text
Chocolate   ₹40
Biscuits    ₹30
Banana      ₹25
```

------------------------------------------------------------------------

## 22. Evaluation

The MVP should demonstrate both model performance and system behavior.

### Model Evaluation

If practical, calculate:

``` text
Precision@K
Recall@K
```

Also show:

``` text
Training Loss
Validation Loss
```

### System Evaluation

Demonstrate:

``` text
Initial Cart
Threshold
Remaining Amount
Recommended Products
Updated Cart
Threshold Status
```

Do not invent performance numbers.

Do not claim that the system actually increases real-world revenue, AOV,
or reduces real-world abandonment unless this is experimentally
measured.

------------------------------------------------------------------------

## 23. Success Criteria

The MVP is complete when:

-   [ ] Synthetic dataset is created.
-   [ ] Neural recommendation model is trained.
-   [ ] User/product embeddings are implemented.
-   [ ] Cart calculation works.
-   [ ] ₹150 threshold detection works.
-   [ ] Remaining amount is calculated.
-   [ ] Products already in cart are excluded.
-   [ ] Recommendations are generated.
-   [ ] Recommendations consider the threshold gap.
-   [ ] Top 3 products are displayed.
-   [ ] User can add recommendations to the cart.
-   [ ] Cart total updates dynamically.
-   [ ] Free delivery status changes when cart \>= ₹150.
-   [ ] Streamlit application runs successfully.
-   [ ] End-to-end demo works.

------------------------------------------------------------------------

## 24. Out of Scope

The MVP will not attempt to solve:

-   Production-scale recommendation
-   Millions of users
-   Distributed real-time serving
-   Advanced sequential recommendation
-   Full Two-Tower retrieval
-   SASRec Transformer
-   Cloud deployment
-   Microservices
-   MLOps
-   Real payment processing
-   Real inventory integration
-   Real customer data
-   A/B testing
-   Actual business revenue measurement

------------------------------------------------------------------------

## 25. Future Enhancements

These may be mentioned in the final presentation but should not be
implemented unless the MVP is already complete:

1.  Full Two-Tower candidate retrieval.
2.  SASRec Transformer ranking.
3.  Real customer transaction data.
4.  PostgreSQL integration.
5.  Redis caching.
6.  FastAPI recommendation service.
7.  MLflow experiment tracking.
8.  Docker/Kubernetes deployment.
9.  Real-time monitoring.
10. Continuous model retraining.

------------------------------------------------------------------------

## 26. Final Project Definition

> **Threshold-Aware Checkout Recommendation System is a simplified
> deep-learning-based recommendation system for quick commerce. The
> system monitors a customer's cart, detects when the cart is below the
> ₹150 free-delivery threshold, calculates the remaining amount, and
> uses a neural recommendation model to recommend personalized low-cost
> products. The recommendations are filtered and ranked based on both
> user-product relevance and threshold usefulness. When the user adds a
> recommended product and the cart reaches ₹150 or more, the system
> displays that free delivery has been unlocked.**

------------------------------------------------------------------------

## 27. Agent Development Rules

The coding agent must follow these rules throughout development:

1.  Treat this document as the **source of truth**.
2.  Do not expand the project scope without explicit user approval.
3.  Do not introduce unnecessary technologies.
4.  Do not build production-scale infrastructure.
5.  Keep the implementation understandable to an undergraduate student.
6.  Prefer simple, readable Python code.
7.  Use synthetic data only.
8.  Keep the free-delivery threshold fixed at **₹150**.
9.  The neural recommendation model must be real, but simple.
10. The recommendation system must consider both personalization and
    threshold relevance.
11. The UI must demonstrate the complete checkout flow.
12. Do not fabricate model metrics or business results.
13. If a design decision is ambiguous, choose the simplest
    implementation consistent with this document.
14. If a requested change conflicts with this document, ask for
    clarification rather than silently changing the project scope.
15. Do not implement future enhancements until the core MVP is working
    end-to-end.

------------------------------------------------------------------------

## 28. Definition of Done

The project is considered successfully completed when a user can run the
application and demonstrate:

``` text
Select User
    ↓
View Existing Cart
    ↓
See Cart Total
    ↓
See Remaining Amount to ₹150
    ↓
Receive AI Recommendations
    ↓
Add Recommended Product
    ↓
Cart Updates
    ↓
Reach ₹150+
    ↓
Free Delivery Unlocked
```

The final system should be **simple, functional, explainable, and
demonstrable** rather than production-scale.
