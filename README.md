# Credit Risk Modelling

A machine learning project that classifies loan applicants as low-risk or high-risk (default) using Logistic Regression, with a focus on evaluating performance under class imbalance.

## Problem

Lenders need to predict whether a borrower is likely to default, so they can make informed, data-driven lending decisions. This is a binary classification problem, complicated by the fact that defaults are naturally rare compared to non-defaults — a class imbalance that can make simple accuracy misleading.

## Approach

- **Model:** Logistic Regression
- **Two feature sets compared:**
  - **Full Features** — all available predictors
  - **Filtered Features** — a reduced/selected feature set

## Results

### Full Features
- Training Accuracy: 94%
- Test Accuracy: 91%
- Class distribution: ~9% of test cases (577 of 6,140) were actual defaulters (class 1)
- Precision (class 1): 0.52 | Recall (class 1): 0.07 | F1-score (class 1): 0.12

### Filtered Features
- Training Accuracy: 64%
- Test Accuracy: 60%

## Key finding

While the full-feature model achieves high overall accuracy (91%), its **recall on actual defaulters is only 7%** — meaning it misses the majority of true high-risk borrowers. This is a direct consequence of class imbalance in the dataset (only ~9% of cases are defaults) and highlights why **accuracy alone is a poor metric for imbalanced classification problems** like credit risk. Precision, recall, and F1-score per class give a much more honest picture of real-world model performance.

## Tech stack

- Python
- Jupyter Notebook
- scikit-learn (Logistic Regression, classification metrics)

## Possible extensions

- Address class imbalance directly (SMOTE, class weighting, undersampling)
- Compare against other models (Random Forest, Decision Tree) already explored in related coursework
- Tune the classification threshold to improve recall on defaulters, since in credit risk, missing a defaulter is typically costlier than a false alarm
