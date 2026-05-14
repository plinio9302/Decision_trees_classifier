# Decision Trees Classifier — Spam Detection with Ensemble Methods

> **Course:** MATH 373 — Intro to Machine Learning
> **Topic:** Tree-Based Methods: Decision Trees, Bagging & Random Forests
> **Author:** Plinio Durango

---

## Overview

This repository explores **tree-based classification** applied to email spam detection. Using a real-world dataset of labeled `.eml` email files, we build and compare three progressively more powerful models:

1. **Decision Tree** — a single, interpretable tree
2. **Bagging** — reduces variance by averaging many trees
3. **Random Forest** — bagging with decorrelated trees via feature subsampling

We follow a consistent pipeline: **load → preprocess → vectorize → train → tune → evaluate → compare**.

---

## Repository Structure

```
Decision_trees_classifier/
├── Data/
│   ├── TRAINING/              # Training .eml email files
│   └── TESTING/               # Testing .eml email files
├── Spam Classifier.ipynb      # Main notebook: full spam classification pipeline
├── classification_tree.ipynb  # Case study: Tree methods on news popularity data
├── Ensemble_Methods_Case_Study.pdf  # Course case study reference PDF
└── README.md
```

---

## Notebooks

### `Spam Classifier.ipynb`

The primary notebook walks through the complete spam classification pipeline:

| Section | Description |
|---------|-------------|
| 0. Setup | Imports and library loading |
| 1. Data Loading | Parsing `.eml` files into a structured DataFrame |
| 2. EDA | Exploring shapes, missing values, label distributions |
| 3. Feature Engineering | Combining subject + body into a unified `text` field |
| 4. Text Vectorization | TF-IDF with 5,000 features |
| 5. Train/Test Split | Stratified 70/30 split |
| 6. Decision Tree | Baseline classifier + cross-validation |
| 7. Hyperparameter Tuning | GridSearchCV over depth, min_samples_split, min_samples_leaf |
| 8. Bagging | BaggingClassifier with 100 estimators |
| 9. Random Forest | RandomForestClassifier + GridSearchCV |
| 10. Model Comparison | MSPE comparison across all three models |

### `classification_tree.ipynb`

A structured case study following the course PDF guidelines — applies tree-based methods to the Mashable online news popularity dataset, predicting whether a news article will go "viral".

---

## Dataset

The spam dataset consists of pre-labeled `.eml` email files:

| Split | Ham (0) | Spam (1) | Total |
|-------|---------|----------|-------|
| Training | 4,321 | 6 | 4,327 |
| Testing | 4,288 | 4 | 4,292 |

> **Note:** The dataset is highly imbalanced. Accuracy alone is misleading — we use precision, recall, and F1-score for evaluation.

**Features extracted from each email:**
- `filename` — email file name
- `subject` — email subject line
- `sender` — sender address
- `body` — email body text
- `spam_status` — raw X-Spam-Status header
- `label` — binary target (1 = spam, 0 = ham)

---

## Methods

### 1. TF-IDF Vectorization
Emails are converted to numerical features using **Term Frequency–Inverse Document Frequency (TF-IDF)**. This captures how important a word is to a specific email relative to the full corpus.

### 2. Decision Tree (Baseline)
A single `DecisionTreeClassifier` trained on TF-IDF features. Interpretable but prone to overfitting without pruning.

### 3. Bagging
`BaggingClassifier` with 100 decision tree estimators. Each tree trains on a random bootstrap sample, reducing variance through averaging.

### 4. Random Forest
`RandomForestClassifier` extends bagging by also randomly subsampling features at each split, further decorrelating the trees. Tuned with `GridSearchCV`.

---

## Results Summary

| Model | Accuracy | MSPE |
|-------|----------|------|
| Decision Tree (tuned) | ~96% | 0.0381 |
| Bagging | ~97% | 0.0266 |
| Random Forest | ~97% | 0.0254 |

> **Random Forest achieves the best performance**, with the lowest Mean Squared Prediction Error (MSPE).

---

## Key Concepts

- **Overfitting vs. Underfitting** — controlling tree depth and minimum samples per leaf
- **Cross-Validation** — 5-fold CV to estimate out-of-sample error
- **Precision vs. Recall** — critical tradeoff in imbalanced spam detection
- **Feature Importance** — which words best distinguish spam from ham
- **MSPE** — Mean Squared Prediction Error for numeric model comparison

---

## How to Run

1. Clone the repository:
   ```bash
      git clone https://github.com/plinio9302/Decision_trees_classifier.git
         cd Decision_trees_classifier
            ```

            2. Install dependencies:
               ```bash
                  pip install pandas scikit-learn matplotlib seaborn numpy
                     ```

                     3. Place your `.eml` dataset in `~/Downloads/TRAINING/` and `~/Downloads/TESTING/`  
                        *(or update the paths in Cell 1 of the notebook)*

                        4. Open the notebook:
                           ```bash
                              jupyter notebook "Spam Classifier.ipynb"
                                 ```

                                 ---

                                 ## References

                                 - [Scikit-learn Documentation](https://scikit-learn.org)
                                 - Ensemble Methods Case Study PDF (included in this repo)
                                 - James D. Wilson, University of San Francisco — MATH 373 course materials
