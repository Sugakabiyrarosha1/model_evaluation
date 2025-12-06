# Model Evaluation

This folder contains comprehensive tutorials on **model evaluation metrics** for classification problems, covering confusion matrices, precision, recall, F1 score, ROC curves, and practical comparisons between different algorithms.

## 📚 Contents

### 1. `ModelEvaluation.ipynb`
**Fundamental Classification Metrics and Evaluation**

A comprehensive guide to understanding and computing classification evaluation metrics:

- **The Evaluation Flow**:
  1. Train classifier on labeled data
  2. Compare predictions to true labels
  3. Build confusion matrix
  4. Compute derived metrics
  5. Adjust thresholds for trade-offs
  6. Visualize with ROC curves
  7. Interpret results

- **Confusion Matrix**:
  - Four categories: TP, FP, TN, FN
  - Visual representation
  - Foundation for all other metrics

- **Core Metrics** (with formulas):
  - **Accuracy**: $\frac{TP + TN}{TP + FP + TN + FN}$
    - Overall correctness
  - **Precision**: $\frac{TP}{TP + FP}$
    - When we predict positive, how often is it correct?
    - Reduces false alarms
  - **Recall**: $\frac{TP}{TP + FN}$
    - Among actual positives, how many did we catch?
    - Reduces missed cases
  - **False Positive Rate**: $\frac{FP}{FP + TN}$
    - Among actual negatives, how many false alarms?
  - **F1 Score**: $2 \times \frac{Precision \times Recall}{Precision + Recall}$
    - Harmonic mean balancing precision and recall
    - Penalizes imbalance

- **Harmonic Mean Explanation**:
  - Mathematical foundation
  - Why it's used for F1 score
  - Two-value case formula

- **Manual Example**:
  - Student dataset with 10 examples
  - Step-by-step calculation of all metrics
  - Interpretation of results
  - Visual confusion matrix

**Key Learning Outcomes:**
- Understand each metric's purpose and calculation
- Learn to interpret confusion matrices
- Know when to prioritize precision vs recall
- Understand the F1 score as a balanced measure
- Practice manual calculations for deep understanding

### 2. `ModelEvaluation-II.ipynb`
**Comparing Logistic Regression vs Decision Tree on Titanic Dataset**

A practical comparison of two classification algorithms with comprehensive evaluation:

- **Dataset**: Titanic passenger survival
  - Features: Passenger class, sex, age, siblings/spouses, parents/children, fare, embarkation port
  - Target: Survival (binary classification)

- **Data Preparation**:
  - Loading and exploration
  - Feature selection and cleaning
  - Handling missing values
  - Categorical encoding (one-hot encoding for embarked)
  - Train/test split (80/20)

- **Model 1: Logistic Regression**:
  - Training and prediction
  - Performance evaluation
  - Confusion matrix
  - Accuracy, precision, recall, F1 score

- **Model 2: Decision Tree Classifier**:
  - Training and prediction
  - Performance evaluation
  - Comparison with logistic regression

- **Comprehensive Comparison**:
  - Side-by-side metric comparison
  - Visualization of results
  - Interpretation of which model performs better
  - Understanding trade-offs

**Key Learning Outcomes:**
- Compare different classification algorithms
- Apply evaluation metrics to real-world dataset
- Understand model selection criteria
- Practice complete ML workflow from data to evaluation
- Learn to interpret comparative results

## 🛠️ Technologies Used

- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **seaborn**: Statistical visualization and dataset loading
- **matplotlib**: Plotting
- **scikit-learn**:
  - `LogisticRegression`: Linear classification
  - `DecisionTreeClassifier`: Tree-based classification
  - `train_test_split`: Data splitting
  - `confusion_matrix`, `ConfusionMatrixDisplay`: Confusion matrix
  - `accuracy_score`, `precision_score`, `recall_score`, `f1_score`: Metrics
  - `classification_report`: Comprehensive report

## 📋 Prerequisites

- Understanding of classification problems
- Familiarity with confusion matrices
- Basic knowledge of logistic regression and decision trees
- Understanding of train/test splits

## 🚀 Getting Started

1. **Install Required Packages**:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

2. **Run the Notebooks**:
   - Start with `ModelEvaluation.ipynb` for metric fundamentals
   - Follow with `ModelEvaluation-II.ipynb` for practical application

## 📊 Datasets Used

1. **Student Dataset** (ModelEvaluation.ipynb):
   - 10 students with hours studied and attendance
   - Binary classification: Pass/Fail
   - Small dataset for manual calculation practice

2. **Titanic Dataset** (ModelEvaluation-II.ipynb):
   - 891 passengers (714 after cleaning)
   - Features: pclass, sex, age, sibsp, parch, fare, embarked
   - Target: survived (0 or 1)
   - Classic binary classification problem

## 💡 Key Concepts

### Confusion Matrix Components

| | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

### Metric Trade-offs

- **High Precision**: Few false positives (trustworthy positive predictions)
- **High Recall**: Few false negatives (catches most actual positives)
- **F1 Score**: Balances both (harmonic mean)

### When to Prioritize Which Metric?

- **Precision**: When false positives are costly
  - Example: Spam detection (don't mark important emails as spam)
- **Recall**: When false negatives are costly
  - Example: Disease diagnosis (don't miss actual cases)
- **F1 Score**: When both matter equally
  - Example: General classification tasks

## 🎯 Evaluation Workflow

1. **Split Data**: Train/test (and optionally validation)
2. **Train Models**: Fit on training data
3. **Make Predictions**: Predict on test set
4. **Build Confusion Matrix**: Count TP, FP, TN, FN
5. **Calculate Metrics**: Accuracy, precision, recall, F1
6. **Compare Models**: Side-by-side comparison
7. **Interpret Results**: Choose best model for use case

## 📝 Notes

- `ModelEvaluation.ipynb` includes detailed manual calculations
- All formulas are explained with examples
- Visualizations help understand metric relationships
- Real-world dataset (Titanic) provides practical context
- Results are reproducible with fixed random seeds

## 🔗 Related Topics

- **Classification Algorithms**: See `Classification/` folder
- **Logistic Regression**: See `logistic_regression/` folder
- **ROC Curves and AUC**: Advanced evaluation techniques (may be covered in other notebooks)
