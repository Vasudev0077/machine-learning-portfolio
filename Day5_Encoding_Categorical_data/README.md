# Categorical Data Encoding and Preprocessing Pipelines

This repository features an educational guide on **Categorical Data Encoding**, demonstrating how to convert unstructured text categories into structured numerical values for machine learning. Real-world machine learning models cannot process text fields directly; this project explores proper mathematical encodings that preserve underlying feature properties without causing dimensionality or collinearity traps.

---

## Technical Concepts & Pipelines

The project demonstrates encoding and cleanup strategies across two distinct data environments using Python's data science ecosystem (`Pandas`, `NumPy`, and `Scikit-Learn`):

### 1. Customer Profiling Pipeline (`customer.csv`)
* **Objective:** Clean and encode a customer matrix containing text behaviors and educational backgrounds to predict a binary purchase flag (`purchased`).
* **Categorical Layout Breakdown:**
  * **Ordinal Encoding (`review` & `education`):** Replaces ordered text slots with rank-preserving integer steps ($0, 1, 2$) based on their natural logical hierarchies:
    * `review`: $\text{Poor} \rightarrow 0$, $\text{Average} \rightarrow 1$, $\text{Good} \rightarrow 2$
    * `education`: $\text{School} \rightarrow 0$, $\text{UG} \rightarrow 1$, $\text{PG} \rightarrow 2$
  * **Label Encoding (`purchased`):** Uses Scikit-Learn's `LabelEncoder` to cast the dependent text target vector into clean binary indicators ($\text{No} \rightarrow 0$, $\text{Yes} \rightarrow 1$).

### 2. Vehicle Valuation Pipeline (`cars.csv`)
* **Objective:** Encode high-cardinality nominal features to prepare data for vehicle pricing models.
* **Categorical Layout Breakdown:**
  * **One-Hot Encoding (`fuel` & `owner`):** Builds discrete binary columns (0 or 1) for each independent state to prevent distance distortion in non-parametric models.
  * **Dummy Variable Trap Prevention:** Applies the $K-1$ standard transformation matrix (`drop='first'`) to discard one redundant column per categorical factor. This eliminates perfect multicollinearity, protecting linear estimators from matrix calculation errors.
  * **High-Cardinality Handling (`brand`):** Reduces feature inflation by applying an operational threshold count ($n \le 100$). Low-frequency manufacturers are collapsed into a unified `'uncommon'` label before encoding, keeping the feature space compact.

### 3. Unified Column Transformer Preprocessing (`covid_toy.csv`)
* **Objective:** Consolidate data cleaning and multi-strategy encoding into a single, clean pipeline using Scikit-Learn's `ColumnTransformer`.
* **Architectural Flow:**
  * **Missing Value Imputation:** Fills out structural missing data arrays in continuous features (`fever`) using a `SimpleImputer` mean-replacement strategy.
  * **Simultaneous Multi-Encoding:** Executes an `OrdinalEncoder` for ordered categorical splits (`cough`) and a `OneHotEncoder` for non-ordered fields (`gender`, `city`) in parallel within a single step.
  * **Feature Preservation (`remainder='passthrough'`):** Passes unaffected numerical parameters (such as `age`) cleanly through the pipeline without altering their structure.

---

## Structural Summary of Encoding Strategies

| Encoding Strategy | Category Context | Data Property Preserved | Common Risk Factors | Primary Model Classes |
| :--- | :--- | :--- | :--- | :--- |
| **Ordinal Encoding** | Ordinal (S, M, L) | Stepwise rank and natural mathematical scale | Arbitrary numerical assumptions | Tree-based models (XGBoost, Random Forests) |
| **One-Hot Encoding** | Nominal (Low Cardinality) | Absolute equity and distance independence | Feature space explosion, Dummy Trap | Linear Regression, Logistic Regression, KNN |
| **Column Transformer** | Mixed Data Tables | Unified workflow structure and pipeline security | Column order mismatches between splits | Production ML Pipelines |

---

## Technical Stack & Dependencies

To execute this workspace and reproduce the numerical transformations, initialize an environment with the following dependencies:

```bash
pip install numpy pandas scikit-learn
```

* **NumPy:** For managing low-level matrix configurations and array stacking operations.
* **Pandas:** For loading datasets, performing frequency value inspections, and doing fast column substitutions.
* **Scikit-Learn:** For implementing train-test splits and deploying production preprocessing estimators (`OrdinalEncoder`, `OneHotEncoder`, `LabelEncoder`, `SimpleImputer`, `ColumnTransformer`).
