# Feature Scaling and Machine Learning Model Evaluation

This repository features an educational Jupyter notebook demonstrating the principles of **Feature Engineering**, specifically focusing on **Feature Scaling Techniques** (`StandardScaler` and `MinMaxScaler`) and their performance impact on distance-based vs. tree-based machine learning algorithms.

---

## Workspace Structure

The project evaluates data processing techniques across two core machine learning workflows:

### 1. Social Network Ads Classification (`Social_Network_Ads.csv`)
* **Objective:** Predict user purchasing behavior based on demographic features.
* **Key Steps:** 
  * Categorical Variable Encoding (One-Hot Encoding for gender data).
  * Train-Test Validation Split ($70\%$ training, $30\%$ testing).
  * **Standardization (Z-Score Normalization):** Transforming feature ranges to have a mean ($\mu$) of 0 and a standard deviation ($\sigma$) of 1 using `StandardScaler`.
  * **Algorithm Comparison:** Evaluates how scaling changes the prediction boundaries of a distance/gradient-reliant model (`LogisticRegression`) versus a non-parametric tree-based model (`DecisionTreeClassifier`).

### 2. Wine Quality Clustering & Classification (`wine.csv`)
* **Objective:** Classify wine types based on primary chemical profiling features (`Alcohol` and `Malic Acid`).
* **Key Steps:**
  * Train-Test Validation Split ($70\%$ training, $30\%$ testing).
  * **Min-Max Scaling (Normalization):** Squeezing feature values into a rigid, bounded range between $0$ and $1$ using `MinMaxScaler`.
  * Visualizing scatter plots and Kernel Density Estimation (KDE) curves to verify distribution spreads before and after min-max scaling operations.

---

## Structural Summary of Scaling Methods

| Scaling Strategy | Mathematical Transformation | Fixed Range Constraints | Sensitivity to Outliers | Target Algorithm Class |
| :--- | :--- | :--- | :--- | :--- |
| **Standardization** | $z = \frac{x - \mu}{\sigma}$ | No bounded limits | Moderate | Logistic Regression, KNN, SVM, PCA, Neural Networks |
| **Normalization** | $x' = \frac{x - x_{min}}{x_{max} - x_{min}}$ | Bounded to $[0, 1]$ | High | Algorithms requiring strict bounded inputs (e.g., Image Processing) |

---

## Essential Findings & Architectural Takeaways

1. **Distance-Dependent Models require Feature Scaling:** Without scaling, features with higher raw numerical magnitudes (like `EstimatedSalary`) completely dominate distance metrics, skewing weights. Applying `StandardScaler` improved the baseline accuracy of the `LogisticRegression` model from **86.7% to 88.3%**.
2. **Tree-Based Models are Scale-Invariant:** Algorithms like `DecisionTreeClassifier` evaluate features independently at each node split based on information gain thresholds. Consequently, feature scaling does not improve tree performance; raw and scaled datasets achieved equivalent structural performance (with small stochastic variations due to split randomization).

---

## Technical Stack & Dependencies

To execute this workspace and reproduce the numerical transformations, initialize a Python 3 ecosystem containing the following core libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

* **NumPy & Pandas:** For array transformations, boolean indexing, and data science dataframe manipulations.
* **Matplotlib & Seaborn:** For rendering scattered data distributions and comparing Kernel Density Estimation (KDE) profiles.
* **Scikit-Learn:** For implementing train-test splits, feature scaling preprocessors (`StandardScaler`, `MinMaxScaler`), and model estimators.

---

## How to Run the Notebook
1. Clone this repository to your local workspace environment.
2. Ensure `Social_Network_Ads.csv` and `wine.csv` are placed in the same execution directory as the notebook.
3. Launch Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook
   ```
4. Run the code cells sequentially to view the scaling pipelines, performance evaluations, and distribution plots.
