# Exploratory Data Analysis (EDA) Workflow

A step-by-step Python project demonstrating the fundamentals of **Exploratory Data Analysis (EDA)**. This repository serves as a guide for cleaning, understanding, and visualizing data using popular Python libraries before building machine learning models.

---

## Project Overview

The project is divided into three main phases:

### 1. Data Understanding & Auditing
Initial structural checks on the dataset to understand its shape, data types, and overall quality:
* Checking data dimensions (`shape`) and data types (`info`).
* Viewing data samples (`head`, `sample`) and mathematical summaries (`describe`).
* Detecting missing values (`isnull`) and duplicate entries (`duplicated`).
* Checking simple feature correlations (`corr`).

### 2. Univariate Analysis
Analyzing variables one at a time to understand their individual distributions:
* **Categorical Data:** Visualized using **Countplots** (frequency bars) and **Pie Charts** (percentage distributions).
* **Numerical Data:** Visualized using **Histograms** and **Distplots** (data spread/skewness) and **Boxplots** (identifying outliers).

### 3. Bivariate & Multivariate Analysis
Analyzing multiple variables together to uncover hidden relationships and patterns across different datasets:
* **Scatterplots:** Continuous trends across up to 5 dimensions (using color, size, and markers).
* **Bar & Box Plots:** Comparing numerical values across different categories.
* **Distplots (KDE):** Comparing the distribution/probability density of groups.
* **Heatmaps & Clustermaps:** Visualizing cross-tabulated categorical data through color grids and hierarchical clustering.
* **Pairplots:** An all-in-one matrix view showing pairwise relationships for entire datasets.

---

## Getting Started

### Prerequisites
Make sure you have Python installed, along with the required data science libraries:

```bash
pip install pandas seaborn matplotlib
```

### How to Run
1. Clone this repository to your local machine.
2. Ensure you have the required `.csv` datasets (like `train.csv` for the Titanic data) in the same directory.
3. Open the Jupyter Notebook file and run the cells sequentially to see the step-by-step analysis and visualizations.

---

## Datasets Used
* **Titanic Dataset:** For structural auditing, univariate analysis, and basic categorical relationships.
* **Tips Dataset:** For multi-dimensional scatterplots.
* **Flights Dataset:** For time-series lineplots and pivot-table heatmaps/clustermaps.
* **Iris Dataset:** For pairwise scatterplot matrices.
