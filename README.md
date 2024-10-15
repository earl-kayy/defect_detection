# Defective Product Detection Project

## Introduction
The goal of this project was to develop a predictive model to **detect defective products** using the given tabular data. The dataset contained 10,000 records with 12 features, including a label that indicates whether a product is defective or not.

### Project Overview
The project followed these main steps:
1. **Exploratory Data Analysis (EDA)**
2. **Feature Engineering**
3. **Model Development and Hyperparameter Tuning**
4. **Final Stacking Ensemble Model**

## Exploratory Data Analysis (EDA)
During the initial exploratory analysis, I identified a key issue with the dataset: **data imbalance**. The number of non-defective products significantly outnumbered defective ones, which posed a challenge in using metrics like accuracy for evaluation. For instance, even if a model predicted all products as non-defective, it would achieve high accuracy simply due to the imbalance.

To address this, evaluation metrics were shifted to more appropriate ones for imbalanced data:
- **AUC Score**
- **TPR/FNR (True Positive Rate / False Negative Rate)**

### Invalid Data and Outliers
During the EDA process, some **invalid records** that violated logical constraints (e.g., sum of ingredients exceeding the total weight) were identified. Instead of removing these records prematurely, I conducted **outlier analysis** to ensure that adjusting these records wouldn’t distort the data.

### Data Cleaning and Preprocessing
Missing values, particularly in the `wtcd` column, were addressed after determining no strong correlations between `wtcd` and other variables. Given the minimal impact on the dataset, rows with missing `wtcd` values were removed.

Further, I engineered features such as:
- Ratios of ingredients (instead of absolute values)
- Date transformations to extract useful information like seasons and durations between key events (purchase and inspection dates).

## Model Development

### Feature Engineering
Various features were engineered to improve the model’s performance:
- **Ingredient Ratios**: Hypothesized that relative proportions of ingredients (e.g., `A/gram`, `extra/gram`) could provide better insight than absolute values.
- **Date Features**: Date columns were transformed to capture **seasonality** and **durations**. This included calculating the difference between purchase date and inspection date.

### Handling Data Imbalance
As identified during the EDA, the dataset was imbalanced, with non-defective products significantly outnumbering defective ones. To counteract this, I focused on using evaluation metrics like **AUC** rather than accuracy. I also implemented techniques like **stratified K-Fold cross-validation** to ensure the model performed well across both classes.

### Hyperparameter Tuning and Model Selection
I explored several machine learning models and conducted **hyperparameter tuning** using grid search for each model. The primary models explored were:
- **Random Forest**
- **XGBoost**
- **Logistic Regression**

For each model, I performed cross-validation and evaluated the AUC score to assess their effectiveness. While individual models performed reasonably well, combining them led to better results.

## Stacking Ensemble Model
The final approach involved using a **stacking ensemble model**. I used two base models and one meta-model:
- **Base Models**:
  - Random Forest
  - XGBoost
- **Meta Model**:
  - Logistic Regression

The base models were trained on different subsets of the data, and their predictions were combined and fed into the logistic regression meta-model. This approach allowed us to capture the strengths of each individual model while mitigating their weaknesses.

### Stacking Process
1. **Base Models**: Both Random Forest and XGBoost were trained using cross-validation.
2. **Blending**: Predictions from the base models were used to create a new feature set for the meta-model.
3. **Meta Model**: The logistic regression meta-model was then trained on this new feature set.

### Final Evaluation
The final stacking model was evaluated using the **AUC score**, **TPR/FNR (True Positive Rate / False Negative Rate)**, and outperformed individual models in most cases. The use of stacking allowed the model to better capture complex relationships in the data.

### Visualization
#### Individual Model
![Ind._AUC](/image/singlemodel_roc.png)
##### Stacked Model
![Stk.AUC](/image/roc_curve.png)
![Stk.TF](/image/tpr_fnr.png)


## Conclusion
This project involved extensive exploration and feature engineering to address the challenges of data imbalance and feature interactions. After experimenting with various models and hyperparameter tuning, the **stacking ensemble** approach yielded the best performance, combining the strengths of **Random Forest**, **XGBoost**, and **Logistic Regression**. This comprehensive approach allowed us to build a more robust predictive model for detecting defective products.

