# Cardiomegaly Detection Using Machine Learning
## Project Overview

This project aims to predict Cardiomegaly (enlarged heart) using quantitative measurements extracted from radiographic images.
By leveraging classical machine learning algorithms, the project explores how well heart and lung geometry metrics can be used to automatically identify patients with Cardiomegaly.

The objective is to build, evaluate, and compare multiple classifiers — including KNN, SVC, Logistic Regression, Decision Tree, and Random Forest — to determine which model provides the most reliable predictions.

## Dataset
The dataset used in this project is task_data.csv, containing geometric and statistical measurements extracted from radiographic images.
Each record corresponds to a single patient or image sample, with “Cardiomegaly” as the target variable (1 = present, 0 = absent).

### Example Features:
- Heart width
- Lung width
- CTR - Cardiothoracic Ratio
- Inscribed circle radius
- Heart area
- Lung area
- Polygon Area Ratio
- Heart perimeter
- Statistical moments: xx, yy, xy, etc.

## Workflow
The project follows a complete machine learning pipeline:
#### 1. Data Preprocessing
- Handle numeric and string conversions (replace commas with dots).
- Convert all features to floating-point values.
- Split dataset into train (80%) and test (20%) sets.
- Apply StandardScaler to normalize feature scales.
#### 2. Model Training
- Train and cross-validate the following models:
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVC)
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
#### 3. Hyperparameter Optimization
- Use GridSearchCV and RepeatedStratifiedKFold for reliable model tuning.
#### 4. Model Evaluation
- Evaluate each model on the test set using accuracy as the primary metric.

## Results
| Model                               | Test Accuracy |
| ----------------------------------- | ------------- |
| **K-Nearest Neighbors (KNN)**       | **0.7500**    |
| **Support Vector Classifier (SVC)** | **0.7500**    |
| **Logistic Regression**             | **0.7500**    |
| Decision Tree                       | 0.5000        |
| Random Forest                       | 0.6250        |

### Insights
- **KNN**, **SVC**, and **Logistic Regression** achieved the highest accuracy (75%), showing strong and consistent generalization.
- **Decision Tree** underperformed, suggesting possible overfitting or insufficient model depth.
- **Random Forest** performed moderately well but did not surpass simpler models.

## Conclusions
This analysis demonstrates that classical machine learning models may effectively predict Cardiomegaly from basic radiographic measurements — without the need for deep learning or image processing.
Given the relatively small dataset, regularized and distance-based models (SVC, KNN, Logistic Regression) proved most stable.

## Technologies Used
- Python 3.x
- NumPy, Pandas – data manipulation
- Scikit-learn – model training and evaluation
- Matplotlib / Seaborn – visualization (optional)
- Jupyter Notebook – interactive experimentation
