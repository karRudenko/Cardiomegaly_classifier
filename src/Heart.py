from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

data = pd.read_csv("src/task_data.csv")
# some data are string type, so we need to make them float type
for col in data.columns:
    if data[col].dtype == 'object': 
        data[col] = data[col].str.replace(',', '.').astype(float)

X = data[[
   'Heart width', 'Lung width', 'CTR - Cardiothoracic Ratio', 'xx', 'yy', 'xy', 'normalized_diff', 'Inscribed circle radius', 'Polygon Area Ratio', 'Heart perimeter', 'Heart area ', 'Lung area'
]]                          # features
y = data["Cardiomegaly"]    #target column

# training 80%, testing 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler();
X_scaled_train = scaler.fit_transform(X_train)
X_scaled_test = scaler.transform(X_test)


# K-Nearest Neighbors classifier
param_grid = {
    "model__n_neighbors": [6, 7, 8],  
    "model__weights": ["uniform"],  
    "model__metric": ["manhattan"], 
}
rskf = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=100,
    random_state=None
)
pipe_knn = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", KNeighborsClassifier())
])

grid_search = GridSearchCV(
    estimator=pipe_knn,       
    param_grid=param_grid,      
    scoring="accuracy",         
    cv=rskf,                    
    verbose=1,                  
    n_jobs=-1                    
)
grid_search.fit(X_train, y_train)

print(f"Best accuracy (averaged CV): {grid_search.best_score_:.4f}")
# Cheking which method is the best
the_best = grid_search.best_score_, "K-Nearest Neighbors"


# Decision Tree
clf_tree = DecisionTreeClassifier(
    max_depth=7, 
    criterion="log_loss",
    min_samples_split=7,
    min_samples_leaf=4,
    class_weight=None
)

clf_tree.fit(X_train, y_train)

cv_score = np.round(cross_val_score(clf_tree, X_train, y_train), 2)

print(f"\nCross-validation mean score: {np.mean(cv_score):.3}")
if(the_best[0] < np.mean(cv_score)):
    the_best = np.mean(cv_score), "Decision Tree"


# Support Vector Machine (SVM)
pipe_svc = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", SVC(
        kernel="rbf",
        C=3,
        gamma="scale",
        class_weight=None
    ))
])

pipe_svc.fit(X_train, y_train)

cv_score = np.round(cross_val_score(pipe_svc, X_train, y_train), 2)
print(f"\nCross-validation mean score: {np.mean(cv_score):.3}")
if(the_best[0] < np.mean(cv_score)):
    the_best = np.mean(cv_score), "SVM"


# Logistic Regression
pipe_log = Pipeline(steps=[
    ("scaler", StandardScaler()),       
    ("model", LogisticRegression(        
        C=1,                            
        penalty="l1",                      
        solver="liblinear",                
        max_iter=1000,                     
        class_weight=None                  
    ))
])

pipe_log.fit(X_train, y_train)

cv_score = np.round(cross_val_score(pipe_log, X_train, y_train), 2)

print(f"\nCross-validation mean score: {cv_score.mean():.3f}")
if(the_best[0] < np.mean(cv_score)):
    the_best = np.mean(cv_score), "Logistic Regression"


# Random Forest Classifier
clf_rf = RandomForestClassifier(
    max_depth=6,                  
    min_samples_split=6,           
    n_estimators=125,              
    min_samples_leaf=2,             
    max_features='sqrt'            
)

clf_rf.fit(X_train, y_train)

cv_score = np.round(cross_val_score(clf_rf, X_train, y_train), 2)

print(f"\nCross-validation mean score: {np.mean(cv_score):.3f}")
if(the_best[0] < np.mean(cv_score)):
    the_best = np.mean(cv_score), "Random Forest Classifier"

print(f"the highest accuracy: {the_best[0]*100:.2f}%, the method: {the_best[1]}")


# Model on the Test Dataset
pipe_knn.fit(X_train, y_train) # haven't done it yet
y_pred_knn = pipe_knn.predict(X_test)
y_pred_svc = pipe_svc.predict(X_test)
y_pred_log = pipe_log.predict(X_test)
y_pred_tree = clf_tree.predict(X_test)
y_pred_rf = clf_rf.predict(X_test)

# Model evaluation: calculate accuracy for each model separately
acc_knn = accuracy_score(y_test, y_pred_knn)
acc_svc = accuracy_score(y_test, y_pred_svc)
acc_log = accuracy_score(y_test, y_pred_log)
acc_tree = accuracy_score(y_test, y_pred_tree)
acc_rf = accuracy_score(y_test, y_pred_rf)

print(f"Accuracy on test set:")
print(f"- Accuracy of KNN Classifier model on test dataset: {acc_knn:.4f}")
print(f"- Accuracy of SVC model on test dataset: {acc_svc:.4f}")
print(f"- Accuracy of Logistic Regression model on test dataset: {acc_log:.4f}")
print(f"- Accuracy of Decision Tree model on test dataset: {acc_tree:.4f}")
print(f"- Accuracy of Random Forest model on test dataset: {acc_rf:.4f}")