import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 5, 10, 20, 30],
}
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)


mlflow.set_experiment('brest_cancer_gridsearch_v4')
with mlflow.start_run(run_name="GridSearch_Run"):
    grid_search.fit(X_train, y_train)
    
    # for storing each result and params
    results = grid_search.cv_results_
    for i in range(len(grid_search.cv_results_['params'])):
        with mlflow.start_run(nested = True) as child:
            mlflow.log_params(grid_search.cv_results_['params'][i])
            mlflow.log_metric("accuracy",grid_search.cv_results_['mean_test_score'][i])
        
    
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_accuracy", grid_search.best_score_)
    
    mlflow.sklearn.log_model(grid_search.best_estimator_, "best_random_forest_model")
    
    mlflow.log_artifact(__file__)
    
    mlflow.set_tag('Author', 'Tuhin Barai')
    
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Score: {grid_search.best_score_}")