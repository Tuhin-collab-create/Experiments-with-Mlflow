import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

# Dagshub integration
import dagshub
dagshub.init(repo_owner='Tuhin-collab-create', repo_name='Experiments-with-Mlflow', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Tuhin-collab-create/Experiments-with-Mlflow.mlflow")

data  = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(random_state=42)
param_grid = {'n_estimators': [10, 50, 100],'max_depth': [None, 5, 10,20,30]}

grid_search = GridSearchCV(estimator=rf,param_grid=param_grid,cv=5,scoring='accuracy',n_jobs=1,verbose=2)

# Mlflow with Autolog
mlflow.sklearn.autolog()
mlflow.set_experiment('brest_cancer_gridsearch')

with mlflow.start_run():
    grid_search.fit(X_train, y_train) 
    
    train_df = X_train.copy()
    train_df['target'] = y_train
    test_df = X_test.copy()
    test_df['target'] = y_test
    
    test_df = mlflow.data.from_pandas(test_df)
    mlflow.log_input(test_df,'test_data')
    train_df = mlflow.data.from_pandas(train_df)
    mlflow.log_input(train_df,'training_data')
    
    mlflow.log_artifact(__file__)
    
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Score: {grid_search.best_score_}")   