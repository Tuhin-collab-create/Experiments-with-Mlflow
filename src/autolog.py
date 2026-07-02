##--------------------- Mlflow auto logging -------------------------##

### Remote server deployment Dagsflow ###

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# Dagshub integration
import dagshub
dagshub.init(repo_owner='Tuhin-collab-create', repo_name='Experiments-with-Mlflow', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Tuhin-collab-create/Experiments-with-Mlflow.mlflow") # Setting tracking url of Dagshub



wine = load_wine()
x = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=42)

# params for RF
max_depth = 2
n_estimators = 22

mlflow.autolog()                            ## Auto log features of mlflow, it will automatically log the parameters, metrics, and model artifacts.
mlflow.set_experiment('YT_MLOPS-Expe2(with auto logging)')  ## Experiment name
with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    
    # Creating a confusion metrics plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    mlflow.log_artifact(__file__)
    # tags
    mlflow.set_tag('Author', 'Tuhin Barai')
    mlflow.set_tag('Project', 'MLOps_Mlflow')
    
    print(f"Accuracy: {accuracy}")
