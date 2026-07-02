import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import matplotlib.pyplot as plt
import seaborn as sns

wine = load_wine()
x = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=42)

# params for RF
max_depth = 2
n_estimators = 22

mlflow.set_tracking_uri('http://127.0.0.1:5000')
with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    mlflow.log_metric('accuracy',accuracy)
    mlflow.log_param('max_depth',max_depth)
    mlflow.log_param('n_estimators',n_estimators)
    print(f"Accuracy: {accuracy}")

    # Creating a confusion metrics plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    # log aritfacts using MLFLOw
    save_path = "confusion_matrix.png"
    plt.savefig(save_path)
    mlflow.log_artifact(save_path)
    mlflow.log_artifact(__file__)
    
    print(f"Accuracy: {accuracy}")
