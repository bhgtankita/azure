import argparse
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import os
import pandas as pd
import mlflow

def select_first_file(path):
    files = os.listdir(path)
    return os.path.join(path, files[0])

mlflow.start_run()
mlflow.sklearn.autolog()

os.makedirs("./outputs", exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="path to train data")
    parser.add_argument("--test_data", type=str, help="path to test data")
    parser.add_argument("--n_estimators", required=False, default=100, type=int)
    parser.add_argument("--learning_rate", required=False, default=0.1, type=float)
    parser.add_argument("--registered_model_name", type=str, help="model name")
    parser.add_argument("--model", type=str, help="path to model file")
    args = parser.parse_args()

    train_df = pd.read_csv(select_first_file(args.train_data))
    y_train = train_df.pop("default payment next month")
    X_train = train_df.values
    
    test_df = pd.read_csv(select_first_file(args.test_data))
    y_test = test_df.pop("default payment next month")
    X_test = test_df.values

    print(f"Training with data of shape {X_train.shape}")

    clf = GradientBoostingClassifier(
        n_estimators=args.n_estimators, 
        learning_rate=args.learning_rate
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    print("Registering the model via MLFlow")

    mlflow.sklearn.log_model(
        sk_model=clf,
        registered_model_name=args.registered_model_name,
        artifact_path=args.registered_model_name
    )

    mlflow.sklearn.save_model(
        sk_model=clf,
        path=os.path.join(args.model, "trained_model")
    )

    mlflow.end_run()

if __name__ == "__main__":
    main()
