from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import os
import time

from util.plotting import plot, feature_importance


class SVMCLS:
    classifier: SVC
    scaler: StandardScaler

    def __init__(self):
        self.scaler = StandardScaler()
        self.classifier = SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            random_state=42
        )

    def train_svm(self, X_train, y_train):
        start = time.time()

        print("Scaling training data...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        print("Scaling complete.")

        print("Starting SVM training...")
        self.classifier.fit(X_train_scaled, y_train)

        end = time.time()

        print("SVM training complete")
        print(f"Training time: {end - start:.2f} seconds")
        # feature_importance(self.classifier.classes_, X_train.columns, model_name="svm")

    def save_model(self, model_path="models/svm_model.joblib"):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model_artifact = {
            "classifier": self.classifier,
            "scaler": self.scaler,
        }
        joblib.dump(model_artifact, model_path)
        print(f"SVM model saved to {model_path}")

    def load_model(self, model_path="models/svm_model.joblib"):
        model_artifact = joblib.load(model_path)
        self.classifier = model_artifact["classifier"]
        self.scaler = model_artifact["scaler"]
        print(f"SVM model loaded from {model_path}")
        

    def predict(self, X_test):
        print("Scaling test data...")
        X_test_scaled = self.scaler.transform(X_test)
        print("Making predictions...")
        return self.classifier.predict(X_test_scaled)

    def evaluate(self, y_pred, y_test):
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        plot(y_pred, y_test, model_name="svm")