# Random Forest Implementation for classifying benign and DoS traffic

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os

from util.plotting import plot, feature_importance

import time

class RandomForestCLS:
    
    classifier: RandomForestClassifier
    
    def __init__(self, n_estimators=100, random_state=42):
        self.classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, class_weight="balanced")
            
    
    # def get_training_split(self, X, y, test_size=0.3, random_state=42):
    #     X_train, _, y_train, _ = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify = y)
    #     return X_train, y_train

    # def get_testing_split(self, X, y, test_size=0.3, random_state=42):
    #     _, X_test, _, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify = y)
    #     return X_test, y_test

    def train_random_forest(self, X_train, y_train):
        # time the dataset training
        print("Training Random Forest...")
        start_time = time.time()
        
        # train dataset 
        self.classifier.fit(X_train, y_train)
        
        end_time = time.time()
        
        print("Training complete.")
        print(f"Training time: {end_time - start_time:.2f} seconds")
        self.feature_importance(X_train)

    def save_model(self, model_path="models/random_forest_model.joblib"):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.classifier, model_path)
        print(f"Random Forest model saved to {model_path}")

    def load_model(self, model_path="models/random_forest_model.joblib"):
        self.classifier = joblib.load(model_path)
        print(f"Random Forest model loaded from {model_path}")
        
    def predict(self, X_test):
        return self.classifier.predict(X_test)
    
    def evaluate(self, y_pred, y_test):
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        self.plot(y_pred, y_test)
        
    def plot(self, y_pred, y_test):
        # plot
        plot(y_pred, y_test, model_name="random_forest")
        
        
    def feature_importance(self, X_train):
        feature_importance(self.classifier.feature_importances_, X_train.columns, model_name="random_forest")

if __name__ == "__main__":
    rf_cls = RandomForestCLS()
    rf_cls.train_random_forest()
