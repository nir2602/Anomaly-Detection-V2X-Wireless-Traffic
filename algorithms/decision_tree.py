from sklearn.tree import DecisionTreeClassifier
#from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os
#from util.process_dataset import get_dataset
import time

from util.plotting import plot, feature_importance


class DecisionTreeCLS:
    classifier: DecisionTreeClassifier

    def __init__(self):
        self.classifier = DecisionTreeClassifier(
            criterion="gini",
            random_state=42,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced"
        )

    # def get_training_split(self, X, y, test_size=0.3, random_state=42):
    #     X_train, _, y_train, _ = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y
    #     )
    #     return X_train, y_train

    # def get_testing_split(self, X, y, test_size=0.3, random_state=42):
    #     _, X_test, _, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y
    #     )
    #     return X_test, y_test

    def train_decision_tree(self, X_train, y_train):
        start = time.time()
        self.classifier.fit(X_train, y_train)
        end = time.time()

        print("Decision Tree training complete")
        print(f"Training time: {end - start:.2f} seconds")
        feature_importance(self.classifier.feature_importances_, 
                           X_train.columns, model_name="decision_tree")

    def save_model(self, model_path="models/decision_tree_model.joblib"):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.classifier, model_path)
        print(f"Decision Tree model saved to {model_path}")

    def load_model(self, model_path="models/decision_tree_model.joblib"):
        self.classifier = joblib.load(model_path)
        print(f"Decision Tree model loaded from {model_path}")

    def predict(self, X_test):
        return self.classifier.predict(X_test)

    def evaluate(self, y_pred, y_test):
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        plot(y_pred, y_test, model_name="decision_tree")
        


# if __name__ == "__main__":
#     dt_cls = DecisionTreeCLS()
#     dt_cls.train_decision_tree()

#     dataset = get_dataset()
#     if dataset is not None:
#         X, y = dataset
#         X_test, y_test = dt_cls.get_testing_split(X, y)
#         y_pred = dt_cls.predict(X_test)
#         dt_cls.evaluate(y_pred, y_test)