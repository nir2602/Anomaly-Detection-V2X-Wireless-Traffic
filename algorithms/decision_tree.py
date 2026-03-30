from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from util.process_dataset import get_dataset
import time

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

    def get_training_split(self, X, y, test_size=0.3, random_state=42):
        X_train, _, y_train, _ = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        return X_train, y_train

    def get_testing_split(self, X, y, test_size=0.3, random_state=42):
        _, X_test, _, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        return X_test, y_test

    def train_decision_tree(self, test_size=0.3):
        dataset = get_dataset()
        if dataset is None:
            print("Failed to load dataset")
            return

        X, y = dataset

        X_train, y_train = self.get_training_split(X, y, test_size=test_size, random_state=42
        )

        start_time = time.time()
        self.classifier.fit(X_train, y_train)
        end_time = time.time()

        print("Decision Tree training complete")
        print(f"Training time: {end_time - start_time:.2f} seconds")

    def predict(self, X_test):
        return self.classifier.predict(X_test)

    def evaluate(self, y_pred, y_test):
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    dt_cls = DecisionTreeCLS()
    dt_cls.train_decision_tree()

    dataset = get_dataset()
    if dataset is not None:
        X, y = dataset
        X_test, y_test = dt_cls.get_testing_split(X, y)
        y_pred = dt_cls.predict(X_test)
        dt_cls.evaluate(y_pred, y_test)