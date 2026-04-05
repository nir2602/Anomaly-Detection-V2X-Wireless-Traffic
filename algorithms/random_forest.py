# Random Forest Implementation for classifying benign and DoS traffic

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import time

class RandomForestCLS:
    
    classifier: RandomForestClassifier
    
    def __init__(self, n_estimators=100, random_state=42):
        self.classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
            
    
    # def get_training_split(self, X, y, test_size=0.3, random_state=42):
    #     X_train, _, y_train, _ = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify = y)
    #     return X_train, y_train

    # def get_testing_split(self, X, y, test_size=0.3, random_state=42):
    #     _, X_test, _, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify = y)
    #     return X_test, y_test

    def train_random_forest(self, X_train, y_train):
        # time the dataset training
        start_time = time.time()
        
        # train dataset 
        self.classifier.fit(X_train, y_train)
        
        end_time = time.time()
        
        print("Training complete.")
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
    rf_cls = RandomForestCLS()
    rf_cls.train_random_forest()