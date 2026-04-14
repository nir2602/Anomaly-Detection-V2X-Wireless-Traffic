from algorithms.random_forest import RandomForestCLS
from algorithms.decision_tree import DecisionTreeCLS
from algorithms.svm import SVMCLS
from sklearn.model_selection import train_test_split
from util.process_dataset import get_dataset
import os


def main():
    print("Which algorithm do you want to run?")
    print("(1) Random Forest")
    print("(2) SVM")
    print("(3) Decision Trees")
    while True:
        try:
            choice = int(input())
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue
        except KeyboardInterrupt:
            print("\nExiting program.")
            exit(0)
            
        dataset = get_dataset()
        if dataset is None:
            print("Failed to load dataset")
            return
        
        X, y = dataset

        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 42, stratify = y)

        print("Use saved model for inference if available? ([Y]/n)")
        use_saved_model = input().strip().lower() != "n"
        if use_saved_model:
            print("Using saved model if available. If no saved model is found, a new model will be trained.")
        else:
            print("Training new model.")
            
        if choice == 1:
            rf_cls = RandomForestCLS()
            rf_model_path = "models/random_forest_model.joblib"
            if use_saved_model and os.path.exists(rf_model_path):
                rf_cls.load_model(rf_model_path)
            else:
                rf_cls.train_random_forest(X_train, y_train)
                rf_cls.save_model(rf_model_path)
            y_pred = rf_cls.predict(X_test)
            rf_cls.evaluate(y_pred, y_test)
            break
        elif choice == 2:
            svm_cls = SVMCLS()
            svm_model_path = "models/svm_model.joblib"
            if use_saved_model and os.path.exists(svm_model_path):
                svm_cls.load_model(svm_model_path)
            else:
                svm_cls.train_svm(X_train, y_train)
                svm_cls.save_model(svm_model_path)
            y_pred = svm_cls.predict(X_test)
            svm_cls.evaluate(y_pred, y_test)
            break 
        elif choice == 3:
            dt_cls = DecisionTreeCLS()
            dt_model_path = "models/decision_tree_model.joblib"
            if use_saved_model and os.path.exists(dt_model_path):
                dt_cls.load_model(dt_model_path)
            else:
                dt_cls.train_decision_tree(X_train, y_train)
                dt_cls.save_model(dt_model_path)
            y_pred = dt_cls.predict(X_test)
            dt_cls.evaluate(y_pred, y_test)
            break
                
        else:
            print("Invalid choice. Please enter a number between 1 and 3.")


if __name__ == "__main__":
    main()