from algorithms.random_forest import RandomForestCLS
from algorithms.decision_tree import DecisionTreeCLS
from algorithms.svm import SVMCLS
from sklearn.model_selection import train_test_split
from util.process_dataset import get_dataset


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

         # ===== TEMPORARY: SVM testing only =====
        USE_SAMPLE = (choice == 2)  # only to test SVM

        if USE_SAMPLE:
            import pandas as pd
            sample_size = 50000

            df = pd.concat([X, y.rename("label_binary")], axis=1)

            df = df.sample(n=sample_size, random_state=42)

            X = df.drop(columns=["label_binary"])
            y = df["label_binary"]

            print(f"Using sample of size: {sample_size}")

        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 42, stratify = y)

            
        if choice == 1:
            rf_cls = RandomForestCLS()
            rf_cls.train_random_forest(X_train, y_train)
            y_pred = rf_cls.predict(X_test)
            rf_cls.evaluate(y_pred, y_test)
            break
        elif choice == 2:
            svm_cls = SVMCLS()
            svm_cls.train_svm(X_train, y_train)
            y_pred = svm_cls.predict(X_test)
            svm_cls.evaluate(y_pred, y_test)
            break 
        elif choice == 3:
            dt_cls = DecisionTreeCLS()
            dt_cls.train_decision_tree(X_train, y_train)
            y_pred = dt_cls.predict(X_test)
            dt_cls.evaluate(y_pred, y_test)
            break
                
        else:
            print("Invalid choice. Please enter a number between 1 and 3.")


if __name__ == "__main__":
    main()