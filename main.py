from algorithms.random_forest import RandomForestCLS
from algorithms.decision_tree import DecisionTreeCLS
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
            
        if choice == 1:
            rf_cls = RandomForestCLS()
            rf_cls.train_random_forest()
            dataset = get_dataset()
            if dataset is not None:
                x, y = dataset
                x_test, y_test = rf_cls.get_testing_split(x, y)
                y_pred = rf_cls.predict(x_test)
                rf_cls.evaluate(y_pred, y_test)
            break
        elif choice == 2:
            print("SVM not implemented yet.")
            break
        elif choice == 3:
            dt_cls = DecisionTreeCLS()
            dt_cls.train_decision_tree()
            dataset = get_dataset()
            if dataset is not None:
                x, y = dataset
                x_test, y_test = dt_cls.get_testing_split(x, y)
                y_pred = dt_cls.predict(x_test)
                dt_cls.evaluate(y_pred, y_test)
            break
                
        else:
            print("Invalid choice. Please enter a number between 1 and 3.")


if __name__ == "__main__":
    main()