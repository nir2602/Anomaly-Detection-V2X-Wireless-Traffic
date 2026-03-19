from algorithms.random_forest import RandomForestCLS


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
            print("Rundom Forest not implemented yet.")
            break
        elif choice == 2:
            print("SVM not implemented yet.")
            break
        elif choice == 3:
            print("Decision Trees not implemented yet.")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 3.")


if __name__ == "__main__":
    main()