import pandas as pd
from sklearn.model_selection import train_test_split
from modules.preprocessing import preprocess_data
from modules.TitanicModel import TitanicModel

# Set the float display format
pd.set_option('display.float_format', '{:.5f}'.format)
# Set the option to display all rows without truncation
pd.set_option('display.max_rows', None)
# Set the option to display all columns without truncation
pd.set_option('display.max_columns', None)


def main():
    """
    Main function to preprocess data, train a model, make predictions, and save results.
    """
    # Load data as DataFrame
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')

    print("RAW DATA:\n", train_data)

    # Preprocess data
    X_train, y_train, X_predic = preprocess_data(train_data, test_data)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Initialize Model
    model = TitanicModel(100, 7, 1)

    # Train the model
    model.train(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_predic)

    # Evaluate the model
    model.evaluate(X_train, y_train)

    # Save predictions
    model.save_predictions(test_data.PassengerId, predictions, 'submission.csv')


if __name__ == "__main__":
    main()
