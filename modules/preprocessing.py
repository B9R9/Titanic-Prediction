from . import Transformers

def align_columns(train_df, test_df):
    """
    Aligns the columns of train and test DataFrames by adding missing columns with False values.

    Parameters:
    train_df (DataFrame): DataFrame containing the training data.
    test_df (DataFrame): DataFrame containing the test data.

    Returns:
    DataFrame: Aligned train DataFrame.
    DataFrame: Aligned test DataFrame.
    """
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    
    # Find columns missing in each DataFrame
    missing_from_train = test_cols - train_cols
    missing_from_test = train_cols - test_cols
    
    # Add missing columns to each DataFrame with False values
    for col in missing_from_train:
        train_df[col] = False
    for col in missing_from_test:
        test_df[col] = False
    
    return train_df, test_df

def preprocess_data(train_data, test_data):
    """
    Preprocesses the train and test data.

    Parameters:
    train_data (DataFrame): DataFrame containing the training data.
    test_data (DataFrame): DataFrame containing the test data.

    Returns:
    DataFrame: Processed features for training.
    Series: Target labels for training.
    DataFrame: Processed features for testing.
    """
    # Separate features and target labels
    X_train = train_data.drop('Survived', axis=1)
    y_train = train_data['Survived']
    X_test = test_data.copy()

    # Initialize transformers for data cleaning and feature engineering
    categorial_clean = Transformers.CategorialClean()
    numerical_clean = Transformers.NumericalClean()
    data_cleaner = Transformers.DataCleaner((categorial_clean, ['Sex', 'Embarked', 'Cabin', 'Name', 'Ticket']),
                                            (numerical_clean, ['Pclass', 'Age', 'Fare', 'SibSp', 'Parch']))
    
    # Clean and transform the data
    X_train = data_cleaner.fit_clean(X_train)
    X_test = data_cleaner.clean(X_test)
    
    transformers = Transformers.DataTransformer((Transformers.CustomOneHotEncoder(), ['Sex', 'Embarked']),
                                                (Transformers.Age_Group_Transformer(), ['Age']),
                                                (Transformers.Deck_encoder(), ['Cabin']),
                                                (Transformers.FamilySize_Transform(), ['SibSp', 'Parch']),
                                                (Transformers.FarePerClassTransformer(), ['Fare', 'Pclass']),
                                                (Transformers.AgeClassTransformer(), ['Age', 'Pclass']),
                                                (Transformers.SexClassTransformer(), ['Sex', 'Pclass']),
                                                (Transformers.FareByClassTransformer(), ['Fare', 'Pclass']),
                                                (Transformers.AverageAgePerDeckTransformer(), ['Deck', 'Age']),
                                                (Transformers.AverageFarePerDeckTransformer(), ['Deck', 'Fare']),
                                                (Transformers.AverageAgePerEmbarkedTransformer(), ['Age', 'Embarked']),
                                                (Transformers.StatusTransformer(), ['Name']),
                                                (Transformers.Ticket_encoder(), ['Ticket']))
    
    X_train = transformers.fit_transform(X_train)
    X_test = transformers.transform(X_test)

    # Align columns between train and test DataFrames
    X_train, X_test = align_columns(X_train, X_test)
    
    # Reorder train DataFrame columns to match test DataFrame
    X_train = X_train[X_test.columns]

    # Drop unnecessary columns
    cols_to_drop = ['Sex', 'Embarked', 'Cabin', 'Name', 'Ticket', 'Deck', 'PrefixeTicket', 'PassengerId']
    X_train = X_train.drop(cols_to_drop, axis=1)
    X_test = X_test.drop(cols_to_drop, axis=1)

    # Remove columns with only one unique value
    for col in X_train.columns:
        if X_train[col].nunique() == 1:
            X_train.drop(col, axis=1, inplace=True)
    for col in X_test.columns:
        if X_test[col].nunique() == 1:
            X_test.drop(col, axis=1, inplace=True)
            
    return X_train, y_train, X_test

