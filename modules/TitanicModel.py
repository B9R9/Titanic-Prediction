from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score
import pandas as pd

class TitanicModel:
    """
    A class representing the Titanic survival prediction model using Random Forest Classifier.
    """

    def __init__(self, estimator, depth, rs):
        """
        Initializes the TitanicModel with the specified parameters.

        Parameters:
        estimator (int): The number of trees in the forest.
        depth (int): The maximum depth of the trees.
        rs (int): Random state for reproducibility.
        """
        self.model = RandomForestClassifier(n_estimators=estimator, max_depth=depth, random_state=rs)
        self.trained = False

    def train(self, X_train, y_train):
        """
        Trains the model using the provided training data.

        Parameters:
        X_train (DataFrame): The features of the training data.
        y_train (Series): The target labels of the training data.
        """
        self.model.fit(X_train, y_train)
        self.trained = True
    
    def predict(self, X_test):
        """
        Predicts the target labels for the given features.

        Parameters:
        X_test (DataFrame): The features for which predictions are to be made.

        Returns:
        array: Predicted target labels.
        """
        if not self.trained:
            raise ValueError("Model must be trained first.")
        return self.model.predict(X_test)

    def evaluate(self, X, y):
        """
        Evaluates the model's performance using the provided data.

        Parameters:
        X (DataFrame): The features for evaluation.
        y (Series): The target labels for evaluation.
        """
        if not self.trained:
            raise ValueError("Model must be trained first.")
        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print("Accuracy:", accuracy)
        print(classification_report(y, y_pred))

        nb_samples_class_0 = (y == 0).sum()
        nb_samples_class_1 = (y == 1).sum()
        print("Nombre d'échantillons de la classe 0:", nb_samples_class_0)
        print("Nombre d'échantillons de la classe 1:", nb_samples_class_1)
        class_ratio = nb_samples_class_1 / nb_samples_class_0
        print("Ratio entre les classes (1/0):", class_ratio)

        y_pred_proba = self.model.predict_proba(X)[:, 1]
        roc_auc = roc_auc_score(y, y_pred_proba)
        print("Aire sous la courbe ROC (ROC AUC):", roc_auc)

        scores = cross_val_score(self.model, X, y, cv=10)
        print("Cross-validation scores:", scores)
        print("Mean accuracy:", scores.mean())

    def save_predictions(self, id, predictions, filename):
        """
        Saves the predictions to a CSV file.

        Parameters:
        id (array): The passenger IDs.
        predictions (array): The predicted survival outcomes.
        filename (str): The name of the CSV file to save the predictions.
        """
        predictions_df = pd.DataFrame({'PassengerId': id, 'Survived': predictions})
        predictions_df.to_csv(filename, index=False)

