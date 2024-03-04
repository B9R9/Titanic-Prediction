import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class DataTransformer:
	def __init__(self, *transformers):
		self.transformers = transformers
	
	def fit(self, X, y=None):
		return self
	
	def transform (self, X):
		X_transformed = X.copy()

		for transformer, columns in self.transformers:
			if columns:
				preprocessed_colums = transformer.fit_transform(X_transformed[columns])
				X_transformed = pd.concat([X_transformed.drop(columns, axis=1), preprocessed_colums], axis=1)
		return X_transformed
	
	def fit_transform(self, X):
		self.fit(X)
		return self.transform(X)

class DataCleaner:
	def __init__(self, *cleaners):
		self.cleaners = cleaners

	def fit(self, X, y=None):
		return self
	
	def clean(self, X):
		X_transformed = X.copy()

		for cleaner, columns in self.cleaners:
			if columns:
				cleaned_columns = cleaner.fit_clean(X_transformed[columns])
				X_transformed = pd.concat([X_transformed.drop(columns, axis=1), cleaned_columns], axis=1)
		return X_transformed

	def fit_clean(self, X):
		self.fit(X)
		return self.clean(X)
	
class CategorialClean(BaseEstimator, TransformerMixin):
	def __init__(self):
		pass

	def fit(self, X, y=None):
		return self

	def clean(self, X):
		X_transformed = X.copy()

		for column in X_transformed:
			if X_transformed[column].isnull().any():
				X_transformed[column] = X_transformed[column].fillna("Unknow")
		return X_transformed
	
	def fit_clean(self, X):
		self.fit(X)
		return self.clean(X)
	
class NumericalClean(BaseEstimator, TransformerMixin):
	def __init__(self):
		pass

	def fit(self, X, y=None):
		return self

	def clean(self, X):
		X_transformed = X.copy()

		for column in X_transformed:
			if X_transformed[column].isnull().any():
				X_transformed[column] = X_transformed[column].fillna(X_transformed[column].mean())		
		return X_transformed
	
	def fit_clean(self, X):
		self.fit(X)
		return self.clean(X)

class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
	def __init__(self):
		pass

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		X_transformed = X.copy()

		for column in X_transformed:
			data_encoded = pd.get_dummies(X_transformed[column], prefix=column)
			X_transformed = pd.concat([X_transformed, data_encoded], axis=1)
		
		return X_transformed
	
	def fit_transform(self, X, y=None):
		self.fit(X)
		return self.transform(X)

class Age_Group_Transformer(BaseEstimator, TransformerMixin):
	def __init__(self):
		self.max_age = float('-inf')

	def fit(self, X, y=None):
		max_age_in_X = X['Age'].max()
		if not pd.isnull(max_age_in_X):
			self.max_age = max(max_age_in_X, self.max_age)
		return self

	def transform(self, X):
		X_transformed = X.copy()

		X_transformed['Age_Group'] = pd.cut(X_transformed['Age'], bins=range(0, 101, 5), right=False, labels=False)
		age_encoded = pd.get_dummies(X_transformed['Age_Group'], prefix='Age')
		age_encoded.columns = [f'Age_{i*5}-{(i+1)*5}' for i in range(len(age_encoded.columns))]

		X_transformed = pd.concat([X_transformed, age_encoded], axis=1)

		return X_transformed
	
	def fit_transform(self, X, y=None):
		self.fit(X)
		return self.transform(X)

class Deck_encoder(BaseEstimator, TransformerMixin):
	def __init__(self):
		pass
	def fit(self, X, y=None):
		return self

	def transform(self, X):
		X_transformed = X.copy()
		
		X_transformed['Deck'] = X_transformed['Cabin'].str.slice(0,1)
		deck_encoded = pd.get_dummies(X_transformed['Deck'], prefix='Deck')
		X_transformed = pd.concat([X_transformed, deck_encoded], axis=1)

		return X_transformed
	
	def fit_transform(self, X, y=None):
		self.fit(X)
		return self.transform(X)

class Ticket_encoder(BaseEstimator, TransformerMixin):
	def __init__(self):
		pass
	def fit(self, X, y=None):
		return self

	def transform(self, X):
		X_transformed = X.copy()
		
		prefixe = []
		ticket_number = []

		for ticket in X_transformed['Ticket']:
			if ticket[0].isalpha(): 
				split_ticket = ticket.split(" ", 1)
				if len(split_ticket) == 2:
					prefixe.append(split_ticket[0])
				else:
					prefixe.append(split_ticket[0])
			else:
				prefixe.append("")

		X_transformed['PrefixeTicket'] = prefixe
		prefixeEncoded = pd.get_dummies(X_transformed['PrefixeTicket'], prefix = "PrefixeTicket")
		X_transformed = pd.concat([X_transformed, prefixeEncoded], axis=1) 

		return X_transformed
	
	def fit_transform(self, X, y=None):
		self.fit(X)
		return self.transform(X)
	
class FamilySize_Transform(BaseEstimator, TransformerMixin):
	def __init__(self):
		pass
	def fit(self, X, y=None):
		return self

	def transform(self, X):
		X_transformed = X.copy()

		X_transformed['Family_Size'] = X_transformed['SibSp'] + X_transformed['Parch']

		return X_transformed
	
	def fit_transform(self, X, y=None):
		self.fit(X)
		return self.transform(X)

class FarePerClassTransformer(BaseEstimator, TransformerMixin):
	def __init__(self):
		pass

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		X_transformed = X.copy()

		X_transformed['Fare_Per_Class'] = X_transformed['Fare'] / X_transformed['Pclass']
		return X_transformed
	
	def fit_transform(self, X, y=None):
		self.fit(X)
		return self.transform(X)

class AgeClassTransformer(BaseEstimator, TransformerMixin):
	def __init__(self):
		pass
	def fit(self, X, y=None):
		return self

	def transform(self, X):
		X_transformed = X.copy()

		X_transformed['Age_class'] = X_transformed['Age'] * X_transformed['Pclass']
		return X_transformed
	
	def fit_transform(self, X, y=None):
		self.fit(X)
		return self.transform(X)

class SexClassTransformer(BaseEstimator, TransformerMixin):
	def __init__(self):
		pass
	def fit(self, X, y=None):
		return self

	def transform(self, X):
		X_transformed = X.copy()

		X_transformed['Female_Class_1'] = ((X_transformed['Sex'] == 'female') & (X_transformed['Pclass'] == 1)).astype(int)
		X_transformed['Female_Class_2'] = ((X_transformed['Sex'] == 'female') & (X_transformed['Pclass'] == 2)).astype(int)
		X_transformed['Female_Class_3'] = ((X_transformed['Sex'] == 'female') & (X_transformed['Pclass'] == 3)).astype(int)
		X_transformed['male_Class_1'] = ((X_transformed['Sex'] == 'male') & (X_transformed['Pclass'] == 1)).astype(int)
		X_transformed['male_Class_2'] = ((X_transformed['Sex'] == 'male') & (X_transformed['Pclass'] == 2)).astype(int)
		X_transformed['male_Class_3'] = ((X_transformed['Sex'] == 'male') & (X_transformed['Pclass'] == 3)).astype(int)

		return X_transformed
	
	def fit_transform(self, X, y=None):
		self.fit(X)
		return self.transform(X)

class FareByClassTransformer(BaseEstimator, TransformerMixin):
	def __init__(self):
		pass
	def fit(self, X, y=None):
		return self

	def transform(self, X):
		X_transformed = X.copy()

		fare_by_class = X_transformed.groupby('Pclass')['Fare'].mean().reset_index()
		fare_by_class.columns = ['Pclass', 'Average_Fare_By_Class']
		X_transformed = X_transformed.merge(fare_by_class, on='Pclass', how='left')
		return X_transformed

	def fit_transform(self, X, y=None):
		self.fit(X)
		return self.transform(X)

class AverageAgePerDeckTransformer(BaseEstimator, TransformerMixin):
	def __init__(self):
		pass
	def fit(self, X, y=None):
		return self

	def transform(self, X):
		X_transformed = X.copy()

		average_age_per_deck = X_transformed.groupby('Deck')['Age'].mean().reset_index()
		average_age_per_deck.columns = ['Deck', 'Average_Age_Per_Deck']
		X_transformed = X_transformed.merge(average_age_per_deck, on='Deck', how='left')
		return X_transformed

	def fit_transform(self, X, y=None):
		self.fit(X)
		return self.transform(X)

class AverageFarePerDeckTransformer(BaseEstimator, TransformerMixin):
	def __init__(self):
		pass
	def fit(self, X, y=None):
		return self

	def transform(self, X):
		X_transformed = X.copy()

		average_age_Fare_deck = X_transformed.groupby('Deck')['Fare'].mean().reset_index()
		average_age_Fare_deck.columns = ['Deck', 'Average_Age_Fare_Deck']
		X_transformed = X_transformed.merge(average_age_Fare_deck, on='Deck', how='left')
		return X_transformed

	def fit_transform(self, X, y=None):
		self.fit(X)
		return self.transform(X)

class AverageAgePerEmbarkedTransformer(BaseEstimator, TransformerMixin):
	def __init__(self):
		pass

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		X_transformed = X.copy()

		average_age_per_embarked = X_transformed.groupby('Embarked')['Age'].mean().reset_index()
		average_age_per_embarked.columns = ['Embarked', 'Average_Age_Per_embarked']
		X_transformed = X_transformed.merge(average_age_per_embarked, on='Embarked', how='left')
		return X_transformed

	def fit_transform(self, X, y=None):
		self.fit(X)
		return self.transform(X)

class StatusTransformer(BaseEstimator, TransformerMixin):
	def __init__(self):
		pass

	def fit(self, X, y=None):
		return self
		
	def transform(self, X):
		X_transformed = X.copy()

        # Identifier si le nom contient 'Mrs', 'Miss' ou 'Mr'
		X_transformed['is_Mrs'] = X_transformed['Name'].str.contains('Mrs').astype(int)
		X_transformed['is_Miss'] = X_transformed['Name'].str.contains('Miss').astype(int)
		X_transformed['is_Mr'] = X_transformed['Name'].str.contains('Mr').astype(int)
		
		return X_transformed

	def fit_transform(self, X, y=None):
		self.fit(X)
		return self.transform(X)