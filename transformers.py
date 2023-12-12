import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer


class GenderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column='gender'):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        def gender_transform(x):
            return 0 if x == 'Female' else 1

        X[self.column] = X[self.column].apply(gender_transform)
        return X


class ImputeColsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, method):
        self.method = method
        self.impute_values = {}

    def fit(self, X, y=None):
        if self.method == 'mode':
            for col in X:
                mode = X[col].mode()
                self.impute_values[col] = mode
        elif self.method == 'median':
            for col in X:
                median = X[col].median()
                self.impute_values[col] = median
        else:
            raise Exception("Available methods: mode, median")
        return self

    def transform(self, X, y=None):
        for col in X:
            value_to_impute = self.impute_values[col]
            X[col] = X[col].fillna(value_to_impute)
        return X


class SmokingHistoryTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column='smoking_history'):
        self.column = column
        self.diabetes_rates_dict = {}

    def fit(self, X, y=None):
        y = pd.DataFrame(y)
        y.columns = ['target']
        df = pd.concat([X, y], axis=1)
        self.diabetes_rates_dict = df.groupby([self.column])['target'].mean().to_dict()
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        X_copy[self.column] = X_copy[self.column].map(self.diabetes_rates_dict)
        return X_copy
