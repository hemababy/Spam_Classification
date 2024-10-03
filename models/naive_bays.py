from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import RandomizedSearchCV
from common_functions import CommonModelFunctions

class NaiveBayesModel:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.nb = None
        self.best_model = None
        self.common_functions = CommonModelFunctions(random_state)

    # Handle NaN values in the dataset
    def handle_nan_values(self, X):
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
        return X_imputed

    # Define the Naive Bayes model
    def define_model(self):
        self.nb = MultinomialNB()

    # Hyperparameter tuning
    def tune_model(self, X_train, y_train):
        # Set up RandomizedSearchCV for hyperparameter tuning
        # - param_distributions: specifies the range of alpha values to try
        # - n_iter: number of iterations for the search (here, 5 different configurations)
        # - scoring: 'roc_auc' is used as the evaluation metric to optimize the area under the ROC curve
        # - n_jobs: number of parallel jobs (None defaults to 1, use -1 for all processors)
        # - verbose: 3 shows detailed logs during fitting
        # - cv: 5-fold cross-validation to evaluate model performance
        # Handle NaN values in the training data
        X_train = self.handle_nan_values(X_train)
        param_distributions = {
            'alpha': [0.01, 0.1, 0.5, 1.0, 1.5, 2.0]
        }
        random_search = RandomizedSearchCV(estimator=self.nb, param_distributions=param_distributions, n_iter=5,
                                           scoring='roc_auc', cv=5, verbose=3, n_jobs=None)
        random_search.fit(X_train, y_train)
        self.best_model = random_search.best_estimator_
        print("Best Parameters:", random_search.best_params_)

# Example usage
if __name__ == "__main__":
    nb_model = NaiveBayesModel()
    X_train, X_test, y_train, y_test = nb_model.common_functions.load_data('data/emails.mat')
    nb_model.define_model()
    nb_model.tune_model(X_train, y_train)
    # Handle NaN values in the test data
    X_test = nb_model.handle_nan_values(X_test)
    nb_model.common_functions.evaluate_model(nb_model.best_model, X_test, y_test)
