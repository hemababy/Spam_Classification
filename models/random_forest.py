from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from common_functions import CommonModelFunctions

class RandomForestModel:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.rf_model = None
        self.best_model = None
        self.common_functions = CommonModelFunctions(random_state)

    # Define the Random Forest model
    def define_model(self):
        self.rf_model = RandomForestClassifier(random_state=self.random_state)

    # Hyperparameter tuning
    def tune_model(self, X_train, y_train):
        param_distributions = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }
        random_search = RandomizedSearchCV(estimator=self.rf_model, param_distributions=param_distributions, n_iter=10,
                                           scoring='accuracy', cv=5, verbose=3, random_state=self.random_state, n_jobs=-1)
        random_search.fit(X_train, y_train)
        self.best_model = random_search.best_estimator_
        print("Best Parameters:", random_search.best_params_)

# Example usage
if __name__ == "__main__":
    rf_model = RandomForestModel()
    X_train, X_test, y_train, y_test = rf_model.common_functions.load_data('data/emails.mat')
    rf_model.define_model()
    rf_model.tune_model(X_train, y_train)
    rf_model.common_functions.evaluate_model(rf_model.best_model, X_test, y_test)
