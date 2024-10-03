from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from common_functions import CommonModelFunctions

class LogisticRegressionModel:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.log_reg = None
        self.best_model = None
        self.common_functions = CommonModelFunctions(random_state)

    # Define the Logistic Regression model
    def define_model(self):
        self.log_reg = LogisticRegression(solver='liblinear', random_state=self.random_state)

    # Hyperparameter tuning
    def tune_model(self, X_train, y_train):
        param_distributions = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2']
        }
        random_search = RandomizedSearchCV(estimator=self.log_reg, param_distributions=param_distributions, n_iter=10,
                                           scoring='accuracy', cv=5, verbose=3, random_state=self.random_state, n_jobs=-1)
        random_search.fit(X_train, y_train)
        self.best_model = random_search.best_estimator_
        print("Best Parameters:", random_search.best_params_)

# Example usage
if __name__ == "__main__":
    log_reg_model = LogisticRegressionModel()
    X_train, X_test, y_train, y_test = log_reg_model.common_functions.load_data('data/emails.mat')
    log_reg_model.define_model()
    log_reg_model.tune_model(X_train, y_train)
    log_reg_model.common_functions.evaluate_model(log_reg_model.best_model, X_test, y_test)
