import os

from scipy.io import loadmat
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns

class CommonModelFunctions:
    def __init__(self, random_state=42, results_dir='results/'):
        self.random_state = random_state
        self.results_dir = results_dir

        # Create results directory if it doesn't exist
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    # Function to load and preprocess data
    def load_data(self, filepath):
        data = loadmat(filepath)
        X = data['X'].T
        y = data['Y'].T

        # Handle imbalanced dataset using RandomOverSampler
        ros = RandomOverSampler(random_state=self.random_state)
        X_resampled, y_resampled = ros.fit_resample(X, y)

        return train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=self.random_state)

    # Function to save plots and text results
    def save_plot(self, fig, filename):
        path = os.path.join(self.results_dir, filename)
        fig.savefig(path)
        plt.close(fig)
        print(f"Saved plot to {path}")

    def save_text(self, text, filename):
        path = os.path.join(self.results_dir, filename)
        with open(path, 'w') as file:
            file.write(text)
        print(f"Saved text to {path}")

    # Function to evaluate the model
    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)

        # Confusion Matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Spam', 'Spam'], yticklabels=['Non-Spam', 'Spam'], ax=ax)
        ax.set_title(f'Confusion Matrix - {model.__class__.__name__}')
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        self.save_plot(fig, f'{model.__class__.__name__}_confusion_matrix.png')

        # Classification report
        report = classification_report(y_test, y_pred)
        print(f"{model.__class__.__name__} Classification Report:")
        print(report)
        self.save_text(report, f'{model.__class__.__name__}_classification_report.txt')

        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy of the {model.__class__.__name__} model: {accuracy * 100:.2f}%")

        # True Positive Rate (TPR), False Positive Rate (FPR), and ROC-AUC
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)

        # ROC Curve
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.2f}')
        ax.plot([0, 1], [0, 1], linestyle='--', color='red')  # Dashed diagonal line
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_title(f'ROC Curve - {model.__class__.__name__}')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right')
        self.save_plot(fig, f'{model.__class__.__name__}_roc_curve.png')

        # Save metrics summary
        metrics_summary = f"{model.__class__.__name__} Metrics:\n" \
                          f"Accuracy: {accuracy * 100:.2f}%\n" \
                          f"ROC AUC: {roc_auc:.2f}\n"
        self.save_text(metrics_summary, f'{model.__class__.__name__}_metrics_summary.txt')

