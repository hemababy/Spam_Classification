# Spam Email Filter Project

## Overview
This project implements a **machine learning-based spam email filter**. The task is to classify incoming emails as either **Spam** or **Non-Spam** based on a dataset.

The project includes implementations of three machine learning models:
- **Random Forest**
- **Logistic Regression**
- **Naive Bayes**
    

## Project Structure

```text
├── data/
│   └── emails.csv               # Dataset containing email data
├── models/
│   ├── random_forest.py         # Random Forest model and tuning
│   ├── logistic_regression.py   # Logistic Regression model and tuning
│   └── naive_bayes.py           # Naive Bayes model and tuning
├── notebooks/
│   └── spam_filter_analysis.ipynb  # Jupyter notebook with full analysis
├── results/
│   └── confusion_matrix_plots/  # Saved confusion matrix plots
│   └── metrics/                 # Saved metrics reports (accuracy, FPR, etc.)
├── README.md                    # This README file
├── requirements.txt             # List of required Python packages
└── LICENSE                      # License for this project
```

## Installation

Follow these steps to set up and run the project locally:

### 1. Clone the repository:
```bash
git clone https://github.com/hemababy/Spam_Classification
cd spam_email_filter
```
### 2. Create a virtual environment:
```bash
python3 -m venv env
source env/bin/activate
```
### 3. Install the required packages:
```bash
pip install -r requirements.txt
```
### 4. Run the Jupyter notebook:
```bash
jupyter notebook notebooks/spam_filter_analysis.ipynb
```
## 5. Run the models:
```bash
python models/random_forest.py
python models/logistic_regression.py
python models/naive_bayes.py
```
## 6. View the results:
The results of the models will be saved in the `results/` directory.


### Usage
The project can be used to classify incoming emails as either **Spam** or **Non-Spam** based on the trained machine learning models. The models can be retrained on new data to improve performance.
## Hyperparameter Tuning:
The **random_forest.py** and **logistic_regression.py** scripts include hyperparameter tuning using RandomizedSearchCV. You can modify the hyperparameters in the script and rerun it for optimization.
## Evaluation Metrics
For each model, the following metrics are calculated:
* Accuracy
* Precision
* Recall (True Positive Rate)
* False Positive Rate (FPR)
* F1-Score
* ROC-AUC Curve

You can find the performance summaries and confusion matrix plots in the `results/` folder.


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Citation

Citation information for this project could be found in [CITATION.cff](./CITATION.cff) file.

## Contact Information

For any questions or feedback, don't hesitate to get in touch with `hematechie@gmail.com`.
Name: Hemalatha Sekar.