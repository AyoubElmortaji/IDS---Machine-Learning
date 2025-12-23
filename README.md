
# Intrusion Detection System using Machine Learning

[![GitHub repo size](https://img.shields.io/github/repo-size/AyoubElmortaji/IDS---Machine-Learning)](https://github.com/AyoubElmortaji/IDS---Machine-Learning)
[![GitHub last commit](https://img.shields.io/github/last-commit/AyoubElmortaji/IDS---Machine-Learning)](https://github.com/AyoubElmortaji/IDS---Machine-Learning/commits/main)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project implements an **Intrusion Detection System (IDS)** using classical machine learning algorithms to detect network intrusions and attacks. It is based on the **NSL-KDD** dataset (an improved version of the classic KDD Cup 99 dataset).

The project trains and evaluates multiple models for binary classification (normal vs. attack) and provides saved pre-trained models for reuse.

## Project Structure

```
IDS---Machine-Learning/
├── KDDTrain+.txt                 # Training dataset
├── KDDTest+.txt                  # Test dataset
├── Logistique Reg.ipynb          # Logistic Regression model
├── DT - RF.ipynb                 # Decision Tree & Random Forest models
├── XGboost.ipynb                 # XGBoost model
├── notebook.ipynb                # Additional/general notebook (optional)
├── ids_model_logistic_regression.pkl   # Saved Logistic Regression model
├── ids_model_random_forest.pkl         # Saved Random Forest model
├── ids_model_xgboost.pkl               # Saved XGBoost model
└── app.py                        # Optional script for model inference/deployment
```

## Features

- Preprocessing of the NSL-KDD dataset (handling categorical features, scaling, etc.)
- Training of three popular classifiers:
  - Logistic Regression
  - Decision Tree & Random Forest
  - XGBoost
- Model evaluation (accuracy, precision, recall, F1-score, confusion matrix)
- Saving trained models in pickle format for easy reuse
- Basic inference script (`app.py`) for loading and using the models

## Dataset

The project uses the **NSL-KDD** dataset, which is a refined version of the original KDD Cup 99 dataset with improved quality (no duplicate records, better class distribution).

- **Training file**: `KDDTrain+.txt`
- **Test file**: `KDDTest+.txt`

These files are already included in the repository.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AyoubElmortaji/IDS---Machine-Learning.git
   cd IDS---Machine-Learning
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(If no `requirements.txt` exists yet, use the following command:)*
   ```bash
   pip install pandas numpy scikit-learn xgboost joblib jupyter matplotlib seaborn
   ```

## Usage

### 1. Re-run Training (via Jupyter Notebooks)

Open any of the notebooks and run the cells:

```bash
jupyter notebook
```

- `Logistique Reg.ipynb` → Logistic Regression
- `DT - RF.ipynb` → Decision Tree & Random Forest
- `XGboost.ipynb` → XGBoost

### 2. Use Pre-trained Models

Load a saved model and make predictions:

```python
import joblib
import pandas as pd

# Load model
model = joblib.load('ids_model_random_forest.pkl')

# Example: predict on new data (replace with actual preprocessed data)
# new_data = pd.read_csv('your_new_data.csv')
# predictions = model.predict(new_data)
# print(predictions)
```

### 3. Run the Inference App (if `app.py` exists)

```bash
python app.py
```

(Depending on the implementation, this might load a model and allow input for predictions or run a simple web interface.)

## Results

The notebooks include evaluation metrics and visualizations (confusion matrices, ROC curves, etc.).

Random Forest and XGBoost typically achieve the best performance on this dataset.

## Contributing

Contributions are welcome!  
Feel free to open an issue or submit a pull request.

## License

Feel free to use, modify, and share.

## Acknowledgments

- Dataset: NSL-KDD (https://www.unb.ca/cic/datasets/nsl.html)
- Built as an educational/experimental project

