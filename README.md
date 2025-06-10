# Customer Churn Prediction using a Neural Network

This project demonstrates a complete workflow for predicting customer churn from a banking dataset using a PyTorch-based neural network and leverages `GridSearchCV` to find the optimal model hyperparameters. At the end, the results are compared to a XGBoost model, which typically performs well in churn modeling.

## Dataset

* **Source**: The project uses the "Churn Modelling" dataset, downloaded via the `kagglehub` library.
* **Features**: The model is trained on customer data including `CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, and `EstimatedSalary`.
* **Target Variable**: The model predicts the `Exited` column, where `1` indicates the customer has churned and `0` indicates they have not.

## Project Workflow

The project follows these key steps:

1.  **Data Preprocessing**:
    * Irrelevant columns (`RowNumber`, `CustomerId`, `Surname`) are removed.
    * Categorical features (`Geography`, `Gender`) are converted to numerical format using `OneHotEncoder`.
    * All features are scaled using `StandardScaler` to normalize the data.
    * To address the significant class imbalance in the training data, the minority class is oversampled using the **SMOTE** (Synthetic Minority Over-sampling Technique).
    * For XGBoost, the class imbalance is handled directly using the `scale_pos_weight` parameter.

2.  **Model Architecture**:
    * A neural network is defined in PyTorch with three fully-connected (linear) layers and ReLU activation functions.
    * The model is wrapped using `skorch`'s `NeuralNetClassifier`, making it compatible with the scikit-learn ecosystem.
    * `BCEWithLogitsLoss` is used as the loss function, which is a stable choice for binary classification.

3.  **Hyperparameter Tuning & Training**:
    * `GridSearchCV` is used to systematically search for the best model hyperparameters across a defined grid.
    * The search tunes the learning rate, number of epochs, batch size, hidden layer sizes, and optimizer type (`Adam` vs. `SGD`).
    * The model is optimized for the **Recall**, which is an appropriate metric to identify as many at-risk customers as possible to minimize lost revenue.

4.  **Evaluation**:
    * The best-performing model from the grid search is used to make predictions on the unseen test set.
    * The final performance is measured by the Precision, Recall, F1 Score, and AUC on this test data.
    * The results are compared to the results from the XGBoost model. The XGBoost model demonstrated a more balanced performance with a higher F1-score, while the neural network achieved a higher recall at the cost of precision.

## Libraries & Frameworks

* **Data Handling**: `numpy`, `pandas`, `kagglehub`
* **Machine Learning**: `scikit-learn`, `imblearn`
* **Neural Network**: `pytorch`, `skorch`
* **XGBoost**: `xgboost`
* **Plotting**: `matplotlib`, `seaborn`
