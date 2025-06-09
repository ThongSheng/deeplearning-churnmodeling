# Customer Churn Prediction using a Neural Network

This project demonstrates a complete workflow for predicting customer churn from a banking dataset. It uses a PyTorch-based neural network and leverages `GridSearchCV` to find the optimal model hyperparameters.

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

2.  **Model Architecture**:
    * A neural network is defined in PyTorch with three fully-connected (linear) layers and ReLU activation functions.
    * The model is wrapped using `skorch`'s `NeuralNetClassifier`, making it compatible with the scikit-learn ecosystem.
    * `BCEWithLogitsLoss` is used as the loss function, which is a stable choice for binary classification.

3.  **Hyperparameter Tuning & Training**:
    * `GridSearchCV` is used to systematically search for the best model hyperparameters across a defined grid.
    * The search tunes the learning rate, number of epochs, batch size, hidden layer sizes, and optimizer type (`Adam` vs. `SGD`).
    * The model is optimized for the **F1-score**, which is an appropriate metric given the class imbalance where false negatives and false positives are both significant.

4.  **Evaluation**:
    * The best-performing model from the grid search is used to make predictions on the unseen test set.
    * The final performance is measured by the F1-score on this test data.

## Libraries & Frameworks

* **Data Handling**: `pandas`, `kagglehub`
* **Machine Learning**: `scikit-learn`, `imblearn`
* **Neural Network**: `torch`, `skorch`
* **Plotting**: `matplotlib`
