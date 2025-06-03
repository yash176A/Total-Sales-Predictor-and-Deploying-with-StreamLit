# Total-Sales-Predictor-and-Deploying-with-StreamLit
# Quick Stop Sales Predictor

## Project Overview

This project, "Quick Stop Sales Predictor," is a comprehensive machine learning endeavor focused on analyzing and predicting sales data for a convenience store. The core of the project involves:

Data Simulation: Generating a synthetic dataset of daily sales across 11 different product categories (e.g., Groceries, Beer, Tobacco, Lottery) over 100 days. A Total_Sales column is derived by summing these categories, and a binary High_Sales target is created based on whether Total_Sales exceeds the median.

Regression Modeling (Total Sales):

Linear Regression: An initial LinearRegression model is trained to predict Total_Sales, with its performance evaluated using Mean Squared Error (MSE) and cross-validation.
Ridge Regression with Hyperparameter Tuning: A more robust Ridge regression model is employed, and its optimal alpha hyperparameter is determined using GridSearchCV to minimize MSE. This optimized Ridge model is identified as the best_model for predicting total sales.
Classification Modeling (High Sales):

Logistic Regression: A LogisticRegression model is trained to classify whether daily sales fall into the "High Sales" category.
Data Scaling: StandardScaler is used to preprocess the features, which is crucial for the performance of many machine learning algorithms like Logistic Regression.
Evaluation: The classification model's performance is assessed using standard metrics such as Confusion Matrix, Classification Report (including Precision, Recall, F1-score), Sensitivity, Specificity, and the Area Under the Receiver Operating Characteristic (ROC AUC) curve.
Dimensionality Reduction (PCA):

Principal Component Analysis (PCA) is applied to the scaled sales data to reduce its dimensionality to 5 components. This helps in understanding the underlying structure of the data and potentially reducing noise. The explained variance ratio for each component is visualized.
Model Persistence and Deployment:

The best-performing regression model (the Ridge model obtained from GridSearchCV) is saved using Python's pickle module into a file named model.pkl.
This model.pkl file is then loaded by a separate Streamlit web application, allowing users to input new sales data for various categories and receive real-time predictions for Total_Sales.

## Features

-   **Synthetic Data Generation:** Creates a dataset of daily sales for 11 different product categories.
-   **Total Sales Calculation:** Derives total daily sales from individual category sales.
-   **High Sales Classification:** Identifies days with "High Sales" (above median total sales) for classification tasks.
-   **Regression Modeling:**
    -   Predicts `Total_Sales` using `LinearRegression`.
    -   Optimizes `Total_Sales` prediction using `Ridge` regression with `GridSearchCV` for hyperparameter tuning.
-   **Classification Modeling:**
    -   Predicts `High_Sales` using `LogisticRegression`.
-   **Data Preprocessing:** Utilizes `StandardScaler` for feature scaling.
-   **Dimensionality Reduction:** Applies Principal Component Analysis (PCA) to explore data structure and reduce dimensions.
-   **Model Evaluation:** Employs various metrics:
    -   Regression: Mean Squared Error (MSE), Cross-Validation MSE.
    -   Classification: Confusion Matrix, Classification Report (Precision, Recall, F1-score), Sensitivity, Specificity, ROC AUC.
-   **Model Persistence:** Saves the best-trained model (`Ridge` regression) using `pickle` for deployment.
-   **Streamlit Web Application:** A user-friendly interface for real-time sales prediction.

## Models Used

-   `LinearRegression` (for initial regression analysis)
-   `LogisticRegression` (for classification)
-   `Ridge` (for optimized regression, chosen as the final deployment model)

## Setup and Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yash176A/Total-Sales-Predictor-and-Deploying-with-StreamLit.git](https://github.com/yash176A/Total-Sales-Predictor-and-Deploying-with-StreamLit.git)
    cd Total-Sales-Predictor-and-Deploying-with-StreamLit/App # Or wherever your app.py is
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    -   **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    -   **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install the required Python packages:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib streamlit
    ```
 Run the Streamlit Web Application
After generating model.pkl, navigate to the App directory (where app.py is located) and run the Streamlit application:

Bash

cd App
streamlit run app.py
This will open the Streamlit app in your default web browser. You can then input values for the various sales categories and get a real-time prediction for total sales.

Usage
The Streamlit application provides a user-friendly interface:

Input Fields: Adjust the numerical values for each sales category (Groceries, Beer, Instant Lottery, etc.).
Predict Button: Click the "PREDICT TOTAL SALES" button.
Prediction Display: The predicted total sales will be displayed prominently.
Insights and Results
The project demonstrates how to build a full ML pipeline from data generation to deployment.
Ridge Regression, with its regularization, helps prevent overfitting and provides robust predictions for total sales.
The classification model provides insights into factors contributing to "High Sales" days.
PCA helps understand the main drivers of variance in the sales data.
