import streamlit as st


def model_guide_page():

    st.title("_Model Guide_")
    st.write("Models are mathematical formulas or algorithms used to make predictions"
            " about unknown data based on patterns found in known datasets.")
    st.divider()
    st.subheader("| Linear Regression")

    st.write("Linear Regression is a predictive modeling technique used to estimate the relationship between a dependent"
             " variable (target) and one or more independent variables (features). It assumes this relationship is linear,"
             " meaning changes in the inputs lead to proportional changes in the output."
             " Although it is a simple and interpretable model, it does not handle non-linear relationships well and is sensitive to outliers in the data.")
    st.divider()
    st.subheader("| Decision Tree")
    st.write("Decision Tree is a non-linear predictive model that splits data into branches based on feature values to "
             "make decisions or predictions. It works well for both classification and regression tasks, is easy to visualize,"
             " and does not require feature scaling."
             " However, it can easily overfit the data, especially without pruning or depth constraints.")
    st.divider()
    st.subheader("| Random Forest")
    st.write("Random Forest is an ensemble learning method that combines multiple decision trees to improve prediction "
             "accuracy. It reduces overfitting by averaging results from many trees and handles missing data and feature "
             "importance well. However, it can be computationally expensive and less interpretable compared to a single decision tree.")
    st.divider()
    st.subheader("| Support Vector Machine (SVM)")
    st.write("SVM is a classification and regression technique that finds the hyperplane that best separates data into classes. "
             "It performs well with high-dimensional data and complex relationships. However, SVMs are computationally intensive,"
             " especially with large datasets, and require careful tuning of kernel functions and parameters.")
    st.divider()
    st.subheader("| K-Nearest Neighbors (KNN)")
    st.write("KNN is a simple, instance-based learning method that makes predictions by finding the closest data points (neighbors)"
             " to a given input. It works well for smaller datasets and is easy to implement. "
             "However, it can be computationally expensive for large datasets and is sensitive to irrelevant features and scaling.")
    st.divider()
    st.subheader("| Logistic Regression")
    st.write("Logistic Regression is a linear model for binary classification tasks that estimates the probability of a "
             "target belonging to a class. It is simple, interpretable, and works well when the relationship between features "
             "and the target is linear. However, it struggles with non-linear relationships and requires careful handling of "
             "imbalanced data.")
    st.divider()