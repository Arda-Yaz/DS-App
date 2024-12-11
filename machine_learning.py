import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.tree import DecisionTreeRegressor


def machine_learning_page():
    st.title("Machine Models")
    st.write("Choose a machine model for prediction")


    # Make sure users uploaded their dataframe in Data Cleaning section
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("No data found. Please upload and clean your dataset in the Data Cleaning section.")
        return

    #
    elif "df" in st.session_state:
        df = st.session_state.df
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

        if not numeric_columns:
            st.error("Your dataset has no numeric columns. Prediction is not possible.")
            return

        target_column = st.selectbox("Choose the target column", options=numeric_columns)

        #Every column in numeric_columns except target_column
        feature_columns = [col for col in numeric_columns if col != target_column]

        #Instead of using multiple selectbox, multiselect method has been used
        selected_features = st.multiselect("Select feature columns", options=feature_columns, default=feature_columns)


        if len(selected_features) > 0:
            model_option = st.selectbox("Select a model", ["Linear Regression", "Decision Tree"])

            if st.button("Train Model"):
                X = df[selected_features]
                y = df[target_column]
                # X = Input values
                # y = Output values
                #test_size = 0.2  -> 20% for testing, 80% for training (Can leave the choice up to User in the future)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



                if model_option == "Linear Regression":
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                elif model_option == "Decision Tree":
                    model_depth = st.slider("Max Depth", 1, 20, 5)
                    model = DecisionTreeRegressor(max_depth=model_depth)
                    model.fit(X_train, y_train)

                # Save the trained model and features in session state in order to avoid rendering issues
                st.session_state.model = model
                st.session_state.selected_features = selected_features

                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)

                #Used RMSE in order to return error in original unit
                train_error = root_mean_squared_error(y_train, y_pred_train)
                test_error = root_mean_squared_error(y_test, y_pred_test)
                #!!!!! Look for ways to improve RMSE

                st.write(f"Training RMSE: {train_error:.2f}")
                st.write(f"Testing RMSE: {test_error:.2f}")

    if "model" in st.session_state:
        st.subheader("Make Predictions")
        #Loop to go through each iteration in selected features and take input for them
        input_data = {
            col: st.number_input(f"Input value for {col}", value=0.0) for col in st.session_state.selected_features
        }

        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])
            prediction = st.session_state.model.predict(input_df)[0]
            st.success(f"Prediction: {prediction:.2f}")
    else:
        st.warning("Please train a model first!")