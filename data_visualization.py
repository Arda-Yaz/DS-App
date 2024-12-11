import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def data_visualization_page():
    st.title("Exploratory Data Analysis")

    # File upload option
    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.success("New dataset uploaded successfully!")

    # Check if a dataframe exists
    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("No dataset found. Please upload a dataset to proceed.")
        return

    df = st.session_state.df

    #General information about dataset
    st.header("Dataset Overview", divider = True)
    st.write(f"**Number of Rows:** {df.shape[0]}")
    st.write(f"**Number of Columns:** {df.shape[1]}")
    st.dataframe(df.head())

    st.subheader("Missing Values", divider = True)
    st.write(df.isnull().sum())

    st.subheader("Descriptive Statistics", divider = True)
    st.write(df.describe())

    #Visualizing Data

    st.subheader("Visualizations", divider = True)
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    st.write("### Histogram")
    col = st.selectbox("Choose a column for Histogram", options=numeric_columns)
    if col:
        fig, ax = plt.subplots()  # Create figure and axes with subplots
        sns.histplot(df[col], kde=True, ax=ax)  #Kernel density estimation  = True for smooth graph
        st.pyplot(fig)


    #Boxplot
    st.write("### Boxplot")
    box_col = st.selectbox("Choose a column for Boxplot", options=numeric_columns)
    if box_col:
        fig, ax = plt.subplots()
        sns.boxplot(y=df[box_col], ax=ax)
        st.pyplot(fig)

    #Heatmap
    st.write("### Correlation Heatmap")
    if st.button("Generate Heatmap"):
        numeric_df = df.select_dtypes(include=['float64', 'int64'])  # Filter to numeric columns
        if numeric_df.empty:
            st.error("No numeric columns available to calculate correlations.")
        else:
            corr = numeric_df.corr()
            fig, ax = plt.subplots(figsize=(12, 9))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

    #Scatter Plot
    st.write("### Scatter Plot")
    x = st.selectbox("Choose X-axis", options=numeric_columns)
    y = st.selectbox("Choose Y-axis", options=numeric_columns)
    if x and y:
        fig, ax = plt.subplots()
        sns.scatterplot(x=df[x], y=df[y], ax=ax)
        st.pyplot(fig)

    #Bar Chart
    st.write("### Bar Chart")
    bar_col = st.selectbox("Choose a categorical column for Bar Chart", options=categorical_columns)
    if bar_col:
        fig, ax = plt.subplots()
        df[bar_col].value_counts().plot(kind='bar', ax=ax)
        st.pyplot(fig)