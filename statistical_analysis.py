import pandas as pd
import streamlit as st
from scipy.stats import ttest_ind, ttest_rel

def statistical_analysis_page():
    st.title("T-Testing")

    #File Loading/Screening
    df_file = st.file_uploader("Your datasheet in CSV format", type=["csv"])

    if df_file is not None:
        df = pd.read_csv(df_file)
        st.dataframe(df, use_container_width=True)

        # CL level
        conf_level = st.slider("Confidence Level (%)", 80, 99, 95)
        alpha = 1 - (conf_level / 100)

        #!!!! Filter columns by type and turn them to list
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        #Check if there are at least 2 columns to select
        if len(numeric_columns) < 2:
            st.warning("You need at least 2 column to proceed with testing.")
            return

        t_type = st.radio("Choose Test Type", ["Paired", "Unpaired"])
        column_1 = st.selectbox("Select the first column", numeric_columns)
        column_2 = st.selectbox("Select the second column", numeric_columns)

        if st.button("Test"):

            if t_type == "Paired": #Same Group, Check For Length
                if len(df[column_1]) != len(df[column_2]):
                    st.error("Paired t-test requires both columns to have the same number of rows.")
                else: #Testing with ttest_rel, returns Tuple with 2 values
                    t_stat, p_value = ttest_rel(df[column_1], df[column_2], nan_policy = "omit") #omit to skip NaN values
                    st.write(f"Paired t-test results:  (Value can be negative depending on column order) ")
                    st.write(f"T-statistic: {t_stat:.4f}")
                    st.write(f"P-value: {p_value:.4f}")

                    if p_value < alpha:
                        st.info("Reject the null hypothesis: Significant difference exists.")
                    else:
                        st.info("Reject the alternative hypothesis: No significant difference.")

            elif t_type == "Unpaired": #Different Groups, No Need
                t_stat, p_value = ttest_ind(df[column_1], df[column_2], nan_policy='omit')
                st.write(f"Unpaired t-test results:  (Value can be negative depending on column order) ")
                st.write(f"T-statistic: {t_stat:.4f}")
                st.write(f"P-value: {p_value:.4f}")

                if p_value < alpha:
                    st.info("Reject the null hypothesis: Significant difference exists.")
                else:
                    st.info("Reject the alternative hypothesis: No significant difference.")