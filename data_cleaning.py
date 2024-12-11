import pandas as pd
import streamlit as st

def data_cleaning_page():
    st.title("Let's see if this works")
    st.write("WHY DID YOU REDEEM IT")

    # After 50+ line of code I learned that you need to create a session state
    # in order to save the changes we made to our dataframe
    if "df" not in st.session_state:
        st.session_state.df = None
    if "action_done" not in st.session_state:
        st.session_state.action_done = False


    df_file = st.file_uploader("Your datasheet in CSV format", type = ["csv"])


    if df_file is not None and st.session_state.df is None:
            st.session_state.df = pd.read_csv(df_file)


    if st.session_state.df is not None:
        st.dataframe(st.session_state.df, use_container_width = True)

        csv = st.session_state.df.to_csv(index=False)

        missing_values = st.session_state.df.isnull().sum()
        total_misses = missing_values.sum()
        columns_with_missing = missing_values[missing_values > 0].index.tolist()


        columns_with_missing.insert(0, "None")
        if total_misses > 0:
            st.write("There are missing values")
            st.write(missing_values)
            selected_column = st.selectbox("Please choose a column to work on",
                                           columns_with_missing
                                           )
            if st.session_state.action_done:
                action = "None"
                st.session_state.action_done = False
            else:
                action = st.selectbox("Choose an action for missing values",
                                      ("None", "Drop Rows", "Remove Column",
                                       "Fill with Mean(Numerical only)",
                                       "Fill with Median(Numerical only)",
                                       "Fill with Mode(Numerical,Categorical)")
                                      )

            if action == "Drop Rows":
                st.session_state.df = st.session_state.df.dropna(subset = [selected_column])
                st.success("Rows successfully deleted")
                st.write("Current Dataframe")
                st.dataframe(st.session_state.df)
                st.session_state.action_done = True
            elif action == "Remove Column":
                st.session_state.df = st.session_state.df.drop(selected_column, axis = 1)
                st.dataframe(st.session_state.df)
                st.session_state.action_done = True
            elif action == "Fill with Mean" and selected_column != "None":
                # Specified selected_column as not None in order to bypass app's SPA behaviour.
                # !!!!!!!! Further explanation needed!!!!!!!!!!!!!!!
                st.session_state.df[selected_column] = st.session_state.df[selected_column].fillna(
                    st.session_state.df[selected_column].mean()
                )
                st.dataframe(st.session_state.df)
                st.session_state.action_done = True
            elif action == "Fill with Median":

                st.session_state.df[selected_column] = (st.session_state.df[selected_column]
                .fillna(st.session_state.df[selected_column].median()))

                st.dataframe(st.session_state.df)
                st.session_state.action_done = True
            elif action == "Fill with Mode":

                st.session_state.df[selected_column] = (st.session_state.df[selected_column]
                .fillna(st.session_state.df[selected_column].mode()[0]))

                st.dataframe(st.session_state.df)
                st.session_state.action_done = True
        else:
            st.write("Your Data Is Clean")
            #Had to add it because when downloaded from "Download as csv" button shown in st.dataframe
            #It added "Unnamed" column containing indexes.
            st.download_button(
                label="Download Data as CSV",
                data=csv,
                file_name="cleaned_data.csv",
                mime="text/csv"
            )

# Testing code to see if our database getting updated
#if st.button("Delete Column Lunch"):
#       st.session_state.df = st.session_state.df.dropna(subset = ["lunch"])
#       st.dataframe(st.session_state.df, use_container_width = True)
