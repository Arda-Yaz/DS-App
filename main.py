import streamlit as st
from data_cleaning import data_cleaning_page
from statistical_analysis import statistical_analysis_page
from machine_learning import machine_learning_page
from data_visualization import data_visualization_page
from model_guide import model_guide_page

def home_page():
    #Choose Page
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a Page", ["Data Cleaning","Data Visualization",
                                                  "Statistical Tests", "Machine Models", "Model Guide"])

    if page == "Data Cleaning":
        data_cleaning_page()
    elif page == "Statistical Tests":
        statistical_analysis_page()
    elif page == "Data Visualization":
        data_visualization_page()
    elif page == "Machine Models":
        machine_learning_page()
    elif page == "Model Guide":
        model_guide_page()

if __name__ == "__main__":
    home_page()