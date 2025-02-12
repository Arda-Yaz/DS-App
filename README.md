# DS-App

In this project, I developed a Streamlit application that simplifies data analysis and visualization. 
Users can upload their data, visualize it, and choose different types of analysis to dive deeper into the data. 
The application guides users step-by-step while providing an interactive and user-friendly experience. 
It covers essential data science functions such as data cleaning, statistical analysis, model training, and visualization.
Users can filter data, create graphs, and perform predictive analysis directly through the app. 
This way, not only data science professionals but also individuals from various sectors can easily access tools to 
perform data analysis.


Encountered Problems

Problem:
Using the st.slider component for model configuration caused an issue where the page re-rendered upon any change to the 
slider value, resetting the previously trained model. This made it impossible to experiment with different slider values.

Solution:
To overcome this, I moved the st.slider component outside the scope of the "Train Model" button. By doing so, Streamlit 
no longer treats slider value changes as a trigger to reset the session state.