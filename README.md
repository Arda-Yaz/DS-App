# DS-App


Encountered Problems

Problem:
Using the st.slider component for model configuration caused an issue where the page re-rendered upon any change to the 
slider value, resetting the previously trained model. This made it impossible to experiment with different slider values.

Solution:
To overcome this, I moved the st.slider component outside the scope of the "Train Model" button. By doing so, Streamlit 
no longer treats slider value changes as a trigger to reset the session state.