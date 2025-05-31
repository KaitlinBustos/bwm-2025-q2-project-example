import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os # Import the os module

# --- Define paths relative to the app.py script ---
# Get the directory where app.py is located
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, 'svm_model.joblib')
COLUMNS_PATH = os.path.join(APP_DIR, 'training_columns.json')

# Load the trained model
try:
    model = joblib.load(MODEL_PATH) # Use the new MODEL_PATH
except FileNotFoundError:
    st.error(f"Model file '{MODEL_PATH}' not found. Please ensure 'svm_model.joblib' is in the 'src/' directory alongside 'app.py'.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Load the training columns
try:
    with open(COLUMNS_PATH, 'r') as f: # Use the new COLUMNS_PATH
        training_columns = json.load(f)
except FileNotFoundError:
    st.error(f"Training columns file '{COLUMNS_PATH}' not found. Please ensure 'training_columns.json' is in the 'src/' directory alongside 'app.py'.")
    st.stop()
except Exception as e:
    st.error(f"Error loading training columns: {e}")
    st.stop()
# Title of the app
st.title('Mental Health Stigma in the Workplace Survey 🧠')

st.markdown("""
This survey will help predict the likelihood of an individual facing a mental health consequence in the workplace based on their responses.
Please answer the questions below.
""")

# --- Define the `encode_df` function from your notebook ---
# This function is crucial for preprocessing the input to match the model's training data
def encode_df(df_input):
    df_processed = df_input.copy()

    # Ordinal encoding (as per your notebook)
    ordinal_maps = {
        'no_employees': {
            '1-5': 0,
            '6-25': 1,
            '26-100': 2,
            '100-500': 3,
            '500-1000': 4,
            'More than 1000': 5
        },
        'leave': {
            'Very easy': 0,
            'Somewhat easy': 1,
            "Don't know": 2,
            'Somewhat difficult': 3,
            'Very difficult': 4
        },
        'coworkers': {
            'No': 0,
            'Some of them': 1,
            'Yes': 2
        },
        'supervisor': {
            'No': 0,
            'Some of them': 1,
            'Yes': 2
        },
        'age_group': { # Assuming 'age_group' is the name of the binned age column
            '18-25': 0,
            '26-30': 1,
            '31-35': 2,
            '36-40': 3,
            '41-45': 4,
            '46-50': 5,
            '51-55': 6,
            '55+': 7
        }
        # Note: mental_health_consequence is the target, not encoded here
    }
    for col, mapping in ordinal_maps.items():
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].map(mapping)

    # One-hot encode nominal categorical variables
    # Make sure these are the exact columns used for one-hot encoding in the notebook
    # before fitting the SVM with GridSearchCV
    nominal_cols = ['Country', 'family_history', 'benefits', 'care_options',
                    'wellness_program', 'seek_help', 'remote_work', 'tech_company',
                    'anonymity', 'mental_vs_physical', 'obs_consequence',
                    'gender_cleaned', 'treatment']

    for col in nominal_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(str) # Ensure consistent type for get_dummies
            # Use pd.get_dummies with the same parameters as in your notebook
            # The `columns` parameter ensures only specified columns are dummied
            # `prefix` and `prefix_sep` should match your notebook's get_dummies usage
            # `drop_first=False` was specified in the notebook cell 22.
            df_processed = pd.get_dummies(df_processed, columns=[col], prefix=col, drop_first=False)

    return df_processed

# --- Survey Questions ---
# These should correspond to the features your model was trained on *before* encoding
st.header("Survey Questions")

# Demographics
age_group_options = ['18-25', '26-30', '31-35', '36-40', '41-45', '46-50', '51-55', '55+']
age_group = st.selectbox("What is your age group?", age_group_options)

gender_cleaned_options = ['Male', 'Female', 'Other'] # From your notebook's clean_gender function
gender_cleaned = st.selectbox("What is your gender?", gender_cleaned_options)

country_options = ['United States', 'United Kingdom', 'Canada', 'Germany', 'Ireland', 'Netherlands', 'Australia', 'France'] # From your notebook's filtering
country = st.selectbox("What is your country?", country_options)

# Workplace & Health History
family_history_options = ['No', 'Yes']
family_history = st.selectbox("Do you have a family history of mental illness?", family_history_options)

treatment_options = ['Yes', 'No']
treatment = st.selectbox("Have you sought treatment for a mental health condition?", treatment_options)

no_employees_options = ['1-5', '6-25', '26-100', '100-500', '500-1000', 'More than 1000']
no_employees = st.selectbox("How many employees does your company or organization have?", no_employees_options)

remote_work_options = ['No', 'Yes']
remote_work = st.selectbox("Do you work remotely (outside of an office) at least 50% of the time?", remote_work_options)

tech_company_options = ['Yes', 'No']
tech_company = st.selectbox("Is your employer primarily a tech company/organization?", tech_company_options)

benefits_options = ['Yes', "Don't know", 'No']
benefits = st.selectbox("Does your employer provide mental health benefits?", benefits_options)

care_options_options = ['No', 'Yes', 'Not sure']
care_options = st.selectbox("Do you know the options for mental health care your employer provides?", care_options_options)

wellness_program_options = ['No', 'Yes', "Don't know"]
wellness_program = st.selectbox("Has your employer ever discussed mental health as part of an employee wellness program?", wellness_program_options)

seek_help_options = ['No', "Don't know", 'Yes']
seek_help = st.selectbox("Does your employer provide resources to learn more about mental health issues and how to seek help?", seek_help_options)

anonymity_options = ["Don't know", 'Yes', 'No']
anonymity = st.selectbox("Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources?", anonymity_options)

leave_options = ["Don't know", 'Somewhat easy', 'Very easy', 'Somewhat difficult', 'Very difficult']
leave = st.selectbox("How easy is it for you to take medical leave for a mental health condition?", leave_options)

coworkers_options = ['Some of them', 'No', 'Yes']
coworkers = st.selectbox("Would you be willing to discuss a mental health issue with your coworkers?", coworkers_options)

supervisor_options = ['Yes', 'No', 'Some of them']
supervisor = st.selectbox("Would you be willing to discuss a mental health issue with your direct supervisor(s)?", supervisor_options)

mental_vs_physical_options = ["Don't know", 'No', 'Yes']
mental_vs_physical = st.selectbox("Do you feel that your employer takes mental health as seriously as physical health?", mental_vs_physical_options)

obs_consequence_options = ['No', 'Yes']
obs_consequence = st.selectbox("Have you heard of or observed negative consequences for coworkers with mental health conditions in your workplace?", obs_consequence_options)


# --- Prediction Logic ---
if st.button('Predict Stigma Likelihood', type="primary"):
    # Create a dictionary from the inputs
    # Ensure keys match the column names expected by encode_df (original feature names)
    input_data = {
        'age_group': age_group,
        'gender_cleaned': gender_cleaned,
        'Country': country,
        'family_history': family_history,
        'treatment': treatment,
        'no_employees': no_employees,
        'remote_work': remote_work,
        'tech_company': tech_company,
        'benefits': benefits,
        'care_options': care_options,
        'wellness_program': wellness_program,
        'seek_help': seek_help,
        'anonymity': anonymity,
        'leave': leave,
        'coworkers': coworkers,
        'supervisor': supervisor,
        'mental_vs_physical': mental_vs_physical,
        'obs_consequence': obs_consequence
    }

    input_df = pd.DataFrame([input_data])

    # Preprocess the input using the encode_df function
    # This is the most critical step for ensuring correct input to the model
    try:
        processed_df = encode_df(input_df)

        # Align columns with the training data
        # Add missing columns (from one-hot encoding) and fill with 0
        # Reorder columns to match the training data order
        # This step is VITAL for the model to work correctly.
        current_cols = processed_df.columns.tolist()
        aligned_data = {}
        for col in training_columns:
            if col in current_cols:
                aligned_data[col] = processed_df[col].iloc[0]
            else:
                aligned_data[col] = 0 # For one-hot encoded columns not present in current input

        final_input_df = pd.DataFrame([aligned_data], columns=training_columns)


        # Make prediction
        prediction_encoded = model.predict(final_input_df)

        # Decode prediction
        # Assuming 0 means 'No consequence' and 1 means 'Yes, consequence'
        # This mapping should match your notebook's target variable after filtering 'Maybe'
        prediction_label = "Yes" if prediction_encoded[0] == 1 else "No"

        st.subheader("Prediction Result:")
        if prediction_label == "Yes":
            st.error(f"Likely to face a mental health consequence in the workplace: **{prediction_label}**")
        else:
            st.success(f"Likely to face a mental health consequence in the workplace: **{prediction_label}**")

       
    except Exception as e:
        st.error(f"An error occurred during preprocessing or prediction: {e}")
        st.write("Input DataFrame before encoding:", input_df)
        if 'processed_df' in locals():
             st.write("Processed DataFrame before alignment:", processed_df.head())
             st.write("Processed DF Columns:", processed_df.columns.tolist())
        st.write("Expected Training Columns (first 10):", training_columns[:10])
        st.write("Expected Training Columns (count):", len(training_columns))


st.sidebar.header("About")
st.sidebar.info(
    "This application uses a machine learning model to predict the likelihood "
    "of experiencing negative consequences related to mental health in the workplace. "
    "The prediction is based on the survey responses provided."
)
st.sidebar.warning("This is a predictive model and not a diagnostic tool. Results should be interpreted with caution.")
