import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import warnings
np.warnings = warnings
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

st.set_page_config(
    page_title='Predict Page',
    page_icon=':',
    layout='wide'
)

# Load models
@st.cache_data(show_spinner='Model Loading')
def logistic_regression_pipeline():
    model = joblib.load('./models/logisticregression.joblib')
    return model

@st.cache_data(show_spinner='Model Loading')
def random_forest_pipeline():
    model = joblib.load('./models/randomforest.joblib')
    return model

@st.cache_resource(show_spinner='Model Loading')
def load_encoder():
    encoder = joblib.load('./models/encoder.joblib')
    return encoder

def select_model():
    column1, column2 = st.columns(2)
    
    with column1:
        model_name = st.selectbox('Select a Model', options=['Logistic Regression', 'Random Forest'])
        if model_name == 'Logistic Regression':
            selected_model = logistic_regression_pipeline()   
        elif model_name == 'Random Forest':
            selected_model = random_forest_pipeline()
        encoder = load_encoder()
    with column2:
        pass
        
    return selected_model, encoder

def make_prediction(model, encoder):
    df = st.session_state['df']
    prediction = model.predict(df)
    probabilities = model.predict_proba(df)
    
    st.session_state['prediction'] = prediction
    st.session_state['probability'] = probabilities
    
    if isinstance(model, LogisticRegression):
        model_name = "Logistic Regression"
    elif isinstance(model, RandomForestClassifier):
        model_name = "Random Forest"
    elif isinstance(model, Pipeline):
        # Check if the pipeline contains Logistic Regression or Random Forest
        pipeline_steps = model.named_steps.values()
        if any(isinstance(step, LogisticRegression) for step in pipeline_steps):
            model_name = "Logistic Regression"
        elif any(isinstance(step, RandomForestClassifier) for step in pipeline_steps):
            model_name = "Random Forest"
        else:
            model_name = "Unknown Model: Pipeline"
    else:
        model_name = "Unknown Model: " + str(type(model))
        print("Unknown Model: ", type(model))
        
    # Save prediction to history CSV file
    save_prediction_to_csv(df, prediction, model_name)
    return prediction

def predict():
    if 'prediction' not in st.session_state:
        st.session_state['prediction'] = None
    
    # Dictionary to store input features
    model, encoder = select_model()
    st.session_state['model_name'] = model.__class__.__name__  # Store the selected model name
    
    with st.form('input feature'):
        # Form inputs...

        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write('### Personal Information')
            # Add input fields and store values in input_features dictionary
            gender = st.selectbox('gender', options=['Male', 'Female'], key='gender')
            SeniorCitizen = st.selectbox('seniorcitizen', options=['Yes', 'No'], key='seniorcitizen')
            Partner = st.selectbox('partner', options=['Yes', 'No'], key='partner')
            Dependents = st.selectbox('dependents', options=['Yes', 'No'], key='dependents')
            tenure = st.number_input('tenure', min_value=0, max_value=71, step=1, key='tenure')  
        
        with col2:
            st.write('### Subscriptions')
            # Input fields for subscription-related features
            PhoneService = st.selectbox('phoneservice', options=['Yes', 'No'], key='phoneservice') 
            MultipleLines = st.selectbox('multiplelines', options=['Yes', 'No'], key='multiplelines') 
            InternetService = st.selectbox('internetservice', options=['Fiber optic', 'DSL'], key='internetservice') 
            OnlineSecurity = st.selectbox('onlinesecurity', options=['Yes', 'No'], key='onlinesecurity') 
            OnlineBackup = st.selectbox('onlinebackup', options=['Yes', 'No'], key='onlinebackup')
            DeviceProtection = st.selectbox('deviceprotection', options=['Yes', 'No'], key='deviceprotection') 
            TechSupport = st.selectbox('techsupport', options=['Yes', 'No'], key='techsupport') 
            StreamingTV = st.selectbox('streamingtv', options=['Yes', 'No'], key='streamingtv') 
            StreamingMovies = st.selectbox('streamingmovies', options=['Yes', 'No'], key='streamingmovies')  
        
        with col3:
            st.write('### Payment Options')
            # Input fields for payment-related features
            Contract = st.selectbox('contract', options=['Month-to-month', 'Two year', 'One year'], key='contract') 
            PaperlessBilling = st.selectbox('paperlessbilling', options=['Yes', 'No'], key='paperlessbilling') 
            PaymentMethod = st.selectbox('paymentmethod', options=['Electronic check', 'Credit card (automatic)', 'Mailed check', 'Bank transfer (automatic)'], key='paymentmethod') 
            MonthlyCharges = st.number_input('monthlycharges', min_value=0, key='monthlycharges')
            TotalCharges = st.number_input('totalcharges', min_value=0, key='totalcharges') 
   
        input_features = pd.DataFrame({
            'gender': [gender], 
            'seniorcitizen': [SeniorCitizen], 
            'partner': [Partner], 
            'dependents': [Dependents], 
            'tenure': [tenure],
            'phoneservice': [PhoneService],
            'multiplelines': [MultipleLines], 
            'internetservice': [InternetService],
            'onlinesecurity': [OnlineSecurity],
            'onlinebackup': [OnlineBackup], 
            'deviceprotection': [DeviceProtection], 
            'techsupport': [TechSupport], 
            'streamingtv': [StreamingTV],
            'streamingmovies': [StreamingMovies], 
            'contract': [Contract], 
            'paperlessbilling': [PaperlessBilling], 
            'paymentmethod': [PaymentMethod],
            'monthlycharges': [MonthlyCharges], 
            'totalcharges': [TotalCharges]
        })
        st.session_state['df'] = input_features
        st.form_submit_button('Submit', on_click=make_prediction, kwargs=dict(model=model, encoder=encoder))
   
def save_prediction_to_csv(df, prediction, model_name):
    churn_label = "Churn" if prediction[0] == 1 else "Not Churn"
    
    # Concatenate input features, model name, and churn label
    prediction_df = pd.DataFrame({
        'Gender': df['gender'],
        'SeniorCitizen': df['SeniorCitizen'],
        'Partner': df['Partner'],
        'Dependents': df['Dependents'],
        'Tenure': df['tenure'],
        'PhoneService': df['PhoneService'],
        'MultipleLines': df['MultipleLines'],
        'InternetService': df['InternetService'],
        'OnlineSecurity': df['OnlineSecurity'],
        'OnlineBackup': df['OnlineBackup'],
        'DeviceProtection': df['DeviceProtection'],
        'TechSupport': df['TechSupport'],
        'StreamingTV': df['StreamingTV'],
        'StreamingMovies': df['StreamingMovies'],
        'Contract': df['Contract'],
        'PaperlessBilling': df['PaperlessBilling'],
        'PaymentMethod': df['PaymentMethod'],
        'MonthlyCharges': df['MonthlyCharges'],
        'TotalCharges': df['TotalCharges'],
        'Model': model_name,
        'Churn': churn_label
    })
    
    # Save to CSV file
    prediction_df.to_csv('data/history.csv', mode='a', header=not os.path.exists('data/history.csv'), index=False)


# Call the data function directly
if __name__ == '__main__':
    st.title('Make a Prediction')
    predict()

    # Ensure 'prediction' and 'probability' are initialized in session_state
    if 'prediction' not in st.session_state:
        st.session_state['prediction'] = None

    if 'probability' not in st.session_state:
        st.session_state['probability'] = None

    # Retrieve 'prediction' and 'probability' from session_state
    prediction = st.session_state['prediction']
    probability = st.session_state['probability']

    # Display prediction and probability
    if prediction is None:
        st.markdown("### Prediction will show here")
    elif prediction == "Yes":
        probability_of_yes = probability[0][1] * 100
        st.markdown(f"### The employee will leave the company with a probability of {probability_of_yes:.2f}%")
    else:
        probability_of_no = probability[0][0] * 100
        st.markdown(f"### Employee will not leave with a probability of {probability_of_no:.2f}%")
