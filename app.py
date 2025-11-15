from autots import AutoTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import regression
import joblib
import os
sns.set()
plt.style.use('seaborn-v0_8-whitegrid')

import streamlit as st
st.title("Future Forex Currency Price Prediction Model")

options = {
    'AUSTRALIAN DOLLAR': 'AUSTRALIA - AUSTRALIAN DOLLAR/US$',
    'EURO': 'EURO AREA - EURO/US$',
    'NEW ZEALAND DOLLAR': 'NEW ZEALAND - NEW ZEALAND DOLLAR/US$',
    'GREAT BRITAIN POUNDS': 'UNITED KINGDOM - UNITED KINGDOM POUND/US$',
    'BRAZILIAN REAL': 'BRAZIL - REAL/US$',
    'CANADIAN DOLLAR': 'CANADA - CANADIAN DOLLAR/US$',
    'CHINESE YUAN': 'CHINA - YUAN/US$',
    'HONG KONG DOLLAR': 'HONG KONG - HONG KONG DOLLAR/US$',
    'INDIAN RUPEE': 'INDIA - INDIAN RUPEE/US$',
    'KOREAN WON': 'KOREA - WON/US$',
    'MEXICAN PESO': 'MEXICO - MEXICAN PESO/US$',
    'SOUTH AFRICAN RAND': 'SOUTH AFRICA - RAND/US$',
    'SINGAPORE DOLLAR': 'SINGAPORE - SINGAPORE DOLLAR/US$',
    'DANISH KRONE': 'DENMARK - DANISH KRONE/US$',
    'JAPANESE YEN': 'JAPAN - YEN/US$',
    'MALAYSIAN RINGGIT': 'MALAYSIA - RINGGIT/US$',
    'NORWEGIAN KRONE': 'NORWAY - NORWEGIAN KRONE/US$',
    'SWEDEN KRONA': 'SWEDEN - KRONA/US$',
    'SRILANKAN RUPEE': 'SRI LANKA - SRI LANKAN RUPEE/US$',
    'SWISS FRANC': 'SWITZERLAND - FRANC/US$',
    'NEW TAIWAN DOLLAR': 'TAIWAN - NEW TAIWAN DOLLAR/US$',
    'THAI BAHT': 'THAILAND - BAHT/US$'
}

def get_model_filename(currency_name):
    return f"models/{currency_name.replace(' ', '_').replace('$', '')}_model.pkl"

def load_currency_model(currency_name):
    model_filename = get_model_filename(currency_name)
    
    if os.path.exists(model_filename):
        try:
            model_info = joblib.load(model_filename)
            st.success(f"Loaded pre-trained model for {currency_name}")
            return model_info
        except Exception as e:
            st.error(f"Error loading model for {currency_name}: {str(e)}")
            return None
    else:
        st.warning(f"Training new model for {currency_name} (this may take a while)...")
        return None

def train_new_model(selected_option, forecast_days):
    try:
        data = pd.read_csv("data/Foreign_Exchange_Rates.csv")
        data.dropna(inplace=True)
        data['Time Serie'] = pd.to_datetime(data['Time Serie'], format='%d-%m-%Y')
        
        model = AutoTS(
            forecast_length=int(forecast_days), 
            frequency='infer', 
            ensemble='simple', 
            drop_data_older_than_periods=200
        )
        model = model.fit(
            data, 
            date_col='Time Serie', 
            value_col=options[selected_option], 
            id_col=None
        )
        
        model_info = {
            'model': model,
            'currency_name': selected_option,
            'trained_date': pd.Timestamp.now()
        }
        os.makedirs('models', exist_ok=True)
        joblib.dump(model_info, get_model_filename(selected_option))
        
        st.success(f"Model trained and saved for {selected_option}")
        return model
        
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None

def make_forecast(selected_option, forecast_days):
    model_info = load_currency_model(selected_option)
    
    if model_info is not None:
        model = model_info['model']
        model.forecast_length = int(forecast_days)
    else:
        # Train a new model
        model = train_new_model(selected_option, forecast_days)
        if model is None:
            return None
    
    # Make prediction
    try:
        prediction = model.predict()
        forecast = prediction.forecast
        return forecast
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

# Create models directory
os.makedirs('models', exist_ok=True)

# Simple main interface
st.write("Select a currency and forecast period to generate predictions.")

selected_option = st.selectbox('Choose a currency:', list(options.keys()))
forecast_days = st.number_input(
    "Forecast Days:",
    min_value=1,
    max_value=100,
    value=30,
    step=1,
    help="Number of days to forecast into the future"
)

if st.button('Generate Predictions'):
    with st.spinner(f"Generating {forecast_days}-day forecast for {selected_option}..."):
        try:
            forecast = make_forecast(selected_option, forecast_days)
            
            if forecast is not None and len(forecast) > 0:
                st.success("Forecast generated successfully!")
                
                # Display results
                st.subheader("Forecast Chart")
                st.line_chart(forecast)
                
                st.subheader("Forecast Data")
                st.dataframe(forecast)

            else:
                st.error("Failed to generate forecast. Please check if the data file exists.")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Make sure your data file exists at 'data/Foreign_Exchange_Rates.csv'")
