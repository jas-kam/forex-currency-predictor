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
            st.sidebar.success(f"Loaded pre-trained model for {currency_name}")
            return model_info
        except Exception as e:
            st.sidebar.error(f"Error loading model for {currency_name}: {str(e)}")
            return None
    else:
        st.sidebar.warning(f"No pre-trained model found for {currency_name}")
        return None

def train_new_model(selected_option, forecast_days):
    try:
        st.info(f"Training new model for {selected_option}...")
        
        # Load and prepare data
        data = pd.read_csv("data/Foreign_Exchange_Rates.csv")  # Changed to .csv
        data.dropna(inplace=True)
        data['Time Serie'] = pd.to_datetime(data['Time Serie'], format='%d-%m-%Y')
        
        # Train model
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
        
        # Save the model for future use
        model_info = {
            'model': model,
            'currency_name': selected_option,
            'trained_date': pd.Timestamp.now()
        }
        os.makedirs('models', exist_ok=True)
        joblib.dump(model_info, get_model_filename(selected_option))
        
        st.sidebar.success(f"✓ Model trained and saved for {selected_option}")
        return model
        
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None

def make_forecast(selected_option, forecast_days):
    """Make forecasts - tries pre-trained first, falls back to training"""
    # Try to load pre-trained model
    model_info = load_currency_model(selected_option)
    
    if model_info is not None:
        model = model_info['model']
        # Update forecast length for the current prediction
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

# Main app
st.write("Select a currency and forecast period to generate predictions.")

with st.form(key='user_form'):
    selected_option = st.selectbox('Choose a currency:', list(options.keys()))
    forecast_days = st.number_input(
        "Forecast Days:",
        min_value=1,
        max_value=100,
        value=30,
        step=1,
        help="Number of days to forecast into the future"
    )
    submit_button = st.form_submit_button(label='Generate Predictions')

if submit_button:
    with st.spinner(f"Generating {forecast_days}-day forecast for {selected_option}..."):
        try:
            forecast = make_forecast(selected_option, forecast_days)
            
            if forecast is not None and len(forecast) > 0:
                st.success("✅ Forecast generated successfully!")
                
                # Display results
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("Forecast Chart")
                    st.line_chart(forecast)
                
                with col2:
                    st.subheader("Forecast Data")
                    st.dataframe(forecast)
                
                # Download button
                csv = forecast.to_csv()
                st.download_button(
                    label="Download Forecast as CSV",
                    data=csv,
                    file_name=f"{selected_option}_forecast_{forecast_days}_days.csv",
                    mime="text/csv"
                )
            else:
                st.error("Failed to generate forecast. Please check if the data file exists.")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Make sure your data file exists at 'data/Foreign_Exchange_Rates.csv'")

# Footer
st.markdown("---")
st.markdown("**Tip**: Use the sidebar to check which models are pre-trained for faster predictions.")
