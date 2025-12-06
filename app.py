from autots import AutoTS
import numpy as np
import pandas as pd
import joblib
import os

import streamlit as st

st.title("Future Forex Currency Price Prediction Model")
st.write(
    "Select a currency and forecast period to generate predictions "
    "from the pre-trained models."
)

def autots_model_debug(*args, **kwargs):
    return None

options = {
    'AUSTRALIAN DOLLAR': 'AUSTRALIA - AUSTRALIAN DOLLAR/US$',
    'EURO': 'EURO AREA - EURO/US$',
#    'NEW ZEALAND DOLLAR': 'NEW ZEALAND - NEW ZELAND DOLLAR/US$',
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
    #'NEW TAIWAN DOLLAR': 'TAIWAN - NEW TAIWAN DOLLAR/US$',
    #'THAI BAHT': 'THAILAND - BAHT/US$'
}

CURRENCY_TO_MODEL_FILE = {
    'AUSTRALIAN DOLLAR': 'AUSTRALIAN_DOLLAR_ARIMA_model.pkl',
    'BRAZILIAN REAL': 'BRAZILIAN_REAL_AutoTS_model.pkl',
    'CANADIAN DOLLAR': 'CANADIAN_DOLLAR_ARIMA_model.pkl',
    'CHINESE YUAN': 'CHINESE_YUAN$_ARIMA_model.pkl',
    'DANISH KRONE': 'DANISH_KRONE_ARIMA_model.pkl',
    'EURO': 'EURO_ARIMA_model.pkl',
    'GREAT BRITAIN POUNDS': 'GREAT_BRITAIN_POUNDS_SARIMA_model.pkl',
    'HONG KONG DOLLAR': 'HONG_KONG_DOLLAR_SARIMA_model.pkl',
    'INDIAN RUPEE': 'INDIAN_RUPEE_AutoTS_model.pkl',
    'KOREAN WON': 'KOREAN_WON$_ARIMA_model.pkl',
    'MALAYSIAN RINGGIT': 'MALAYSIAN_RINGGIT_ARIMA_model.pkl',
    'MEXICAN PESO': 'MEXICAN_PESO_AutoTS_model.pkl',
    #'NEW TAIWAN DOLLAR': 'NEW_TAIWAN_DOLLAR_SARIMA_model.pkl',
    'JAPANESE YEN': 'JAPANESE_YEN$_SARIMA_model.pkl',
    'NORWEGIAN KRONE': 'NORWEGIAN_KRONE_ARIMA_model.pkl',
    'SINGAPORE DOLLAR': 'SINGAPORE_DOLLAR_SARIMA_model.pkl',
    'SOUTH AFRICAN RAND': 'SOUTH_AFRICAN_RAND$_ARIMA_model.pkl',
    'SRILANKAN RUPEE': 'SRILANKAN_RUPEE_SARIMA_model.pkl',
    'SWEDEN KRONA': 'SWEDEN_KRONA_AutoTS_model.pkl',
    'SWISS FRANC': 'SWISS_FRANC_SARIMA_model.pkl',
    #'THAI BAHT': 'THAI_BAHT_SARIMA_model.pkl',
    # add NEW ZEALAND DOLLAR / JAPANESE YEN etc
}

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def load_saved_model(currency_name: str):
    if currency_name not in CURRENCY_TO_MODEL_FILE:
        st.error(f"No model file mapping defined for '{currency_name}'.")
        return None

    filename = CURRENCY_TO_MODEL_FILE[currency_name]
    filepath = os.path.join(MODELS_DIR, filename)

    if not os.path.exists(filepath):
        st.error(f"Model file not found for {currency_name}: '{filepath}'")
        return None

    try:
        loaded = joblib.load(filepath)
    except Exception as e:
        st.error(f"Error loading model file for {currency_name}: {e}")
        return None

    st.success(f"Loaded existing model file for {currency_name}")
    if isinstance(loaded, dict):
        st.info(
            f"Loaded object for {currency_name} "
            f"is a dict with keys {list(loaded.keys())}."
        )

    return loaded


def _find_model_object(obj, _visited=None):
    if _visited is None:
        _visited = set()

    oid = id(obj)
    if oid in _visited:
        return None
    _visited.add(oid)

    if hasattr(obj, "predict") or hasattr(obj, "forecast"):
        return obj

    if isinstance(obj, dict):
        for v in obj.values():
            found = _find_model_object(v, _visited)
            if found is not None:
                return found

    if isinstance(obj, (list, tuple, set)):
        for v in obj:
            found = _find_model_object(v, _visited)
            if found is not None:
                return found

    return None


def forecast_from_model_container(
    container,
    forecast_days: int,
    display_name: str
) -> pd.DataFrame | None:

    model_obj = _find_model_object(container)

    if model_obj is None:
        st.error(
            "Could not find a model object with .predict() or .forecast() "
            "inside the saved file."
        )
        return None

    module = type(model_obj).__module__.lower()
    cls_name = type(model_obj).__name__
    st.info(f"Using model class: {cls_name} (module: {module})")

    steps = int(forecast_days)

    if "autots" in module or isinstance(model_obj, AutoTS):
        try:
            prediction = model_obj.predict(forecast_length=steps)
            fc = prediction.forecast
            if len(fc) > steps:
                fc = fc.iloc[:steps]
            return fc
        except Exception as e:
            st.error(f"AutoTS prediction error: {e}")
            return None

    if "statsmodels" in module:
        try:
            if hasattr(model_obj, "get_forecast"):
                fc = model_obj.get_forecast(steps=steps).predicted_mean
            elif hasattr(model_obj, "forecast"):
                fc = model_obj.forecast(steps=steps)
            else:
                st.error(
                    "Statsmodels model has neither get_forecast nor forecast."
                )
                return None

            fc = pd.Series(fc, name=display_name)
            return fc.to_frame()
        except Exception as e:
            st.error(f"ARIMA/SARIMA prediction error: {e}")
            return None

    try:
        try:
            yhat = model_obj.predict(steps)
        except TypeError:
            # e.g. pmdarima: predict(n_periods=steps)
            yhat = model_obj.predict(n_periods=steps)

        fc = pd.Series(yhat, name=display_name).to_frame()
        return fc
    except Exception as e:
        st.error(
            f"Don't know how to call predict() on model type {cls_name}: {e}"
        )
        return None


def make_forecast(selected_option: str, forecast_days: int):
    container = load_saved_model(selected_option)
    if container is None:
        return None

    display_name = options[selected_option]
    return forecast_from_model_container(container, forecast_days, display_name)


selected_option = st.selectbox("Choose a currency:", list(options.keys()))

forecast_days = st.number_input(
    "Forecast Days:",
    min_value=1,
    max_value=100,
    value=30,
    step=1,
    help="Number of days to forecast into the future",
)

if st.button("Generate Predictions"):
    with st.spinner(
        f"Generating {forecast_days}-day forecast for {selected_option}..."
    ):
        forecast = make_forecast(selected_option, int(forecast_days))

        if forecast is not None and len(forecast) > 0:
            st.success("Forecast generated successfully!")

            st.subheader("Forecast Chart")
            st.line_chart(forecast)

            st.subheader("Forecast Data")
            st.dataframe(forecast)
        else:
            st.error(
                "Failed to generate forecast from the saved model."
            )
