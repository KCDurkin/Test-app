import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy import stats
from fredapi import Fred
from py_vollib.black_scholes.implied_volatility import implied_volatility
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Constants
FRED_API_KEY = '27982b8033b09fa32648865ef7ee3978'

# [Previous StockPredictorYF class and other functions remain the same]

def main():
    st.set_page_config(page_title="Stock Analysis & Prediction", layout="wide")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a Page", ["Price Prediction", "Stock Analysis"])
    
    # Common stock input in sidebar
    st.sidebar.header("Stock Selection")
    ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL)", "").upper()
    
    # Date range selection in sidebar
    st.sidebar.header("Date Range")
    start_date = st.sidebar.date_input(
        "Start date",
        value=datetime.now() - timedelta(days=365)
    )
    end_date = st.sidebar.date_input(
        "End date",
        value=datetime.now()
    )
    
    if page == "Price Prediction":
        prediction_page(ticker, start_date, end_date)
    else:
        analysis_page(ticker, start_date, end_date)

def prediction_page(ticker, start_date, end_date):
    st.title("Stock Price Prediction")
    st.write("""
    This page predicts stock prices using an ensemble of machine learning models including LSTM, 
    Gradient Boosting, Random Forest, and SVR.
    """)
    
    prediction_days = st.slider("Number of days to predict", 7, 30, 30)
    
    if st.button("Predict"):
        if not ticker:
            st.warning("Please enter a stock ticker in the sidebar.")
            return
            
        try:
            with st.spinner(f"Processing {ticker}..."):
                predictor = StockPredictorYF()
                
                # Prepare data
                progress_bar = st.progress(0)
                st.write("Fetching data and calculating indicators...")
                df = predictor.prepare_features(ticker, start_date, end_date)
                progress_bar.progress(25)
                
                if len(df) == 0:
                    st.warning(f"No data found for {ticker}")
                    return
                    
                X, y, dates = predictor.prepare_data(df)
                progress_bar.progress(50)
                
                st.write("Training models...")
                predictions, actual, split = predictor.train_and_predict(X, y)
                progress_bar.progress(75)
                
                st.write("Generating future predictions...")
                future_predictions = predictor.predict_future(days=prediction_days)
                progress_bar.progress(100)
                
                # Display results
                display_prediction_results(predictions, actual, split, dates, future_predictions, prediction_days, ticker)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please check if the ticker is valid.")

def analysis_page(ticker, start_date, end_date):
    st.title("Stock Analysis Dashboard")
    
    if not ticker:
        st.warning("Please enter a stock ticker in the sidebar.")
        return
        
    try:
        # Add analysis options
        st.subheader("Analysis Options")
        col1, col2 = st.columns(2)
        
        with col1:
            show_volatility = st.checkbox("Show Volatility Analysis", True)
            show_sharpe = st.checkbox("Show Sharpe Ratio", True)
        
        with col2:
            show_returns = st.checkbox("Show Returns Analysis", True)
            show_distribution = st.checkbox("Show Distribution Analysis", True)
        
        # Perform analysis
        with st.spinner(f"Analyzing {ticker}..."):
            stock = yf.Ticker(ticker)
            stock_data = stock.history(start=start_date, end=end_date)
            
            if stock_data.empty:
                st.warning(f"No data available for {ticker}")
                return
            
            # Display basic stock info
            info = stock.info
            st.subheader("Company Information")
            if 'longName' in info:
                st.write(f"Company: {info['longName']}")
            if 'sector' in info:
                st.write(f"Sector: {info['sector']}")
            if 'industry' in info:
                st.write(f"Industry: {info['industry']}")
            
            # Show selected analyses
            if show_volatility or show_sharpe:
                risk_free_rate = get_india_10y_yield()
                sharpe, annual_return, annual_vol = calculate_sharpe_ratio(
                    stock_data,
                    risk_free_rate=risk_free_rate
                )
                
                if show_sharpe:
                    st.subheader("Risk and Return Metrics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                    with col2:
                        st.metric("Annual Return", f"{annual_return*100:.2f}%")
                    with col3:
                        st.metric("Annual Volatility", f"{annual_vol*100:.2f}%")
            
            if show_volatility:
                st.subheader("Volatility Analysis")
                hist_vol = calculate_historical_volatility(stock_data)
            
            if show_returns:
                st.subheader("Returns Analysis")
                returns = stock_data['Close'].pct_change().dropna()
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(returns.index, returns.cumsum(), label='Cumulative Returns')
                plt.title(f"{ticker} Cumulative Returns")
                plt.xlabel('Date')
                plt.ylabel('Cumulative Return')
                plt.legend()
                st.pyplot(fig)
            
            if show_distribution:
                st.subheader("Returns Distribution")
                fig, ax = plt.subplots(figsize=(10, 6))
                returns.hist(bins=50, ax=ax)
                plt.title(f"{ticker} Returns Distribution")
                plt.xlabel('Returns')
                plt.ylabel('Frequency')
                st.pyplot(fig)
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please check if the ticker is valid.")

# [Previous display_prediction_results function remains the same]

if __name__ == "__main__":
    main()
