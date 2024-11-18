# Imports and Basic Setup
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from fredapi import Fred
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
import logging
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class StockAnalysisConfig:
    """Configuration class for stock analysis parameters"""
    fred_api_key: str
    default_risk_free_rate: float = 0.07
    volatility_window: int = 30
    trading_days_per_year: int = 252
    default_period: str = '1y'
    risk_free_rate_source: str = 'INDIRLTLT01STM'

class MarketDataProvider:
    """Class to handle market data retrieval"""
    def __init__(self, config: StockAnalysisConfig):
        self.config = config
        self._fred = None

    @property
    def fred(self) -> Fred:
        """Lazy initialization of FRED client"""
        if self._fred is None:
            self._fred = Fred(api_key=self.config.fred_api_key)
        return self._fred

    def get_risk_free_rate(self) -> float:
        """Fetch current risk-free rate"""
        try:
            # Get the appropriate yield based on selected source
            if self.config.risk_free_rate_source == 'INDIRLTLT01STM':
                yield_data = self.fred.get_series('INDIRLTLT01STM')
                logger.info("Using India risk-free rate")
            else:
                yield_data = self.fred.get_series('IRLTLT01USM156N')
                logger.info("Using USA risk-free rate")
            return float(yield_data.iloc[-1]) / 100
        except Exception as e:
            logger.warning(f"Failed to fetch risk-free rate: {e}")
            return self.config.default_risk_free_rate

    def get_stock_data(self, ticker: str, period: str) -> Optional[pd.DataFrame]:
        """Fetch stock data with error handling"""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            if data.empty:
                logger.warning(f"No data available for {ticker}")
                return None
            return data
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return None


class StockAnalyzer:
    """Main class for stock analysis"""

    def __init__(self, config: StockAnalysisConfig):
        self.config = config
        self.market_data = MarketDataProvider(config)

    def calculate_returns_metrics(self, price_data: pd.Series) -> Tuple[float, float, float]:
        """Calculate annualized return metrics"""
        returns = price_data.pct_change().dropna()
        annual_return = returns.mean() * self.config.trading_days_per_year
        annual_volatility = returns.std() * np.sqrt(self.config.trading_days_per_year)
        risk_free_rate = self.market_data.get_risk_free_rate()
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
        return sharpe_ratio, annual_return, annual_volatility

    def calculate_volatility(self, price_data: pd.Series) -> Tuple[pd.Series, plt.Figure]:
        """Calculate historical volatility and create distribution plot"""
        returns = price_data.pct_change().dropna()
        hist_volatility = returns.rolling(window=self.config.volatility_window).std() * \
                          np.sqrt(self.config.trading_days_per_year)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # Returns distribution
        returns.hist(bins=50, density=True, alpha=0.7, ax=ax1)
        mu, sigma = stats.norm.fit(returns)
        x = np.linspace(returns.min(), returns.max(), 100)
        ax1.plot(x, norm.pdf(x, mu, sigma), 'r-', lw=2, label='Normal Fit')
        ax1.set_title('Returns Distribution')
        ax1.set_xlabel('Returns')
        ax1.set_ylabel('Frequency')
        ax1.legend()

        # Volatility over time
        hist_volatility.plot(ax=ax2)
        ax2.set_title('Historical Volatility')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Volatility')

        # Historical Volatility Distribution
        hist_volatility.dropna().hist(bins=50, density=True, alpha=0.7, ax=ax3)
        mu_vol, sigma_vol = stats.norm.fit(hist_volatility.dropna())
        x_vol = np.linspace(hist_volatility.min(), hist_volatility.max(), 100)
        ax3.plot(x_vol, norm.pdf(x_vol, mu_vol, sigma_vol), 'r-', lw=2, label='Normal Fit')
        ax3.set_title('Historical Volatility Distribution')
        ax3.set_xlabel('Volatility')
        ax3.set_ylabel('Frequency')
        ax3.legend()

        plt.tight_layout()
        return hist_volatility, fig

    def analyze_stock(self, ticker: str, include_options: bool = False) -> Dict[str, Union[float, pd.Series]]:
        """Comprehensive stock analysis"""
        data = self.market_data.get_stock_data(ticker, self.config.default_period)
        if data is None:
            return {}

        price_data = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
        sharpe, annual_return, annual_vol = self.calculate_returns_metrics(price_data)
        hist_vol, vol_plot = self.calculate_volatility(price_data)

        analysis = {
            'current_price': price_data.iloc[-1],
            'sharpe_ratio': sharpe,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'historical_volatility': hist_vol,
            'risk_free_rate': self.market_data.get_risk_free_rate(),
            'volatility_plot': vol_plot
        }

        if include_options:
            stock = yf.Ticker(ticker)
            options = stock.options
            if options:
                nearest_expiry = options[0]
                chain = stock.option_chain(nearest_expiry)
                analysis['options'] = {
                    'nearest_expiry': nearest_expiry,
                    'call_volume': chain.calls['volume'].sum(),
                    'put_volume': chain.puts['volume'].sum(),
                    'put_call_ratio': chain.puts['volume'].sum() / chain.calls['volume'].sum()
                }

        return analysis


class StockPredictorYF:
    def __init__(self, window_size=30, epochs=50, batch_size=32):
        self.window_size = window_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.last_sequence = None
        self.base_models = {
            'LSTM': None,
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1)
        }
        self.meta_model = LinearRegression()

    def calculate_technical_indicators(self, df):
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # SMA and EMA
        df['SMA'] = df['Close'].rolling(window=20).mean()
        df['EMA'] = df['Close'].ewm(span=20, adjust=False).mean()

        # Bollinger Bands
        sma = df['Close'].rolling(window=20).mean()
        std = df['Close'].rolling(window=20).std()
        df['Upper Band'] = sma + (std * 2)
        df['Lower Band'] = sma - (std * 2)

        # Momentum and ATR
        df['Momentum'] = df['Close'].diff(10)

        high = df['High']
        low = df['Low']
        close = df['Close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.DataFrame({'TR1': tr1, 'TR2': tr2, 'TR3': tr3}).max(axis=1)
        df['ATR'] = tr.rolling(window=14).mean()

        return df

    def prepare_features(self, ticker, start_date, end_date):
        # Get stock data
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        if df.empty:
            return pd.DataFrame()

        df = self.calculate_technical_indicators(df)
        feature_columns = ['Close', 'RSI', 'SMA', 'EMA', 'Upper Band',
                           'Lower Band', 'Momentum', 'ATR']
        return df[feature_columns].fillna(method='ffill').fillna(method='bfill')

    def prepare_data(self, df):
        scaled_data = self.scaler.fit_transform(df)
        self.last_sequence = scaled_data[-self.window_size:]

        X, y = [], []
        for i in range(len(scaled_data) - self.window_size):
            X.append(scaled_data[i:(i + self.window_size)])
            y.append(scaled_data[i + self.window_size, 0])

        return np.array(X), np.array(y), df.index[self.window_size:]

    def build_lstm_model(self, input_shape):
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def train_base_models(self, X_train, y_train, X_test):
        predictions = {}

        # Train LSTM
        self.base_models['LSTM'] = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
        self.base_models['LSTM'].fit(X_train, y_train,
                                     epochs=self.epochs,
                                     batch_size=self.batch_size,
                                     verbose=0)
        lstm_predictions = self.base_models['LSTM'].predict(X_test).flatten()

        dummy_array = np.zeros((len(lstm_predictions), self.scaler.n_features_in_))
        dummy_array[:, 0] = lstm_predictions
        predictions['LSTM'] = self.scaler.inverse_transform(dummy_array)[:, 0]

        # Train other models
        X_train_2d = X_train.reshape((X_train.shape[0], -1))
        X_test_2d = X_test.reshape((X_test.shape[0], -1))

        for name, model in self.base_models.items():
            if name != 'LSTM':
                model.fit(X_train_2d, y_train)
                pred = model.predict(X_test_2d)
                dummy_array = np.zeros((len(pred), self.scaler.n_features_in_))
                dummy_array[:, 0] = pred
                predictions[name] = self.scaler.inverse_transform(dummy_array)[:, 0]

        return predictions

    def train_and_predict(self, X, y):
        split = int(len(X) * 0.9)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        base_predictions = self.train_base_models(X_train, y_train, X_test)
        meta_features = np.column_stack([base_predictions[name] for name in base_predictions])

        dummy_array = np.zeros((len(y_test), self.scaler.n_features_in_))
        dummy_array[:, 0] = y_test
        actual = self.scaler.inverse_transform(dummy_array)[:, 0]

        self.meta_model.fit(meta_features, actual)
        meta_predictions = self.meta_model.predict(meta_features)

        return meta_predictions.reshape(-1, 1), actual.reshape(-1, 1), split

    def predict_future(self, days=30):
        if self.meta_model is None:
            raise ValueError("Model must be trained before making predictions")

        future_predictions = []
        current_sequence = self.last_sequence.copy()

        for _ in range(days):
            base_preds = []
            for name, model in self.base_models.items():
                if name == 'LSTM':
                    pred = model.predict(current_sequence.reshape(1, self.window_size, -1), verbose=0)[0, 0]
                else:
                    pred = model.predict(current_sequence.reshape(1, -1))[0]

                dummy_array = np.zeros((1, self.scaler.n_features_in_))
                dummy_array[0, 0] = pred
                scaled_pred = self.scaler.inverse_transform(dummy_array)[0, 0]
                base_preds.append(scaled_pred)

            next_pred = self.meta_model.predict([base_preds])[0]
            future_predictions.append(next_pred)

            dummy_array = np.zeros((1, self.scaler.n_features_in_))
            dummy_array[0, 0] = next_pred
            scaled_next = self.scaler.transform(dummy_array)[0, 0]
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1, 0] = scaled_next

        return np.array(future_predictions)

    def evaluate_model(self, actual, predictions):
        mse = mean_squared_error(actual, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual, predictions)
        mape = np.mean(np.abs((actual - predictions) / actual)) * 100

        return {
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }


# Options Analysis Functions
def d1_d2(S, E, T, rf, sigma):
    d1 = (np.log(S / E) + (rf + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def call_option_price(S, E, T, rf, sigma):
    d1, d2 = d1_d2(S, E, T, rf, sigma)
    return S * stats.norm.cdf(d1) - E * np.exp(-rf * T) * stats.norm.cdf(d2)


def put_option_price(S, E, T, rf, sigma):
    d1, d2 = d1_d2(S, E, T, rf, sigma)
    return E * np.exp(-rf * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)


def implied_volatility(option_price, S, E, T, rf, option_type='call'):
    objective_function = lambda sigma: (
                                           call_option_price(S, E, T, rf, sigma) - option_price
                                           if option_type == 'call' else
                                           put_option_price(S, E, T, rf, sigma) - option_price
                                       ) ** 2
    result = minimize(objective_function, 0.2, bounds=[(1e-5, 3)])
    return result.x[0]


def main():
    st.set_page_config(page_title="Stock & Options Analysis", layout="wide")

    # Add country selection for risk-free rate
    country = st.sidebar.radio(
        "Select Country for Risk-Free Rate",
        ["India", "USA"],
        help="Choose the country whose government bond yield will be used as the risk-free rate"
    )

    # Set the appropriate FRED series based on selection
    risk_free_rate_source = 'INDIRLTLT01STM' if country == 'India' else 'IRLTLT01USM156N'

    config = StockAnalysisConfig(
        fred_api_key='27982b8033b09fa32648865ef7ee3978',
        default_risk_free_rate=0.07,
        volatility_window=30,
        risk_free_rate_source=risk_free_rate_source
    )

    analyzer = StockAnalyzer(config)
    predictor = StockPredictorYF()

    st.title("Advanced Stock & Options Analysis")
    st.sidebar.info(f"Using {country}'s government bond yield as risk-free rate")

    tabs = st.tabs(["Stock Analysis & Prediction", "Options Analysis"])

    with tabs[0]:
        st.write("""
        This section predicts stock prices using a combination of financial analysis and an ensemble of machine learning models.
        """)

        col1, col2 = st.columns(2)

        with col1:
            ticker = st.text_input("Enter a stock ticker (e.g., AAPL)").upper()
            prediction_days = st.slider("Number of days to predict", 7, 30, 30)

        with col2:
            start_date = st.date_input(
                "Start date",
                value=datetime.now() - timedelta(days=365)
            )
            end_date = st.date_input(
                "End date",
                value=datetime.now()
            )

        if st.button("Analyze Stock"):
            if not ticker:
                st.warning("Please enter a stock ticker.")
                return

            try:
                with st.spinner(f"Processing {ticker}..."):
                    # Stock Analysis
                    st.write("Financial Analysis...")
                    analysis = analyzer.analyze_stock(ticker, include_options=True)
                    if not analysis:
                        st.warning(f"No data found for {ticker}")
                        return

                    # Display basic metrics
                    metrics_col1, metrics_col2 = st.columns(2)
                    with metrics_col1:
                        st.write(f"Current Price: ₹{analysis['current_price']:.2f}")
                        st.write(f"Sharpe Ratio: {analysis['sharpe_ratio']:.2f}")
                        st.write(f"Annual Return: {analysis['annual_return'] * 100:.2f}%")
                    with metrics_col2:
                        st.write(f"Annual Volatility: {analysis['annual_volatility'] * 100:.2f}%")
                        st.write(f"Risk-Free Rate: {analysis['risk_free_rate'] * 100:.2f}%")

                    st.pyplot(analysis['volatility_plot'])

                    if 'options' in analysis:
                        st.write("Options Analysis:")
                        st.write(f"Nearest Expiry: {analysis['options']['nearest_expiry']}")
                        st.write(f"Put/Call Ratio: {analysis['options']['put_call_ratio']:.2f}")

                    # Prediction Section
                    st.write("### Price Predictions")
                    df = predictor.prepare_features(ticker, start_date, end_date)
                    if len(df) == 0:
                        st.warning(f"No data found for {ticker}")
                        return

                    X, y, dates = predictor.prepare_data(df)
                    predictions, actual, split = predictor.train_and_predict(X, y)
                    future_predictions = predictor.predict_future(days=prediction_days)

                    # Display model evaluation metrics
                    metrics = predictor.evaluate_model(actual, predictions)
                    met_col1, met_col2, met_col3, met_col4 = st.columns(4)
                    with met_col1:
                        st.metric("MSE", f"{metrics['MSE']:.4f}")
                    with met_col2:
                        st.metric("RMSE", f"{metrics['RMSE']:.4f}")
                    with met_col3:
                        st.metric("R² Score", f"{metrics['R2']:.4f}")
                    with met_col4:
                        st.metric("MAPE", f"{metrics['MAPE']:.2f}%")

                    # Plot predictions
                    test_dates = dates[split:]
                    fig_pred = plt.figure(figsize=(15, 7))
                    plt.plot(test_dates, actual, label='Actual', color='blue', linewidth=2)
                    plt.plot(test_dates, predictions, label='Predictions', color='red', linewidth=2)

                    last_date = test_dates[-1]
                    future_dates = [last_date + timedelta(days=i + 1) for i in range(len(future_predictions))]
                    plt.plot(future_dates, future_predictions, label='Future Predictions',
                             color='green', linestyle='--', linewidth=2)

                    plt.title(f"{ticker} Stock Price Prediction", fontsize=14, pad=20)
                    plt.xlabel('Date', fontsize=12)
                    plt.ylabel('Price', fontsize=12)
                    plt.legend(fontsize=10)
                    plt.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                    st.pyplot(fig_pred)

                    # Display future predictions table
                    st.write(f"\nPredicted prices for next {prediction_days} days:")
                    future_df = pd.DataFrame({
                        'Date': [date.strftime('%Y-%m-%d') for date in future_dates],
                        'Predicted Price': [f"₹{price:.2f}" for price in future_predictions]
                    })
                    st.dataframe(future_df)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.write("Please check if the ticker and dates are valid.")

    with tabs[1]:
        st.write("""
        This section calculates implied volatility and provides options analysis tools.
        """)

        col1, col2 = st.columns(2)

        with col1:
            S_iv = st.number_input("Current Asset Price", value=100.0)
            E_iv = st.number_input("Strike Price", value=100.0)
            T_iv = st.number_input("Time to Maturity (Years)", value=1.0)
            rf_iv = st.number_input("Risk-Free Interest Rate", value=0.05)
            market_price = st.number_input("Market Option Price", value=10.0)
            option_type_iv = st.selectbox("Option Type", ["Call", "Put"])

        with col2:
            if st.button("Calculate"):
                iv = implied_volatility(market_price, S_iv, E_iv, T_iv, rf_iv, option_type_iv.lower())
                st.info(f"Implied Volatility: {iv:.2%}")

                theo_price = (
                    call_option_price(S_iv, E_iv, T_iv, rf_iv, iv)
                    if option_type_iv.lower() == 'call'
                    else put_option_price(S_iv, E_iv, T_iv, rf_iv, iv)
                )
                st.info(f"Theoretical Option Price: {theo_price:.2f}")

                # Create visualization
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                # IV vs Stock Price
                spot_range = np.linspace(S_iv * 0.7, S_iv * 1.3, 50)
                ivs = [implied_volatility(market_price, s, E_iv, T_iv, rf_iv, option_type_iv.lower())
                       for s in spot_range]

                ax1.plot(spot_range, ivs)
                ax1.set_title("Implied Volatility vs Stock Price")
                ax1.set_xlabel("Stock Price")
                ax1.set_ylabel("Implied Volatility")
                ax1.grid(True)

                # Option Price vs Stock Price
                prices = [
                    call_option_price(s, E_iv, T_iv, rf_iv, iv)
                    if option_type_iv.lower() == 'call'
                    else put_option_price(s, E_iv, T_iv, rf_iv, iv)
                    for s in spot_range
                ]

                ax2.plot(spot_range, prices)
                ax2.set_title(f"{option_type_iv} Option Price vs Stock Price")
                ax2.set_xlabel("Stock Price")
                ax2.set_ylabel("Option Price")
                ax2.grid(True)

                plt.tight_layout()
                st.pyplot(fig)


if __name__ == "__main__":
    main()
