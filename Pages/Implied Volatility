import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats

# Function to calculate d1 and d2
def d1_d2(S, E, T, rf, sigma):
    d1 = (np.log(S / E) + (rf + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

# Function to calculate call option price
def call_option_price(S, E, T, rf, sigma):
    d1, d2 = d1_d2(S, E, T, rf, sigma)
    return S * stats.norm.cdf(d1) - E * np.exp(-rf * T) * stats.norm.cdf(d2)

# Function to calculate put option price
def put_option_price(S, E, T, rf, sigma):
    d1, d2 = d1_d2(S, E, T, rf, sigma)
    return E * np.exp(-rf * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)

# Function to calculate implied volatility
def implied_volatility(option_price, S, E, T, rf, option_type='call'):
    objective_function = lambda sigma: (
        call_option_price(S, E, T, rf, sigma) - option_price
        if option_type == 'call' else
        put_option_price(S, E, T, rf, sigma) - option_price
    )**2
    result = minimize(objective_function, 0.2, bounds=[(1e-5, 3)])
    return result.x[0]

st.title("Implied Volatility Calculation")

# Inputs for implied volatility
S_iv = st.number_input("Current Asset Price (for IV)", value=100.0)
E_iv = st.number_input("Strike Price (for IV)", value=100.0)
T_iv = st.number_input("Time to Maturity (Years, for IV)", value=1.0)
rf_iv = st.number_input("Risk-Free Interest Rate (for IV)", value=0.05)
market_price = st.number_input("Market Option Price", value=10.0)
option_type_iv = st.selectbox("Option Type", ["Call", "Put"])

iv = implied_volatility(market_price, S_iv, E_iv, T_iv, rf_iv, option_type_iv.lower())
st.info(f"Implied Volatility: {iv:.2f}")

# Implied Volatility vs. Stock Price
st.markdown("### Implied Volatility vs. Stock Price")
spot_range_iv = np.linspace(S_iv * 0.5, S_iv * 1.5, 50)
implied_vols = [implied_volatility(market_price, s, E_iv, T_iv, rf_iv, option_type_iv.lower()) for s in spot_range_iv]
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(spot_range_iv, implied_vols, label="Implied Volatility")
ax.set_title("Implied Volatility vs. Stock Price")
ax.set_xlabel("Stock Price")
ax.set_ylabel("Implied Volatility")
ax.legend()
st.pyplot(fig)
