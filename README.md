# 🎈 Stock Analysis Dashboard - Streamlit Application

A simple Streamlit app template for you to modify!

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blank-app-template.streamlit.app/)

## Overview
This application is a financial analysis and stock prediction tool built with Streamlit. It provides detailed metrics about selected stocks, including Risk-Free Rate, Sharpe Ratio, Annual Return, Annual Volatility, Historical Volatility, and Implied Volatility (if available). The app also offers options analysis, including metrics such as Put/Call Ratio and Implied Volatility.

### Features:
- **Risk-Free Rate (India 10Y):** Displays the current risk-free rate fetched from FRED API.
- **Sharpe Ratio:** The Sharpe ratio of the stock is calculated and displayed to indicate risk-adjusted return.
- **Annual Return:** Provides the annual return percentage for the selected stock.
- **Annual Volatility:** Shows the annual volatility percentage, giving insights into the risk associated with the stock.
- **Historical Volatility Plot:** A graphical representation of the historical volatility over time.
- **Options Analysis:** If available, additional analysis for options contracts, including Put/Call Ratio, nearest expiry, and Implied Volatility.

## How to Run the Application

1. **Install Dependencies**
   - Clone the repository or copy the code to your local environment.
   - Install the required Python libraries using the following command:
     ```bash
     pip install -r requirements.txt
     ```

2. **Run the Streamlit Application**
   Execute the following command to run the app:
   ```bash
   streamlit run streamlit_app.py
   ```
   Replace `streamlit_app.py` with the name of your Python file containing the Streamlit code.

3. **Access the Application**
   Once the application starts, open your web browser and go to the URL displayed in the terminal (typically `http://localhost:8501`).

## Code Description
The code below is responsible for displaying the key metrics of the stock analysis:

```python
st.write(f"Risk-Free Rate (India 10Y): {analysis['risk_free_rate']*100:.2f}%")
st.write(f"Sharpe Ratio: {analysis['sharpe_ratio']:.2f}")
st.write(f"Annual Return: {analysis['annual_return']*100:.2f}%")
st.write(f"Annual Volatility: {analysis['annual_volatility']*100:.2f}%")
st.write("Current Historical Volatility:")
st.pyplot(analysis['volatility_plot'])

# Option analysis: Add implied volatility if available
if 'options' in analysis:
    st.write("Options Analysis:")
    st.write(f"Nearest Expiry: {analysis['options']['nearest_expiry']}")
    st.write(f"Put/Call Ratio: {analysis['options']['put_call_ratio']:.2f}")
    # Assuming analysis includes 'implied_volatility' key for options
    if 'implied_volatility' in analysis['options']:
        st.write(f"Implied Volatility: {analysis['options']['implied_volatility']*100:.2f}%")
```

### Key Metrics Displayed:
- **Risk-Free Rate (India 10Y):** Displays the current risk-free rate fetched from the FRED API.
- **Sharpe Ratio:** Shows the Sharpe ratio of the stock, which measures the risk-adjusted return.
- **Annual Return:** Displays the annual return of the selected stock.
- **Annual Volatility:** Provides the annual volatility percentage, indicating the stock's risk level.
- **Historical Volatility Plot:** Displays a plot representing historical volatility of the stock over time.
- **Options Analysis (If Available):** Displays the nearest expiry date, put/call ratio, and implied volatility for options data.

## License
This project is licensed under the MIT License.

## Contact
For any inquiries, you can reach out to the project maintainer:
- **Email:** your-email@example.com
- **GitHub:** [your-github-profile](https://github.com/your-github-profile)

Feel free to contribute by submitting issues or pull requests.
