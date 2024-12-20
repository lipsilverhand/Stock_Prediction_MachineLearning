import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import os
os.system('pip install --upgrade matplotlib')

# Streamlit App
st.title("Stock Price Prediction App")

# Stock Symbol Input
symbol = st.text_input("Enter a stock symbol (e.g., 'TSLA', 'NVDA')", 'TSLA')

# Download Stock Data using yfinance
start_date = '2020-01-01'
end_date = datetime.today().strftime('%Y-%m-%d')  

if symbol:
    # Download stock data
    data = yf.download(symbol, start=start_date, end=end_date)

    # Display the dataset preview
    st.write(f"### {symbol} Stock Data (2020):", data.head(8))

    st.write("### End of the Data (Now):", data.tail(8))

    # Data Preprocessing
    scaler = MinMaxScaler()
    data[['Open', 'High', 'Low', 'Close', 'Volume']] = scaler.fit_transform(
        data[['Open', 'High', 'Low', 'Close', 'Volume']])
    data['MA_10'] = data['Close'].rolling(window=10).mean()
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    data['Target'] = data['Close'].shift(-1)
    data.dropna(inplace=True)

    # Train-Test Split
    X = data[['Open', 'High', 'Low', 'Volume', 'MA_10', 'MA_50']]
    y = data['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate Model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"### Model Performance: R² Score = {r2:.2f}, MSE = {mse:.2f}")

    # Plot Predictions
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(y_test.values, label="Actual Prices", color="blue")
    ax.plot(y_pred, label="Predicted Prices", color="red")
    ax.set_title(f"Actual vs Predicted Prices for {symbol}")
    ax.legend()
    st.pyplot(fig)

    # Plot Predicted vs Actual over the entire period (training + test)
    full_data = data[['Close']].copy()

    # Use the entire data for prediction (not just the training set)
    full_X = data[['Open', 'High', 'Low', 'Volume', 'MA_10', 'MA_50']]
    full_data['Predicted'] = model.predict(full_X)  # Use the entire feature set for prediction

    # Create a new figure object for this plot
    fig, ax = plt.subplots(figsize=(10, 6))
    full_data.plot(ax=ax)  # Plot on the created axes
    ax.set_title(f"Predicted vs Actual Prices for {symbol} (Entire Period)")
    st.pyplot(fig) 


