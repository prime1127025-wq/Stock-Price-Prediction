
import yfinance as yf #helps in getting stock data from yahoo finance
import pandas as pd  #helps in data manipulation and analysis
import numpy as np  #helps in numerical computations
import plotly.graph_objects as go #helps in creating interactive visualizations
from keras.models import Sequential #helps in creating sequential neural network models
from keras.layers import Dense, LSTM #helps in creating layers for neural network models
from sklearn.preprocessing import MinMaxScaler #helps in scaling data to a specific range
from sklearn.model_selection import train_test_split #helps in splitting data into training and testing sets
import streamlit as st #helps in creating web applications


model_cache = {}

def fetch_data(symbol):
    stock = yf.Ticker(symbol) #fetches stock data for the given symbol from Yahoo Finance
    data = stock.history(period="5y") #fetches historical stock data for the given symbol
    return data[['Close']] #returns the closing prices of the stock


#Data Preprocessing
def preprocess_data(data):
    df = pd.DataFrame(data)
    df['SMA_50'] = df['Close'].rolling(window=50).mean() #calculates the 50-day simple moving average of the closing prices
    df['EMA_50'] = df['Close'].ewm(span=20, adjust=False).mean() #calculates the 50-day exponential moving average of the closing prices
    df['Daily_Return'] = df['Close'].pct_change() #calculates the daily returns of the closing prices
    df['Log Return'] = np.log(df['Close'] / df['Close'].shift(1)) #calculates the logarithmic returns of the closing prices
    delta = df['Close'].diff(1) #calculates the difference between the current and previous closing prices
    gain = delta.where(delta > 0, 0).rolling(window=14).mean() #calculates the average gain over a 14-day period
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean() #calculates the average loss over a 14-day period
    rs = gain / loss #calculates the relative strength value>70 overbought and <30 oversold
    df['RSI_14'] = 100 - (100 / (1 + rs)) #calculates the 14-day relative strength index of the closing prices
    df['Middle_Band'] = df['Close'].rolling(window=20).mean() #calculates the 20-day simple moving average of the closing prices
    df['Upper_Band'] = df['Middle_Band'] + (df['Close'].rolling(window=20).std() * 2) #calculates the upper Bollinger Band of the closing prices
    df['Lower_Band'] = df['Middle_Band'] - (df['Close'].rolling(window=20).std() * 2) #calculates the lower Bollinger Band of the closing prices
    df['Momentum'] = df['Close'] - df['Close'].shift(4) #calculates the momentum of the closing prices
    df['Volatility'] = df['Close'].rolling(window=21).std() #calculates the 21-day rolling standard deviation of the closing prices
    df.dropna(inplace=True) #removes any rows with missing values
    return df


def normalize_data(data):
    scaler = MinMaxScaler() #scales the data to a specific range (0, 1) using Min-Max scaling
    scaled_data = scaler.fit_transform(data[['Close']]) #fits the scaler to the data and transforms it to the scaled range
    return scaled_data, scaler #returns the scaled data and the scaler object


def prepare_data(scaled_data, time_step=60): #prepares the data for training and testing the LSTM model by creating sequences of a specified time step
    X, y = [], [] #initializes empty lists for the input features (X)
    for i in range(time_step, len(scaled_data)): #iterates through the scaled data starting from the specified time step
        X.append(scaled_data[i-time_step:i, 0]) #appends the previous 'time_step' number of closing prices to the input features list
        y.append(scaled_data[i, 0]) #appends the current closing price to the target variable list
    return np.array(X), np.array(y) #returns the input features and target variable as numpy arrays


def build_model(input_shape): #builds a sequential LSTM model with the specified input shape
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape), #adds an LSTM layer with 50 units and returns sequences for the next layer
        LSTM(50, return_sequences=False), #adds another LSTM layer with 50 units and does not return sequences for the next layer
        Dense(25), #adds a dense layer with 25 units
        Dense(1) #adds a dense layer with 1 unit for the output
    ]) #initializes a sequential model
    model.compile(optimizer='adam', loss='mean_squared_error') #compiles the model with the Adam optimizer and mean squared error loss function
    return model #returns the compiled model


#Streamlit App

st.title("Stock Price Prediction App") #sets the title of the Streamlit app
stock_list = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'] #defines a list of stock symbols for the user to choose from
selected_stock = st.selectbox("Select a stock symbol", stock_list) #creates a dropdown menu for the user to select a stock symbol

st.write(f"Fetching data for {selected_stock}...") #displays a message indicating that data is being fetched for the selected stock symbol
data = fetch_data(selected_stock) #fetches the historical stock data for the selected stock symbol

if data.empty:
    st.error("No stock data found. Please check your internet connection or try another stock.")
    st.stop()

latest_price = data['Close'].iloc[-1] #gets the latest closing price of the stock
st.write(f"Latest closing price for {selected_stock}: ${latest_price:.2f}") #displays the latest closing price of the stock

if st.button("Train and Predict"):
    st.write("Please wait while the model is being trained and predictions are being made...") #displays a message indicating that the model is being trained and predictions are being made

    if selected_stock in model_cache:
        model, scaler, scaled_data = model_cache[selected_stock] #retrieves the cached model, scaler and scaled data for the selected stock symbol
    else:
        data = preprocess_data(data) #preprocesses the historical stock data
        scaled_data, scaler = normalize_data(data) #normalizes the preprocessed data
        X, y = prepare_data(scaled_data) #prepares the data for training and testing the LSTM model

        X = X.reshape(X.shape[0], X.shape[1], 1) #reshapes the input features to match the expected input shape of the LSTM model

        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        ) #splits the data into training and testing sets

        model = build_model((x_train.shape[1], x_train.shape[2])) #builds the LSTM model with the specified input shape
        model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=1) #trains the model on the training data for 5 epochs with a batch size of 32

        model_cache[selected_stock] = (model, scaler, scaled_data) #caches the trained model, scaler and scaled data for the selected stock symbol

    st.write("Predicting for the next 10 days...") #displays a message indicating that predictions are being made for the next 10 days
    predictions = []
    input_sequence = scaled_data[-60:].reshape(1, 60, 1) #gets the last 60 days of scaled data as the input sequence for making predictions

    for day in range(10): #iterates for the next 10 days
        predicted_price = model.predict(input_sequence, verbose=0)[0, 0] #makes a prediction for the next day's closing price using the trained model
        predictions.append(predicted_price) #appends the predicted price to the predictions list
        input_sequence = np.append(
            input_sequence[:, 1:, :],
            np.array(predicted_price).reshape(1, 1, 1),
            axis=1
        ) #updates the input sequence by removing the oldest price and adding the predicted price

    predictions = scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    ) #inverse transforms the predicted prices back to the original scale using the scaler

    days = pd.date_range(
        start=pd.Timestamp.now().normalize() + pd.Timedelta(days=1),
        periods=10,
        freq='B'
    ).strftime('%Y-%m-%d') #creates a date range for the next 10 business days

    prediction_df = pd.DataFrame({
        'Date': days,
        'Predicted_Close': predictions.flatten()
    }) #creates a DataFrame with the predicted closing prices and corresponding dates

    st.write("Predicted Price")
    st.table(prediction_df)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=prediction_df['Date'],
        y=prediction_df['Predicted_Close'],
        mode='lines+markers',
        name='Predicted Close'
    ))
    fig.update_layout(
        title=f"10-Day Price Prediction for {selected_stock}",
        xaxis_title="Date",
        yaxis_title="Predicted Closing Price ($)",
        template="plotly_dark"
    )
    st.plotly_chart(fig)
