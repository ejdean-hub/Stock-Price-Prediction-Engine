import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

from keras.models import Sequential
from keras.layers import LSTM, Dense


# Load stock data using yfinance
def load_stock_from_api(symbol="NFLX"):
    """
    Downloads stock data from Yahoo Finance.

    :param symbol: stock ticker
    :return: dataframe
    """
    print(f"Downloading {symbol} data from Yahoo Finance...")

    df = yf.download(symbol, start="2024-01-01", progress=False)

    if df.empty:
        raise RuntimeError(f"No data returned for symbol {symbol}")

    df = df[["Open", "High", "Low", "Close"]]

    print(f"Loaded {len(df)} rows of {symbol} stock data.")
    return df



# LSTM windowing function
def lstm_split_sequences(data, n_steps):
    """
    Splits a multivariate time‑series dataset into sliding windows for LSTM models.


    :param data:  array-like (e.g., pandas DataFrame or NumPy array)
        A 2D structure of shape (num_timesteps, num_features).
        The last column is assumed to be the target variable, and all
        preceding columns are treated as input features.

    :param n_steps: number of time steps

    :return X: np.ndarray
        Array of shape (num_sequences, n_steps, num_features - 1) containing
        input windows for the LSTM.

    :return y: np.ndarray
        Array of shape (num_sequences,) containing the target values aligned
        with each window.
    """
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps, :-1])
        y.append(data[i + n_steps - 1, -1])
    return np.array(X), np.array(y)


def main():
    # ---------------------------------------------------------
    # MAIN INGESTION
    # ---------------------------------------------------------
    df = load_stock_from_api("NFLX")

    # Scale features
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df.values),
        columns=df.columns,
        index=df.index
    )

    # Windowing
    n_steps = 10
    X_all, y_all = lstm_split_sequences(df_scaled.values, n_steps=n_steps)

    # ---------------------------------------------------------
    # TRAIN/TEST SPLIT (80/20)
    # ---------------------------------------------------------
    train_split = 0.8
    split_idx = int(np.ceil(len(X_all) * train_split))

    X_train, X_test = X_all[:split_idx], X_all[split_idx:]
    y_train, y_test = y_all[:split_idx], y_all[split_idx:]

    # ---------------------------------------------------------
    # RESHAPE FOR LSTM  [samples, timesteps, features]
    # ---------------------------------------------------------
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

    # ---------------------------------------------------------
    # LSTM MODEL
    # ---------------------------------------------------------
    model = Sequential()

    # LSTM layers
    model.add(LSTM(
        50,
        activation ='relu',
        return_sequences=True,
        input_shape=(X_train.shape[1], X_train.shape[2])
    ))
    model.add(LSTM(50, activation='relu', return_sequences=True))
    model.add(LSTM(50, activation='relu'))

    # Output layer
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')


    model.compile(optimizer='adam', loss='mse')

    # Train
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=4,
        shuffle=False,
        verbose=2
    )

    # Plot training loss over epochs (visualize model learning)
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Training Loss", linewidth=2)
    plt.title("Model Training Loss Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print(f"\nRMSE: {rmse}")
    print(f"MAPE: {mape}")

    # Scaled Plot
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label="True (Scaled)", linewidth=2)
    plt.plot(y_pred, label="Predicted (Scaled)", linewidth=2)
    plt.legend()
    plt.grid(True)
    plt.title("LSTM Predictions vs True Values (Scaled)")
    plt.xlabel("Test Sample Index")
    plt.ylabel("Scaled Price")
    plt.show()

    # Inverse transform
    y_test_real = scaler.inverse_transform(
        np.concatenate([np.zeros((len(y_test), 3)), y_test.reshape(-1, 1)], axis=1)
    )[:, -1]

    y_pred_real = scaler.inverse_transform(
        np.concatenate([np.zeros((len(y_pred), 3)), y_pred.reshape(-1, 1)], axis=1)
    )[:, -1]

    # Real metrics
    rmse_real = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    mape_real = mean_absolute_percentage_error(y_test_real, y_pred_real)

    print("Real RMSE:", rmse_real) # Root Mean Squared Error
    print("Real MAPE:", mape_real) # Mean Absolute Percentage Error

    # Real plot (USD dollars)
    # Netflix implemented a 10 for 1 stock
    # split in 2025, whch is why the historical data in Yahoo finance is giving
    # split-adjusted data in the $50 - $100 USD range rather than $500 - $1000 USD.
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_real, label="True (Real)", linewidth=2)
    plt.plot(y_pred_real, label="Predicted (Real)", linewidth=2)
    plt.legend()
    plt.grid(True)
    plt.title("LSTM Predictions vs True Values (Real Prices)")
    plt.xlabel("Test Sample Index")
    plt.ylabel("Price (USD)")
    plt.show()


if __name__ == "__main__":
    main()
