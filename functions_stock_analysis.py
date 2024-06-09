
import pandas as pd
from datetime import date, timedelta

def calculate_rsi(data,window, price='Close'):
    # Calculate daily price changes
    delta = data[price].diff(1)

    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)

    # Calculate average gains and losses over the specified window
    avg_gain = gains.rolling(window=window, min_periods=1).mean()
    avg_loss = losses.rolling(window=window, min_periods=1).mean()

    # Calculate relative strength (RS)
    rs = avg_gain / avg_loss

    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))

    return rsi

def calculate_bollinger_bands(data, window, num_std_dev, price="Close"):
    # Calculate the middle band (SMA)
    data['Middle_Band'] = data[price].rolling(window=window).mean()

    # Calculate the standard deviation
    data['Std_Dev'] = data[price].rolling(window=window).std()

    # Calculate the upper and lower bands
    data['Upper_Band'] = data['Middle_Band'] + (num_std_dev * data['Std_Dev'])
    data['Lower_Band'] = data['Middle_Band'] - (num_std_dev * data['Std_Dev'])

    return data[['Close', 'Middle_Band', 'Upper_Band', 'Lower_Band']]

def calculate_atr(data, window):
    # Calculate True Range
    data['High-Low'] = data['High'] - data['Low']
    data['High-PrevClose'] = abs(data['High'] - data['Close'].shift(1))
    data['Low-PrevClose'] = abs(data['Low'] - data['Close'].shift(1))
    data['True_Range'] = data[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
    
    # Calculate Average True Range
    data['ATR'] = data['True_Range'].rolling(window=window).mean()

    # Drop intermediate columns
    data = data.drop(['High-Low', 'High-PrevClose', 'Low-PrevClose', 'True_Range'], axis=1)

    return data

def date_format(time_dif):
    ans = date.today() - timedelta(days=time_dif)
    return ans.strftime("%Y-%m-%d")

def ranking(data):
    data["Ranking"] = 0
    col_data = data.columns.tolist()
    del col_data[-1]
    for col in col_data:
        if col == "Latest Date" or col == "Slope":
            data = data.sort_values(by=col,ascending=False)
            data = data.reset_index(drop=True)
            data["Ranking"] += data.index + 1
            
        elif col == "RSI":
            data = data.sort_values(by=col,ascending=True)
            data = data.reset_index(drop=True)
            data["Ranking"] += data.index + 1
        else:
            continue
    data = data.sort_values(by="Ranking",ascending=True)
    data = data.reset_index(drop=True)
    return data
    
    
    
    
    
    