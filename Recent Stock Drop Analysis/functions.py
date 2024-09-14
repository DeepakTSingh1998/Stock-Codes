import pandas as pd
import numpy as np
from datetime import date, timedelta
import webbrowser  # To open the web link

def calculate_rsi(data, window, price='Close'):
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

# Example: Opening TipRanks for a selected ticker
def open_tipranks_link(ticker):
    url = f"https://www.tipranks.com/stocks/{ticker}/forecast"
    webbrowser.open(url)  # This will open the URL in the default browser

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

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    # Calculate the short-term EMA (12-day)
    data['EMA12'] = data['Close'].ewm(span=short_window, adjust=False).mean()

    # Calculate the long-term EMA (26-day)
    data['EMA26'] = data['Close'].ewm(span=long_window, adjust=False).mean()

    # Calculate the MACD line
    data['MACD'] = data['EMA12'] - data['EMA26']

    # Calculate the Signal line
    data['Signal_Line'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()

    return data[['MACD', 'Signal_Line']]

def calculate_stochastic_oscillator(data, k_window=14, d_window=3):
    # Calculate the highest high and lowest low over the last k_window periods
    data['L14'] = data['Low'].rolling(window=k_window).min()
    data['H14'] = data['High'].rolling(window=k_window).max()

    # Calculate the %K (Stochastic Oscillator)
    data['%K'] = (data['Close'] - data['L14']) * 100 / (data['H14'] - data['L14'])

    # Calculate the %D (Stochastic %D)
    data['%D'] = data['%K'].rolling(window=d_window).mean()

    # Drop intermediate columns
    data = data.drop(['L14', 'H14'], axis=1)

    return data[['%K', '%D']]

def calculate_obv(data):
    # Calculate On-Balance Volume
    data['OBV'] = (np.sign(data['Close'].diff()) * data['Volume']).cumsum()

    return data['OBV']

def calculate_fibonacci_retracement(data, price='Close'):
    # Calculate the Fibonacci retracement levels
    max_price = data[price].max()
    min_price = data[price].min()

    diff = max_price - min_price
    level_1 = max_price - 0.236 * diff
    level_2 = max_price - 0.382 * diff
    level_3 = max_price - 0.5 * diff
    level_4 = max_price - 0.618 * diff

    return {
        "23.6%": level_1,
        "38.2%": level_2,
        "50%": level_3,
        "61.8%": level_4
    }

def calculate_50dma(data):
    # Calculate 50-Day Moving Average
    data['50DMA'] = data['Close'].rolling(window=50).mean()
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
            data = data.sort_values(by=col, ascending=False)
            data = data.reset_index(drop=True)
            data["Ranking"] += data.index + 1
            
        elif col == "RSI":
            data = data.sort_values(by=col, ascending=True)
            data = data.reset_index(drop=True)
            data["Ranking"] += data.index + 1
        else:
            continue
    data = data.sort_values(by="Ranking", ascending=True)
    data = data.reset_index(drop=True)
    return data

def ranking_short_term(data):
    # Drop rows with NaNs in the required columns
    data = data.dropna(subset=["RSI", "MACD", "Signal_Line", "ATR", "%K", "%D", "Volume", "Slope"])

    # Initialize the ranking column
    data["Ranking"] = 0

    # RSI: Lower RSI is better for oversold conditions (use larger bins)
    bins_rsi = [-float('inf'), 30, 50, 70, float('inf')]  # Group RSI into oversold, neutral, overbought categories
    rsi_labels = [1, 5, 8, 10]  # Oversold gets the best score, overbought the worst
    data["RSI_Score"] = pd.cut(data["RSI"], bins=bins_rsi, labels=rsi_labels)

    # MACD: Crossing above signal line is bullish (use larger differentiation)
    data["MACD_Score"] = np.where(data["MACD"] > data["Signal_Line"], 1, 10)

    # ATR: Lower ATR is better (lower volatility is safer for short-term trades)
    bins_atr = [0, 1, 2, 3, float('inf')]  # Custom bins for volatility levels
    atr_labels = [1, 3, 6, 10]  # Lower volatility gets a better score
    data["ATR_Score"] = pd.cut(data["ATR"], bins=bins_atr, labels=atr_labels)

    # Stochastic Oscillator: %K above %D is bullish (larger penalty for bearish crossovers)
    data["Stochastic_Score"] = np.where(data["%K"] > data["%D"], 1, 10)

    # Volume: Higher volume is better (larger differentiation)
    bins_volume = [0, 1e5, 1e6, 1e7, float('inf')]  # Custom volume tiers
    volume_labels = [10, 5, 3, 1]  # Larger volume gets better score
    data["Volume_Score"] = pd.cut(data["Volume"], bins=bins_volume, labels=volume_labels)

    # Slope (Linear Regression): Higher slope indicates stronger upward trend
    bins_slope = [-float('inf'), 0, 0.01, 0.05, float('inf')]  # Custom bins for slope
    slope_labels = [10, 7, 3, 1]  # Higher slope gets a better score
    data["Slope_Score"] = pd.cut(data["Slope"], bins=bins_slope, labels=slope_labels)

    # Total score combines all factor scores with weights applied
    # Example of weight assignment: RSI (25%), MACD (20%), ATR (15%), Stochastic (10%), Volume (10%), Slope (20%)
    data["Total_Score"] = (
        data["RSI_Score"].astype(int) * 0.25 +
        data["MACD_Score"].astype(int) * 0.20 +
        data["ATR_Score"].astype(int) * 0.15 +
        data["Stochastic_Score"].astype(int) * 0.10 +
        data["Volume_Score"].astype(int) * 0.10 +
        data["Slope_Score"].astype(int) * 0.20  # Weight for slope
    )

    # Sort by total score (lower is better)
    data = data.sort_values(by="Total_Score", ascending=True).reset_index(drop=True)
    
    return data

# Function to calculate the price changes after each buy/sell signal (this should already exist in your code)
def calculate_price_changes(signal_dates, plot_data, signal_type):
    results = []
    for date in signal_dates:
        result = {'Type': signal_type, 'Date': date}
        for period in [1, 3, 7, 14]:
            future_date = date + pd.DateOffset(days=period)
            if future_date in plot_data.index:
                price_diff = plot_data.loc[future_date]['Close'] - plot_data.loc[date]['Close']
                result[f'{period} day price difference'] = price_diff
            else:
                result[f'{period} day price difference'] = None
        results.append(result)
    return results

# Function to calculate descriptive statistics
def calculate_statistics(data, label):
    print(f"\n--- {label} Signal Statistics ---")
    for period in ['1 day price difference', '3 day price difference', '7 day price difference', '14 day price difference']:
        print(f"\n{period}:")
        print(f"  Average: {data[period].mean():.2f}")
        print(f"  Median: {data[period].median():.2f}")
        print(f"  Minimum: {data[period].min():.2f}")
        print(f"  Maximum: {data[period].max():.2f}")

# Function to plot cumulative frequency (CDF)
def plot_cumulative_frequency(data, label, ax):
    for period in ['1 day price difference', '3 day price difference', '7 day price difference', '14 day price difference']:
        sorted_data = data[period].dropna().sort_values()
        cumulative = sorted_data.rank(method='first') / len(sorted_data)  # Cumulative distribution as ranks
        ax.plot(sorted_data, cumulative, label=period)
    
    ax.set_title(f'{label} Signals CDF')
    ax.set_xlabel('Price Change')
    ax.set_ylabel('Cumulative Frequency')
    ax.legend()
    ax.grid(True)
    
    # Fix the y-axis limits to range from 0 to 1
    ax.set_ylim([0, 1])

def linear_regression_slope(X, Y):
    """
    Perform simple linear regression to find the slope and intercept using OLS.
    
    Parameters:
    X (array-like): Independent variable (time indices).
    Y (array-like): Dependent variable (closing prices).
    
    Returns:
    slope (float): The slope of the linear regression line.
    intercept (float): The intercept of the linear regression line.
    """
    # Ensure X and Y are numpy arrays
    X = np.array(X)
    Y = np.array(Y)
    
    # Calculate the mean of X and Y
    X_mean = np.mean(X)
    Y_mean = np.mean(Y)
    
    # Calculate the numerator and denominator for the slope (beta1)
    numerator = np.sum((X - X_mean) * (Y - Y_mean))
    denominator = np.sum((X - X_mean) ** 2)
    
    # Calculate the slope (beta1) and intercept (beta0)
    slope = numerator / denominator
    intercept = Y_mean - slope * X_mean
    
    return slope, intercept

# Usage Example:
# X would be the time indices (for example, days) and Y the corresponding closing prices.

def get_snp500_tickers():
    # Wikipedia URL for the list of S&P 500 companies
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

    # Use pandas to read the tables from the Wikipedia page
    tables = pd.read_html(url)

    # The first table on the page is the list of S&P 500 companies
    snp500_table = tables[0]

    # Extract the tickers from the 'Symbol' column
    tickers = snp500_table['Symbol'].tolist()

    return tickers

def get_nasdaq100_tickers():
    url = 'https://en.wikipedia.org/wiki/NASDAQ-100'
    tables = pd.read_html(url)
    
    # The first table contains the NASDAQ-100 tickers
    nasdaq100_table = tables[4]  # Sometimes the index can vary, check the Wikipedia structure
    tickers = nasdaq100_table['Ticker'].tolist()
    
    return tickers


def get_djia_tickers():
    url = 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average'
    tables = pd.read_html(url)
    
    # The DJIA tickers are usually in the second table
    djia_table = tables[1]
    tickers = djia_table['Symbol'].tolist()
    
    return tickers


    
