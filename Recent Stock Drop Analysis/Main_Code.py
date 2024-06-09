import pandas as pd
import yfinance as yf
from functions import calculate_rsi, calculate_bollinger_bands, calculate_atr, date_format, ranking
from matplotlib.dates import date2num
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as pdr

# Specify the file path
file_path_1 = "SNP500.txt"
file_path_2 = "FTSE250.txt"

# Read the space-separated ticker symbols from the file into a DataFrame
SNP500 = pd.read_csv(file_path_1, header=None, names=['Ticker'], delim_whitespace=True)
FTSE250 = pd.read_csv(file_path_2, header=None, names=['Ticker'], delim_whitespace=True)

# Stock Tickers Append
Ticker_Listing = pd.concat([SNP500, FTSE250], ignore_index=True)

# Latest Date Values to append
latest_date_df = pd.DataFrame(columns=["Ticker","Latest_Date","RSI","ATR"])

# Variables 
start_year = 1.5
lin_reg_year = 3
rsi_window = 14
bb_window = 75
bb_std = 2.25
atr_window = 14
end_day = 0
lin_end_day = 0

for ticker in Ticker_Listing["Ticker"]:

    # Retrieve Stock Data
    data = yf.download(ticker, start = date_format(start_year*365), end =date_format(end_day))
    
    # Calculate RSI using the imported function
    data['RSI'] = calculate_rsi(data,window=rsi_window)

    # Calculate Bollinger Bands using the imported function
    bollinger_data = calculate_bollinger_bands(data,window = bb_window,num_std_dev = bb_std)

    # Calculate ATR
    data = calculate_atr(data, window = atr_window)
    
    # Find latest index value if close price is lower than lower band
    below_data = data[data["Close"] < data["Lower_Band"]]["Close"]
        
    if below_data.empty:
        continue 
    
    else:
        # Find latest stock date
        latest_date = below_data.index[-1]
        
        # Find RSI Data
        latest_RSI = data.loc[latest_date]["RSI"]
        
        # Find ATR Data
        latest_ATR = data.loc[latest_date]["ATR"]
        
        # Append Dates
        data_to_append = pd.DataFrame({
            "Ticker": [ticker],
            "Latest_Date": [latest_date],
            "RSI": [latest_RSI],
            "ATR": [latest_ATR]})
    
        latest_date_df = pd.concat([latest_date_df, data_to_append], ignore_index=True)
 
# Sort by latest dates
sort_dates = latest_date_df.sort_values(by="Latest_Date",ascending=False)
    
# Find top 25 Stocks
top_25 = sort_dates.head(60)

# initialise new columns
top_25["Intercept"] = 0
top_25["Slope"] = 0


for ticker in top_25["Ticker"]:

    
    # Retrieve Stock Data
    data = yf.download(ticker, start = date_format(lin_reg_year * 365), end = date_format(lin_end_day))
    
    # Remove NA data
    data = data.dropna()
    
    # X and Y
    X = np.array(range(len(data))).reshape(-1, 1)
    Y = data["Close"].values
    
    # Linear Regression
    reg = LinearRegression().fit(X,Y) 
    intercept = reg.intercept_
    slope = reg.coef_
    
    # append new regression values
    top_25.loc[top_25['Ticker'] == ticker, 'Intercept'] = intercept
    top_25.loc[top_25['Ticker'] == ticker, 'Slope'] = slope[0]
    
# Reset Index
top_25 = top_25.reset_index(drop=True)

# Rank top_25
top_25_ranked = ranking(top_25)

# Print ranked
print(top_25_ranked)


# Produce Reports 
ans = True
while ans == True:
    ticker = input("What stock ticker would you like to produce a report for (end) to stop : ")
    if ticker == "end":
        ans = False
    else:
        if ticker not in top_25_ranked["Ticker"].tolist():
            print("Error, type again")
        else:
            # Yfinance data
            plot_data = yf.download(ticker, date_format(start_year * 365), end =date_format(end_day))
            
            # Calculate Bollinger Bands using the imported function
            bollinger_data = calculate_bollinger_bands(plot_data,window = bb_window,num_std_dev = bb_std)
            
            
            # Plotting
            plt.figure(figsize=(12, 6))

            # Plot the close prices
            plt.plot(plot_data['Close'], color='black', label='Close')

            # Plot the upper and lower Bollinger Bands
            plt.plot(bollinger_data['Upper_Band'], 'b--', label='Upper Band', alpha=0.5)
            plt.plot(bollinger_data['Lower_Band'], 'b--', label='Lower Band', alpha=0.5)

            # Plot the middle Bollinger Band
            plt.plot(bollinger_data['Middle_Band'], 'r--', label='Middle Band')

            specific_date = top_25_ranked.loc[top_25_ranked['Ticker'] == ticker, "Latest_Date"]
            numeric_date = date2num(pd.to_datetime(specific_date))

            plt.axvline(x=specific_date, color='green', linestyle='--', label=specific_date)
            
            # Intercept and Slope
            intercept = top_25_ranked.loc[top_25_ranked['Ticker'] == ticker, "Intercept"]
            
            slope = top_25_ranked.loc[top_25_ranked['Ticker'] == ticker, "Slope"]
            

            # Create x values (date) for the regression line, considering the 2-year difference
            x_values_regression = np.linspace((lin_reg_year-start_year)*251, lin_reg_year*251, len(plot_data))

            # Calculate corresponding y values using the linear regression equation
            y_values_regression = slope.reset_index(drop=True)[0] * x_values_regression + intercept.reset_index(drop=True)[0]

            plt.plot(date2num(plot_data.index), y_values_regression, color='lightgreen', label='Linear Regression')

            # Customize the plot
            plt.title(f'{ticker} Stock Analysis')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            plt.xlim(plot_data.index[0],plot_data.index[-1])
            plt.show()
            
        
