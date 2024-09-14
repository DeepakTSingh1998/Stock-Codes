import pandas as pd
import yfinance as yf

# Stock calculation functions
from functions_stock_analysis import (
    calculate_rsi,
    calculate_bollinger_bands,
    calculate_macd,
    calculate_atr,
    calculate_stochastic_oscillator,
    calculate_price_changes,
    calculate_statistics,
    linear_regression_slope
)

# Stock index retrieval functions
from functions_stock_analysis import (
    get_snp500_tickers,
    get_nasdaq100_tickers,
    get_djia_tickers,
)

# Ranking and plotting functions
from functions_stock_analysis import (
    date_format,
    ranking_short_term,
    plot_cumulative_frequency,
)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# Read the space-separated ticker symbols from the file into a DataFrame
SNP500 = pd.DataFrame(get_snp500_tickers())
NAS100 = pd.DataFrame(get_nasdaq100_tickers())
DOW30 = pd.DataFrame(get_djia_tickers())

# Stock Tickers Append
Ticker_Listing = pd.concat([SNP500, NAS100, DOW30], ignore_index=True)
#Ticker_Listing = Ticker_Listing[0:30] # Testing (So testing doesn#t take too long)
Ticker_Listing = Ticker_Listing.rename(columns = {0: "Ticker"}) 
Ticker_Listing = Ticker_Listing.drop_duplicates()

# Variables 
start_year = 6
rsi_window = 14
bb_window = 20
bb_std = 2
macd_short_window = 12
macd_long_window = 26
macd_signal_window = 9
end_day = 0

# DataFrame to hold signals
signals_df = pd.DataFrame(columns=["Ticker", "Latest_Date", "RSI", "MACD", "Signal_Line", "Volume", "ATR", "%K", "%D"])

for ticker in Ticker_Listing["Ticker"]:
    # Retrieve Stock Data
    data = yf.download(ticker, start=date_format(start_year * 365), end=date_format(end_day))
    
    if len(data) == 0:
        continue
    
    # Calculate RSI, Bollinger Bands, MACD, ATR, Stochastic Oscillator
    data['RSI'] = calculate_rsi(data, window=rsi_window)
    bollinger_data = calculate_bollinger_bands(data, window=bb_window, num_std_dev=bb_std)
    macd_data = calculate_macd(data, short_window=macd_short_window, long_window=macd_long_window, signal_window=macd_signal_window)
    atr_data = calculate_atr(data, window=14)
    stochastic_data = calculate_stochastic_oscillator(data, k_window=14, d_window=3)
    
    # Calculate Linear Regression Slope using our hardcoded function
    X = np.arange(len(data))  # Time index as independent variable
    Y = data['Close'].values  # Closing prices as dependent variable
    slope, intercept = linear_regression_slope(X, Y)
    
    # Check for buy signals and add to signals_df
    if not data[data["RSI"] < 30].empty and not data[data["Close"] < bollinger_data["Lower_Band"]].empty:
        if not macd_data.empty and macd_data['MACD'].iloc[-1] > macd_data['Signal_Line'].iloc[-1]:
            latest_date = data.index[-1]
            latest_rsi = data.loc[latest_date]["RSI"]
            latest_macd = macd_data.loc[latest_date]["MACD"]
            signal_line = macd_data.loc[latest_date]["Signal_Line"]
            volume = data.loc[latest_date]["Volume"]
            atr = atr_data.loc[latest_date]["ATR"]
            stochastic_k = stochastic_data.loc[latest_date]["%K"]
            stochastic_d = stochastic_data.loc[latest_date]["%D"]

            # Create a DataFrame for the new row
            new_row = pd.DataFrame({
                "Ticker": [ticker],
                "Latest_Date": [latest_date],
                "RSI": [latest_rsi],
                "MACD": [latest_macd],
                "Signal_Line": [signal_line],
                "Volume": [volume],
                "ATR": [atr],
                "%K": [stochastic_k],
                "%D": [stochastic_d],
                "Slope": [slope]  # Add the linear regression slope to the dataframe
            })

            # Concatenate the new row to the signals_df DataFrame
            signals_df = pd.concat([signals_df, new_row], ignore_index=True)


# After collecting all the data, apply the ranking function
signals_df = ranking_short_term(signals_df)
# Keep only the top 25 stocks
top_25_signals_df = signals_df.head(25)


ans = True
while ans:
    # Set pandas display options to show all rows and columns
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', 1000)  # Set the display width to avoid column splitting

    # Print the top 25 signals
    print(top_25_signals_df)

    # Reset display options if necessary
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    ticker = input("What stock ticker would you like to produce a report for (end to stop): ")
    if ticker == "end":
        ans = False
    else:
        if ticker not in top_25_signals_df["Ticker"].tolist():
            print("Error, type again")
        else:
            # Retrieve data again for plotting
            plot_data = yf.download(ticker, date_format(start_year * 365), end=date_format(end_day))
            
            # Recalculate Bollinger Bands for the new ticker data
            bollinger_data = calculate_bollinger_bands(plot_data, window=bb_window, num_std_dev=bb_std)

            # Recalculate MACD
            macd_data = calculate_macd(plot_data, short_window=macd_short_window, long_window=macd_long_window, signal_window=macd_signal_window)

            # Identify buy (bullish) and sell (bearish) signals
            macd_line = macd_data['MACD']
            signal_line = macd_data['Signal_Line']
            buy_signals = macd_data[(macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))].index
            sell_signals = macd_data[(macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))].index

            # Calculate price changes for buy and sell signals
            buy_results = calculate_price_changes(buy_signals, plot_data, signal_type='Buy')
            sell_results = calculate_price_changes(sell_signals, plot_data, signal_type='Sell')

            # Combine the results into a single DataFrame
            signals_report = pd.DataFrame(buy_results + sell_results)
            signals_report = signals_report.sort_values(by='Date').reset_index(drop=True)

            # Calculate and print statistics for Buy and Sell signals
            buy_data = signals_report[signals_report['Type'] == 'Buy']
            sell_data = signals_report[signals_report['Type'] == 'Sell']
            calculate_statistics(buy_data, 'Buy')
            calculate_statistics(sell_data, 'Sell')

            # Create subplots for Buy and Sell signals' CDF
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            plot_cumulative_frequency(buy_data, 'Buy', ax1)
            plot_cumulative_frequency(sell_data, 'Sell', ax2)
            plt.tight_layout()
            plt.show()

            ### Plotting Code Starts Here ###

            # Define the last 60 days' data for zoomed-in plots
            last_60_days_data = plot_data[-60:]
            last_60_days_macd = macd_data[-60:]
            
            # Create a GridSpec layout: 2 rows, 2 columns, with 2/3 for the left and 1/3 for the right
            fig = plt.figure(figsize=(16, 10))
            gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[2, 2])

            # Top left subplot (full price and Bollinger Bands)
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(plot_data['Close'], color='black', label='Close Price')
            ax1.plot(bollinger_data['Upper_Band'], 'b--', label='Upper Bollinger Band', alpha=0.5)
            ax1.plot(bollinger_data['Lower_Band'], 'b--', label='Lower Bollinger Band', alpha=0.5)
            ax1.plot(bollinger_data['Middle_Band'], 'r--', label='Middle Bollinger Band')
            
            # Highlight the specific date (signal date)
            specific_date = top_25_signals_df.loc[top_25_signals_df['Ticker'] == ticker, "Latest_Date"]
            ax1.axvline(x=specific_date.iloc[0], color='green', linestyle='--', label=f'Signal Date: {specific_date.iloc[0].date()}')

            # Set x-axis limits to avoid empty space
            ax1.set_xlim([plot_data.index.min(), plot_data.index.max()])

            # Set labels and title for top subplot
            ax1.set_title(f'{ticker} Stock Analysis')
            ax1.set_ylabel('Price')
            ax1.legend(loc='upper left')
            ax1.grid(True)

            # Bottom left subplot (MACD and Signal Line with Crossovers)
            ax2 = fig.add_subplot(gs[1, 0])
            ax2.plot(macd_data['MACD'], label='MACD', color='green', linewidth=1)  # Decreased thickness
            ax2.plot(macd_data['Signal_Line'], label='Signal Line', color='red', linewidth=1)  # Decreased thickness

            # Highlight crossovers with buy signals (blue) and sell signals (purple)
            ax2.scatter(buy_signals, macd_line.loc[buy_signals], color='blue', label='Bullish Crossover', marker='^', zorder=5)
            ax2.scatter(sell_signals, macd_line.loc[sell_signals], color='purple', label='Bearish Crossover', marker='v', zorder=5)

            # Shade the area between MACD and Signal Line
            ax2.fill_between(macd_data.index, macd_line, signal_line, where=(macd_line > signal_line), color='green', alpha=0.3, interpolate=True)
            ax2.fill_between(macd_data.index, macd_line, signal_line, where=(macd_line < signal_line), color='red', alpha=0.3, interpolate=True)

            # Set x-axis limits to avoid empty space
            ax2.set_xlim([macd_data.index.min(), macd_data.index.max()])

            # Set labels and title for bottom subplot
            ax2.set_xlabel('Date')
            ax2.set_ylabel('MACD')
            ax2.legend(loc='upper left')
            ax2.grid(True)

            # Top right subplot (Zoomed-in price and Bollinger Bands for the last 60 days)
            ax3 = fig.add_subplot(gs[0, 1])
            ax3.plot(last_60_days_data['Close'], color='black', label='Close Price')
            ax3.plot(bollinger_data['Upper_Band'][-60:], 'b--', label='Upper Bollinger Band', alpha=0.5)
            ax3.plot(bollinger_data['Lower_Band'][-60:], 'b--', label='Lower Bollinger Band', alpha=0.5)
            ax3.plot(bollinger_data['Middle_Band'][-60:], 'r--', label='Middle Bollinger Band')

            ax3.set_title('Last 60 Days Price with Bollinger Bands')
            ax3.set_ylabel('Price')
            ax3.grid(True)

            # Set x-axis limits for the last 60 days
            ax3.set_xlim([last_60_days_data.index.min(), last_60_days_data.index.max()])

            # Set the x-axis date labels to vertical
            plt.xticks(rotation=90)

            # Bottom right subplot (Zoomed-in MACD for the last 60 days)
            ax4 = fig.add_subplot(gs[1, 1])

            # Filter buy and sell signals to only include those within the last 60 days
            filtered_buy_signals = buy_signals[buy_signals.isin(last_60_days_macd.index)]
            filtered_sell_signals = sell_signals[sell_signals.isin(last_60_days_macd.index)]

            # Plot the crossovers for the last 60 days using filtered signals
            ax4.scatter(filtered_buy_signals, last_60_days_macd['MACD'].loc[filtered_buy_signals], color='blue', label='Bullish Crossover', marker='^', zorder=5)
            ax4.scatter(filtered_sell_signals, last_60_days_macd['MACD'].loc[filtered_sell_signals], color='purple', label='Bearish Crossover', marker='v', zorder=5)

            ax4.plot(last_60_days_macd['MACD'], label='MACD', color='green', linewidth=1)
            ax4.plot(last_60_days_macd['Signal_Line'], label='Signal Line', color='red', linewidth=1)

            ax4.fill_between(last_60_days_macd.index, last_60_days_macd['MACD'], last_60_days_macd['Signal_Line'], where=(last_60_days_macd['MACD'] > last_60_days_macd['Signal_Line']), color='green', alpha=0.3, interpolate=True)
            ax4.fill_between(last_60_days_macd.index, last_60_days_macd['MACD'], last_60_days_macd['Signal_Line'], where=(last_60_days_macd['MACD'] < last_60_days_macd['Signal_Line']), color='red', alpha=0.3, interpolate=True)

            ax4.set_title('Last 60 Days MACD')
            ax4.set_ylabel('MACD')
            ax4.grid(True)

            # Set x-axis limits for the last 60 days MACD
            ax4.set_xlim([last_60_days_macd.index.min(), last_60_days_macd.index.max()])

            # Set the x-axis date labels to vertical for the bottom-right plot as well
            plt.xticks(rotation=90)

            # Adjust layout
            plt.tight_layout()
            plt.show()
