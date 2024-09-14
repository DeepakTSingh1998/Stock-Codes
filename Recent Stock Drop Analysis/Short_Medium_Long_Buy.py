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
    linear_regression_slope,
    calculate_obv,
    calculate_50dma
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
    open_tipranks_link
)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Read stock tickers from popular indices
SNP500 = pd.DataFrame(get_snp500_tickers())
NAS100 = pd.DataFrame(get_nasdaq100_tickers())
DOW30 = pd.DataFrame(get_djia_tickers())

# Combine tickers and limit for testing purposes
Ticker_Listing = pd.concat([SNP500, NAS100, DOW30], ignore_index=True)
#Ticker_Listing = Ticker_Listing[0:30]  # Limit to 30 tickers for faster testing
Ticker_Listing = Ticker_Listing.rename(columns={0: "Ticker"}) 
Ticker_Listing = Ticker_Listing.drop_duplicates()

# Ask the user which strategy they want to run
strategy = input("Choose the strategy type (short, medium, long): ").lower()

# Set up variable inputs and weights based on the user's choice
if strategy == "short":
    # Short-term strategy
    start_year = 1
    rsi_window = 7
    bb_window = 20
    bb_std = 2
    macd_short_window = 7
    macd_long_window = 21
    macd_signal_window = 7
    atr_window = 7
    end_day = 0
    recent_days = 60
    weights = {
        'RSI': 0.30,
        'MACD': 0.20,
        'ATR': 0.10,
        'Stochastic': 0.10,
        'Volume': 0.20,
        'Slope': 0.10
    }

elif strategy == "medium":
    # Medium-term strategy
    start_year = 2
    rsi_window = 14
    bb_window = 20
    bb_std = 2
    macd_short_window = 12
    macd_long_window = 26
    macd_signal_window = 9
    atr_window = 14
    end_day = 0
    recent_days = 120
    weights = {
        'RSI': 0.25,
        'MACD': 0.20,
        'ATR': 0.15,
        'Stochastic': 0.10,
        'Volume': 0.15,
        'Slope': 0.15
    }

elif strategy == "long":
    # Long-term strategy
    start_year = 5
    rsi_window = 21
    bb_window = 50
    bb_std = 2
    macd_short_window = 26
    macd_long_window = 52
    macd_signal_window = 18
    atr_window = 21
    end_day = 0
    recent_days = 180
    weights = {
        'RSI': 0.10,
        'MACD': 0.10,
        'ATR': 0.20,
        'Stochastic': 0.05,
        'Volume': 0.10,
        'Slope': 0.45
    }

else:
    print("Invalid choice. Defaulting to medium-term strategy.")
    # Default to medium-term strategy
    start_year = 2
    rsi_window = 14
    bb_window = 20
    bb_std = 2
    macd_short_window = 12
    macd_long_window = 26
    macd_signal_window = 9
    atr_window = 14
    end_day = 0
    recent_days = 120
    weights = {
        'RSI': 0.25,
        'MACD': 0.20,
        'ATR': 0.15,
        'Stochastic': 0.10,
        'Volume': 0.15,
        'Slope': 0.15
    }

# DataFrame to hold signals
signals_df = pd.DataFrame(columns=["Ticker", "Latest_Date", "RSI", "MACD", "Signal_Line", "Volume", "ATR", "%K", "%D", "Slope", "OBV"])

for ticker in Ticker_Listing["Ticker"]:
    # Retrieve Stock Data
    data = yf.download(ticker, start=date_format(start_year * 365), end=date_format(end_day))
    
    if len(data) == 0:
        continue
    
    # Calculate technical indicators
    data['RSI'] = calculate_rsi(data, window=rsi_window)
    bollinger_data = calculate_bollinger_bands(data, window=bb_window, num_std_dev=bb_std)
    macd_data = calculate_macd(data, short_window=macd_short_window, long_window=macd_long_window, signal_window=macd_signal_window)
    atr_data = calculate_atr(data, window=atr_window)
    stochastic_data = calculate_stochastic_oscillator(data, k_window=14, d_window=3)
    
    # Calculate Linear Regression Slope
    X = np.arange(len(data))  # Time index as independent variable
    Y = data['Close'].values  # Closing prices as dependent variable
    slope, intercept = linear_regression_slope(X, Y)
    
    # Calculate OBV and 50DMA
    obv = calculate_obv(data).iloc[-1]  # Get the latest OBV value
    data['50DMA'] = calculate_50dma(data)
    
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
                "Slope": [slope],  # Add the linear regression slope to the dataframe
                "OBV": [obv]
            })

            # Concatenate the new row to the signals_df DataFrame
            signals_df = pd.concat([signals_df, new_row], ignore_index=True)

# After collecting all the data, apply the ranking function
signals_df = ranking_short_term(signals_df, weights)
# Keep only the top 25 stocks
top_25_signals_df = signals_df.head(25)


ans = True
while ans:
    
    # Set pandas options to display floats with 2 decimal places
    pd.options.display.float_format = "{:.2f}".format
    
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
            # Open the TipRanks link for the selected ticker
            open_tipranks_link(ticker)
            
            # Retrieve data again for plotting
            plot_data = yf.download(ticker, date_format(start_year * 365), end=date_format(end_day))
            
            # Recalculate Bollinger Bands for the new ticker data
            bollinger_data = calculate_bollinger_bands(plot_data, window=bb_window, num_std_dev=bb_std)

            # Recalculate MACD
            macd_data = calculate_macd(plot_data, short_window=macd_short_window, long_window=macd_long_window, signal_window=macd_signal_window)
            
            # Recalculate OBV
            obv_data = calculate_obv(plot_data)
            
            # Recalculate 50DMA
            plot_data['50DMA'] = calculate_50dma(plot_data)
            
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
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi = 250)
            plot_cumulative_frequency(buy_data, 'Buy', ax1, ticker)
            plot_cumulative_frequency(sell_data, 'Sell', ax2, ticker)
            plt.tight_layout()
            plt.show()

            ### Plotting Code Starts Here ###

            # Define the last N days' data for zoomed-in plots (N is determined by the recent_days variable)
            last_recent_days_data = plot_data[-recent_days:]
            last_recent_days_macd = macd_data[-recent_days:]
            last_recent_days_obv = obv_data[-recent_days:]  # Ensure OBV for the last 'recent_days' window

            # Create a GridSpec layout with different height ratios for each subplot
            fig = plt.figure(figsize=(16, 10), dpi= 500)
            gs = gridspec.GridSpec(3, 2, width_ratios=[2, 1], height_ratios=[3, 2, 1])  # OBV 1/6th, others 3/6 and 2/6

            # Top left subplot (full price and Bollinger Bands)
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(plot_data['Close'], color='black', label='Close Price')
            ax1.plot(bollinger_data['Upper_Band'], 'b--', label='Upper Bollinger Band', alpha=0.5)
            ax1.plot(bollinger_data['Lower_Band'], 'b--', label='Lower Bollinger Band', alpha=0.5)
            ax1.plot(bollinger_data['Middle_Band'], 'r--', label='Middle Bollinger Band')
            ax1.plot(plot_data['50DMA'], color='orange', label='50DMA')  # Add 50DMA

            # Highlight the specific date (signal date)
            specific_date = top_25_signals_df.loc[top_25_signals_df['Ticker'] == ticker, "Latest_Date"]
            ax1.axvline(x=specific_date.iloc[0], color='green', linestyle='--', label=f'Signal Date: {specific_date.iloc[0].date()}')

            # Set x-axis limits to avoid empty space
            ax1.set_xlim([plot_data.index.min(), plot_data.index.max()])
            ax1.set_title(f'{ticker} Stock Analysis')
            ax1.set_ylabel('Price')
            ax1.legend(loc='upper left')
            ax1.grid(True)

            # Bottom left subplot (MACD and Signal Line with Crossovers)
            ax2 = fig.add_subplot(gs[1, 0])
            ax2.plot(macd_data['MACD'], label='MACD', color='green', linewidth=1)
            ax2.plot(macd_data['Signal_Line'], label='Signal Line', color='red', linewidth=1)

            # Highlight crossovers
            ax2.scatter(buy_signals, macd_line.loc[buy_signals], color='blue', label='Bullish Crossover', marker='^', zorder=5)
            ax2.scatter(sell_signals, macd_line.loc[sell_signals], color='purple', label='Bearish Crossover', marker='v', zorder=5)

            # Shade MACD regions
            ax2.fill_between(macd_data.index, macd_line, signal_line, where=(macd_line > signal_line), color='green', alpha=0.3)
            ax2.fill_between(macd_data.index, macd_line, signal_line, where=(macd_line < signal_line), color='red', alpha=0.3)

            # Set x-axis limits
            ax2.set_xlim([macd_data.index.min(), macd_data.index.max()])
            ax2.set_xlabel('Date')
            ax2.set_ylabel('MACD')
            ax2.legend(loc='upper left')
            ax2.grid(True)

            # Top right subplot (Zoomed-in price and Bollinger Bands for the last N days)
            ax3 = fig.add_subplot(gs[0, 1])
            ax3.plot(last_recent_days_data['Close'], color='black', label='Close Price')
            ax3.plot(bollinger_data['Upper_Band'][-recent_days:], 'b--', label='Upper Bollinger Band', alpha=0.5)
            ax3.plot(bollinger_data['Lower_Band'][-recent_days:], 'b--', label='Lower Bollinger Band', alpha=0.5)
            ax3.plot(bollinger_data['Middle_Band'][-recent_days:], 'r--', label='Middle Bollinger Band')

            ax3.set_title(f'Last {recent_days} Days Price with Bollinger Bands')
            ax3.set_ylabel('Price')
            ax3.grid(True)
            ax3.set_xlim([last_recent_days_data.index.min(), last_recent_days_data.index.max()])
            plt.xticks(rotation=90)

            # Bottom right subplot (Zoomed-in MACD for the last N days)
            ax4 = fig.add_subplot(gs[1, 1])

            filtered_buy_signals = buy_signals[buy_signals.isin(last_recent_days_macd.index)]
            filtered_sell_signals = sell_signals[sell_signals.isin(last_recent_days_macd.index)]

            ax4.scatter(filtered_buy_signals, last_recent_days_macd['MACD'].loc[filtered_buy_signals], color='blue', label='Bullish Crossover', marker='^', zorder=5)
            ax4.scatter(filtered_sell_signals, last_recent_days_macd['MACD'].loc[filtered_sell_signals], color='purple', label='Bearish Crossover', marker='v', zorder=5)

            ax4.plot(last_recent_days_macd['MACD'], label='MACD', color='green', linewidth=1)
            ax4.plot(last_recent_days_macd['Signal_Line'], label='Signal Line', color='red', linewidth=1)

            ax4.fill_between(last_recent_days_macd.index, last_recent_days_macd['MACD'], last_recent_days_macd['Signal_Line'], where=(last_recent_days_macd['MACD'] > last_recent_days_macd['Signal_Line']), color='green', alpha=0.3)
            ax4.fill_between(last_recent_days_macd.index, last_recent_days_macd['MACD'], last_recent_days_macd['Signal_Line'], where=(last_recent_days_macd['MACD'] < last_recent_days_macd['Signal_Line']), color='red', alpha=0.3)

            ax4.set_title(f'Last {recent_days} Days MACD')
            ax4.set_ylabel('MACD')
            ax4.grid(True)
            ax4.set_xlim([last_recent_days_macd.index.min(), last_recent_days_macd.index.max()])
            plt.xticks(rotation=90)

            # Bottom left subplot (Full OBV with same time frame as other plots)
            ax5 = fig.add_subplot(gs[2, 0])  # OBV for the full time range
            ax5.plot(plot_data.index, obv_data, color='purple', label='OBV')

            ax5.set_title(f'On-Balance Volume (OBV) for {ticker}')
            ax5.set_ylabel('OBV')
            ax5.grid(True)
            ax5.set_xlim([plot_data.index.min(), plot_data.index.max()])

            # Bottom right subplot (OBV for the last N days)
            ax6 = fig.add_subplot(gs[2, 1])  # OBV for the last recent_days
            ax6.plot(last_recent_days_obv.index, last_recent_days_obv, color='purple', label='OBV')

            ax6.set_title(f'Last {recent_days} Days OBV')
            ax6.set_ylabel('OBV')
            ax6.grid(True)
            ax6.set_xlim([last_recent_days_obv.index.min(), last_recent_days_obv.index.max()])
            plt.xticks(rotation=90)

            # Adjust layout
            plt.tight_layout()
            plt.show()




            
