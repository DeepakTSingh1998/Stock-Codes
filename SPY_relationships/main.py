'''
Small program to identify relationship between stocks in the snp500 listing
Closer to (+)1 -> Stocks have positive relationship
Closer to (-)1 -> Stocks have negetive relationship

'''

import pandas as pd
import yfinance as yf
import numpy as np

# Variables
start_date = "2024-01-01"

# Fetch S&P 500 tickers
def get_sp500_tickers():
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    df = table[0]
    return df['Symbol'].tolist()

# Retrieve Tickers
tickers = get_sp500_tickers()
working_tickers = []
all_data = {}
column_line = None

for ticker in tickers:
    data = yf.download(ticker, start=start_date)
    if not data.empty:
        try:
            close_prices = data["Close"].to_numpy()
            if column_line is None:
                column_line = close_prices  # Initialize with the first ticker's data
            else:
                # Ensure the new data has the same length as the existing data
                if len(close_prices) == column_line.shape[0]:
                    column_line = np.c_[column_line, close_prices]
                else:
                    continue  # Skip this ticker if lengths don't match
            working_tickers.append(ticker)  # Only append if data is successfully added
            all_data[ticker] = data
        except:
            continue  # Skip this ticker if there's an error


# Matrices
correlation_matrix = np.corrcoef(column_line, rowvar = False)

# Convert to a DataFrame for better readability
correlation_df = pd.DataFrame(correlation_matrix, index=working_tickers, columns=working_tickers)

# Export the correlation matrix to an Excel file
output_file = "correlation_matrix.xlsx"
correlation_df.to_excel(output_file)
