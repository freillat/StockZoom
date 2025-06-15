import pandas as pd
import numpy as np
import requests
from io import StringIO
import re
import time
import yfinance as yf
import talib

# Question 1

url = f"https://stockanalysis.com/ipos/withdrawn/"
headers = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/58.0.3029.110 Safari/537.3'
    )
}

try:
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    # Wrap HTML text in StringIO to avoid deprecation warning
    # "Passing literal html to 'read_html' is deprecated and will be removed in a future version. To read from a literal string, wrap it in a 'StringIO' object."
    html_io = StringIO(response.text)
    tables = pd.read_html(html_io)

    if not tables:
        raise ValueError(f"No tables found for year {year}.")

except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
except ValueError as ve:
    print(f"Data error: {ve}")
except Exception as ex:
    print(f"Unexpected error: {ex}")

df = tables[0]

def get_company_class(company_name):
    if not isinstance(company_name, str):
        # Handle non-string inputs, e.g., NaN, None, numbers
        return "Other"

    # Convert to lowercase and split into words for robust matching
    name_lower = company_name.lower()
    words = name_lower.split()
    words = re.findall(r'\b\w+\b', name_lower) # \b is word boundary, \w+ is one or more word characters
    # Define the rules and their order
    # (Pattern check function, Class Name)
    rules = [
        (lambda n, w: ("acquisition" in w and ("corp" in w or "corporation" in w)), "Acq.Corp"),
        (lambda n, w: ("inc" in w or "incorporated" in w), "Inc"),
        (lambda n, w: "group" in w, "Group"),
        (lambda n, w: ("ltd" in w or "limited" in w), "Limited"),
        (lambda n, w: "holdings" in w, "Holdings"),
    ]

    for check_func, class_name in rules:
        if check_func(name_lower, words):
            return class_name

    # If no rule matches
    return "Other"

def parse_price_range(price_range):
    if not isinstance(price_range, str):
        # Handle non-string inputs (e.g., NaN, None) by returning None
        return None

    price_range = price_range.strip() # Remove leading/trailing whitespace

    if price_range == '-':
        return None
    
    # Remove '$' sign and potentially extra spaces
    cleaned_price_range = price_range.replace('$', '').strip()

    if '-' in cleaned_price_range:
        # It's a range, e.g., '8.00-10.00'
        try:
            low_str, high_str = cleaned_price_range.split('-')
            low = float(low_str.strip())
            high = float(high_str.strip())
            return (low + high) / 2.0
        except ValueError:
            # Handle cases where conversion to float fails
            return None
    else:
        # It's a single price, e.g., '5.00'
        try:
            return float(cleaned_price_range)
        except ValueError:
            # Handle cases where conversion to float fails
            return None

def clean_and_convert_shares_offered(value):
    if pd.isna(value) or value is None:
        return pd.NA # Explicitly return pandas' nullable NA for missing values

    s_value = str(value).strip() # Convert to string and remove leading/trailing whitespace

    if not s_value: # Handle empty strings
        return pd.NA

    # Remove commas (e.g., "1,000,000" -> "1000000")
    s_value = s_value.replace(',', '')

    try:
        # Attempt to convert to float. Using float to handle potential decimals,
        # but if you expect only integers, you can cast to int later if desired.
        return float(s_value)
    except ValueError:
        # If conversion fails (e.g., for "N/A", "Unknown", etc.)
        return pd.NA


df['Company Class'] = df['Company Name'].apply(get_company_class)
df['Avg. price'] = df['Price Range'].apply(parse_price_range)
df['Shares Offered'] = df['Shares Offered'].apply(clean_and_convert_shares_offered)
df['Withdrawn Value'] = df['Shares Offered'] * df['Avg. price']
total_withdrawn_by_class = df.groupby('Company Class')['Withdrawn Value'].sum()
print(total_withdrawn_by_class/1000000)

# # Question 2

url = f"https://stockanalysis.com/ipos/2024/"

headers = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/58.0.3029.110 Safari/537.3'
    )
}

try:
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    # Wrap HTML text in StringIO to avoid deprecation warning
    # "Passing literal html to 'read_html' is deprecated and will be removed in a future version. To read from a literal string, wrap it in a 'StringIO' object."
    html_io = StringIO(response.text)
    tables = pd.read_html(html_io)

    if not tables:
        raise ValueError(f"No tables found for year {year}.")

except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
except ValueError as ve:
    print(f"Data error: {ve}")
except Exception as ex:
    print(f"Unexpected error: {ex}")

df = tables[0]
df['IPO Date'] = pd.to_datetime(df['IPO Date'], errors='coerce')
cutoff_date = pd.to_datetime('2024-06-01')
condition_date = df['IPO Date'] < cutoff_date
condition_price = df['IPO Price'].astype(str) != '-'
df_filtered = df[condition_date & condition_price].copy()

print(df_filtered)

stocks_df = pd.DataFrame({'A' : []})

for i,ticker in enumerate(df_filtered['Symbol']):
  print(i,ticker)

  # Work with stock prices
  ticker_obj = yf.Ticker(ticker)

  # historyPrices = yf.download(tickers = ticker,
  #                    period = "max",
  #                    interval = "1d")
  historyPrices = ticker_obj.history(
                     period = "max",
                     interval = "1d")

  # generate features for historical prices, and what we want to predict
  historyPrices['Ticker'] = ticker
  historyPrices['Year']= historyPrices.index.year
  historyPrices['Month'] = historyPrices.index.month
  historyPrices['Weekday'] = historyPrices.index.weekday
  historyPrices['Date'] = historyPrices.index.date

  # historical returns
  for i in [1,3,7,30,90,252,365]:
    historyPrices['growth_'+str(i)+'d'] = historyPrices['Close'] / historyPrices['Close'].shift(i)
  historyPrices['growth_future_30d'] = historyPrices['Close'].shift(-30) / historyPrices['Close']

  # Technical indicators
  # SimpleMovingAverage 10 days and 20 days
  historyPrices['SMA10']= historyPrices['Close'].rolling(10).mean()
  historyPrices['SMA20']= historyPrices['Close'].rolling(20).mean()
  historyPrices['growing_moving_average'] = np.where(historyPrices['SMA10'] > historyPrices['SMA20'], 1, 0)
  historyPrices['high_minus_low_relative'] = (historyPrices.High - historyPrices.Low) / historyPrices['Close']

  # 30d rolling volatility : https://ycharts.com/glossary/terms/rolling_vol_30
  historyPrices['volatility'] =   historyPrices['Close'].rolling(30).std() * np.sqrt(252)

  # what we want to predict
  historyPrices['is_positive_growth_30d_future'] = np.where(historyPrices['growth_future_30d'] > 1, 1, 0)

  # sleep 1 sec between downloads - not to overload the API server
  time.sleep(1)


  if stocks_df.empty:
    stocks_df = historyPrices
  else:
    stocks_df = pd.concat([stocks_df, historyPrices], ignore_index=True)

stocks_df['Sharpe'] = (stocks_df['growth_252d'] - 0.045) / stocks_df['volatility']

condition_day = stocks_df['Date'].astype(str) == "2025-06-06"
stocks_df_filtered = stocks_df[condition_day].copy()
print(stocks_df_filtered['growth_252d'].describe())
print(stocks_df_filtered['Sharpe'].describe())

# Question 3

url = f"https://stockanalysis.com/ipos/2024/"

headers = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/58.0.3029.110 Safari/537.3'
    )
}

try:
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    # Wrap HTML text in StringIO to avoid deprecation warning
    # "Passing literal html to 'read_html' is deprecated and will be removed in a future version. To read from a literal string, wrap it in a 'StringIO' object."
    html_io = StringIO(response.text)
    tables = pd.read_html(html_io)

    if not tables:
        raise ValueError(f"No tables found for year {year}.")

except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
except ValueError as ve:
    print(f"Data error: {ve}")
except Exception as ex:
    print(f"Unexpected error: {ex}")

df = tables[0]
df['IPO Date'] = pd.to_datetime(df['IPO Date'], errors='coerce')
cutoff_date = pd.to_datetime('2024-06-01')
condition_date = df['IPO Date'] < cutoff_date
condition_price = df['IPO Price'].astype(str) != '-'
df_filtered = df[condition_date & condition_price].copy().reset_index(drop=True)

print(df_filtered)

stocks_df = pd.DataFrame({'A' : []})

for i,ticker in enumerate(df_filtered['Symbol']):
  print(i,ticker)

  # Work with stock prices
  ticker_obj = yf.Ticker(ticker)

  # historyPrices = yf.download(tickers = ticker,
  #                    period = "max",
  #                    interval = "1d")
  historyPrices = ticker_obj.history(
                     period = "max",
                     interval = "1d")

  # generate features for historical prices, and what we want to predict
  historyPrices['Ticker'] = ticker
  historyPrices['Year']= historyPrices.index.year
  historyPrices['Month'] = historyPrices.index.month
  historyPrices['Weekday'] = historyPrices.index.weekday
  historyPrices['Date'] = historyPrices.index.date

  # historical returns
  for i in [1,3,7,30,90,252,365]:
    historyPrices['growth_'+str(i)+'d'] = historyPrices['Close'] / historyPrices['Close'].shift(i)
  historyPrices['growth_future_30d'] = historyPrices['Close'].shift(-30) / historyPrices['Close']
  for i in range(1,13):
    historyPrices['future_growth_'+str(i)+'m'] = historyPrices['Close'].shift(-21*i) / historyPrices['Close']

  # Technical indicators
  # SimpleMovingAverage 10 days and 20 days
  historyPrices['SMA10']= historyPrices['Close'].rolling(10).mean()
  historyPrices['SMA20']= historyPrices['Close'].rolling(20).mean()
  historyPrices['growing_moving_average'] = np.where(historyPrices['SMA10'] > historyPrices['SMA20'], 1, 0)
  historyPrices['high_minus_low_relative'] = (historyPrices.High - historyPrices.Low) / historyPrices['Close']

  # 30d rolling volatility : https://ycharts.com/glossary/terms/rolling_vol_30
  historyPrices['volatility'] =   historyPrices['Close'].rolling(30).std() * np.sqrt(252)

  # what we want to predict
  historyPrices['is_positive_growth_30d_future'] = np.where(historyPrices['growth_future_30d'] > 1, 1, 0)

  # sleep 1 sec between downloads - not to overload the API server
  time.sleep(1)


  if stocks_df.empty:
    stocks_df = historyPrices
  else:
    stocks_df = pd.concat([stocks_df, historyPrices], ignore_index=True)

stocks_df['Sharpe'] = (stocks_df['growth_252d'] - 0.045) / stocks_df['volatility']

condition_day = stocks_df['Date'].astype(str) == "2025-06-06"
stocks_df_filtered = stocks_df[condition_day].copy()

min_data = pd.DataFrame()
min_data['Ticker'] = df_filtered['Symbol']

earliest_dates = {}

for ticker in min_data['Ticker']:
    ticker_data = stocks_df[
        (stocks_df['Ticker'] == ticker) &
        (stocks_df['Close'].notna())
    ]
    if not ticker_data.empty:
        earliest_date = ticker_data['Date'].min()
        earliest_dates[ticker] = earliest_date
    else:
        earliest_dates[ticker] = pd.NaT

min_data['Date'] = min_data['Ticker'].map(earliest_dates)

stocks_df_cleaned = stocks_df.reset_index(drop=True)

merged_df = pd.merge(
    min_data,
    stocks_df_cleaned,
    on=['Ticker', 'Date'],
    how='inner'
)

for i in range(1,13):
    print(merged_df['future_growth_'+str(i)+'m'].mean())

# Question 4

# https://companiesmarketcap.com/usa/largest-companies-in-the-usa-by-market-cap/
US_STOCKS = ['MSFT', 'AAPL', 'GOOG', 'NVDA', 'AMZN', 'META', 'BRK-B', 'LLY', 'AVGO','V', 'JPM']

# You're required to add EU_STOCKS and INDIA_STOCS
# https://companiesmarketcap.com/european-union/largest-companies-in-the-eu-by-market-cap/
EU_STOCKS = ['NVO','MC.PA', 'ASML', 'RMS.PA', 'OR.PA', 'SAP', 'ACN', 'TTE', 'SIE.DE','IDEXY','CDI.PA']

# https://companiesmarketcap.com/india/largest-companies-in-india-by-market-cap/
INDIA_STOCKS = ['RELIANCE.NS','TCS.NS','HDB','BHARTIARTL.NS','IBN','SBIN.NS','LICI.NS','INFY','ITC.NS','HINDUNILVR.NS','LT.NS']

ALL_TICKERS = US_STOCKS + EU_STOCKS + INDIA_STOCKS

stocks_df = pd.DataFrame({'A' : []})

for i,ticker in enumerate(ALL_TICKERS):
  print(i,ticker)

  # Work with stock prices
  ticker_obj = yf.Ticker(ticker)

  # historyPrices = yf.download(tickers = ticker,
  #                    period = "max",
  #                    interval = "1d")
  historyPrices = ticker_obj.history(
                     period = "max",
                     interval = "1d")

  # generate features for historical prices, and what we want to predict
  historyPrices['Ticker'] = ticker
  historyPrices['Year']= historyPrices.index.year
  historyPrices['Month'] = historyPrices.index.month
  historyPrices['Weekday'] = historyPrices.index.weekday
  historyPrices['Date'] = historyPrices.index.date

  # historical returns
  for i in [1,3,7,30,90,365]:
    historyPrices['growth_'+str(i)+'d'] = historyPrices['Close'] / historyPrices['Close'].shift(i)
  historyPrices['growth_future_30d'] = historyPrices['Close'].shift(-30) / historyPrices['Close']

  # Technical indicators
  # SimpleMovingAverage 10 days and 20 days
  historyPrices['SMA10']= historyPrices['Close'].rolling(10).mean()
  historyPrices['SMA20']= historyPrices['Close'].rolling(20).mean()
  historyPrices['growing_moving_average'] = np.where(historyPrices['SMA10'] > historyPrices['SMA20'], 1, 0)
  historyPrices['high_minus_low_relative'] = (historyPrices.High - historyPrices.Low) / historyPrices['Close']

  # 30d rolling volatility : https://ycharts.com/glossary/terms/rolling_vol_30
  historyPrices['volatility'] =   historyPrices['Close'].rolling(30).std() * np.sqrt(252)

  # what we want to predict
  historyPrices['is_positive_growth_30d_future'] = np.where(historyPrices['growth_future_30d'] > 1, 1, 0)

  # sleep 1 sec between downloads - not to overload the API server
  time.sleep(1)


  if stocks_df.empty:
    stocks_df = historyPrices
  else:
    stocks_df = pd.concat([stocks_df, historyPrices], ignore_index=True)

def get_ticker_type(ticker:str, us_stocks_list, eu_stocks_list, india_stocks_list):
  if ticker in us_stocks_list:
    return 'US'
  elif ticker in eu_stocks_list:
    return 'EU'
  elif ticker in india_stocks_list:
    return 'INDIA'
  else:
    return 'ERROR'
  
stocks_df['ticker_type'] = stocks_df.Ticker.apply(lambda x:get_ticker_type(x, US_STOCKS, EU_STOCKS, INDIA_STOCKS))
stocks_df['Date'] = pd.to_datetime(stocks_df['Date'])

ind = pd.DataFrame()
tind = pd.DataFrame()
for ticker in stocks_df['Ticker'].unique():
    tind['rsi'] = talib.RSI(stocks_df.loc[stocks_df['Ticker'] == ticker, 'Close'], 14)
    ind = pd.concat([ind,tind])
    tind = tind.iloc[0:0]

stocks_df = pd.merge(stocks_df, ind, left_index=True, right_index=True)

rsi_threshold = 25
selected_df = stocks_df[
    (stocks_df['rsi'] < rsi_threshold) &
    (stocks_df['Date'] >= '2000-01-01') &
    (stocks_df['Date'] <= '2025-06-01')
]

net_income = 1000 * (selected_df['growth_future_30d'] - 1).sum()
print(selected_df.describe())
print(selected_df['Date'].describe())
print(net_income)