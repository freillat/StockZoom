import pandas as pd
import yfinance as yf

# Question 1

url_sp500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
df_sp500 = pd.read_html(url_sp500)[0]
df_sp500['Year added']=df_sp500['Date added'].str[:4]
print(df_sp500['Year added'].value_counts()[:5])

# Question 2

start_date='2025-01-01'
end_date='2025-05-01'

indices = [
[ 'United States', 'S&P 500', '^GSPC' ],
[ 'China', 'Shanghai Composite', '000001.SS' ],
[ 'Hong Kong', 'HANG SENG INDEX', '^HSI' ],
[ 'Australia', 'S&P/ASX 200', '^AXJO' ],
[ 'India', 'Nifty 50', '^NSEI' ],
[ 'Canada', 'S&P/TSX Composite', '^GSPTSE' ],
[ 'Germany', 'DAX', '^GDAXI' ],
[ 'United Kingdom', 'FTSE 100', '^FTSE' ],
[ 'Japan', 'Nikkei 225', '^N225' ],
[ 'Mexico', 'IPC Mexico', '^MXX' ],
[ 'Brazil', 'Ibovespa', '^BVSP' ]
]
df_indices = pd.DataFrame(indices)
df_indices.columns = [ 'Country', 'Name', 'Ticker']

for i in range(0,df_indices['Ticker'].shape[0]):
    ticker = df_indices['Ticker'].iloc[i]
    ticker_obj = yf.Ticker(ticker)
    daily = ticker_obj.history(start=start_date,end=end_date)
    open = daily['Open'].iloc[0]
    close = daily['Close'].iloc[daily.shape[0]-1]
    ytd_return = ( close - open ) / open
    print(df_indices['Name'][i], "{:.2%}".format(ytd_return))

    # Question 3