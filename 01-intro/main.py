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
ticker = '^GSPC'
ticker_obj = yf.Ticker(ticker)
start_date='1950-01-01'
daily = ticker_obj.history(start=start_date)

# rows = daily.shape[0]
# ath_list = [ 0 ]
# for i in range(1, rows-1):
#     if daily['High'].iloc[i]>daily['High'].iloc[ath_list[-1]]:
#         ath_list.append(i)
# data = []
# for i in range(0,len(ath_list)-1):
#     start=ath_list[i]
#     end=ath_list[i+1]
#     low = daily['Low'].iloc[start]
#     low_idx= 0
#     for j in range(start, end):
#         if daily['Low'].iloc[j]<low:
#             low = daily['Low'].iloc[j]
#             low_idx=j
#     high = daily['High'].iloc[ath_list[i]]
#     drawdown = (high - low) / high
#     start_date = daily.index[start].date()
#     end_date = daily.index[end].date()
#     low_date = daily.index[low_idx].date()
#     data.append([start, end, low_idx ,low, high, drawdown, start_date, end_date, low_date])
# df = pd.DataFrame(data)
# df.columns = [ 'Start', 'End', 'LowIdx', 'Low', 'High', 'Drawdown', 'StartDate', 'EndDate', 'LowDate']
# df['Duration']=df['LowDate']-df['StartDate']
# df_filter=df[df['Drawdown']>0.05]
# print(df_filter['Duration'].quantile(0.5))

rows = daily.shape[0]
ath_list = [ 0 ]
for i in range(1, rows-1):
    if daily['Close'].iloc[i]>daily['Close'].iloc[ath_list[-1]]:
        ath_list.append(i)
data = []
for i in range(0,len(ath_list)-1):
    start=ath_list[i]
    end=ath_list[i+1]
    low = daily['Close'].iloc[start]
    low_idx= 0
    for j in range(start, end):
        if daily['Close'].iloc[j]<low:
            low = daily['Close'].iloc[j]
            low_idx=j
    high = daily['Close'].iloc[ath_list[i]]
    drawdown = (high - low) / high
    start_date = daily.index[start].date()
    end_date = daily.index[end].date()
    low_date = daily.index[low_idx].date()
    data.append([start, end, low_idx ,low, high, drawdown, start_date, end_date, low_date])
df = pd.DataFrame(data)
df.columns = [ 'Start', 'End', 'LowIdx', 'Low', 'High', 'Drawdown', 'StartDate', 'EndDate', 'LowDate']
df['Duration']=df['LowDate']-df['StartDate']
df_filter=df[df['Drawdown']>0.05]
print(df_filter['Duration'].quantile(0.5))

# Question 4