import pandas as pd

url_sp500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
df_sp500 = pd.read_html(url_sp500)[0]
df_sp500['Year added']=df_sp500['Date added'].str[:4]
print(df_sp500['Year added'].value_counts()[:5])

