# IMPORTS
import numpy as np
import pandas as pd

#Fin Data Sources
import yfinance as yf
# import pandas_datareader as pdr

# #Data viz
# import plotly.graph_objs as go
# import plotly.graph_objects as go
# import plotly.express as px

import time
from datetime import date

# for graphs
# import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

df_full = pd.read_parquet("./content/stocks_df_combined_2025_06_13.parquet.brotli")
print(df_full.info())
# growth indicators (but not future growth)
GROWTH = [g for g in df_full.keys() if (g.find('growth_')==0)&(g.find('future')<0)]
# leaving only Volume ==> generate ln(Volume)
OHLCV = ['Open','High','Low','Close','Adj Close_x','Volume']
CATEGORICAL = ['Month', 'Weekday', 'Ticker', 'ticker_type']
TO_PREDICT = [g for g in df_full.keys() if (g.find('future')>=0)]
TO_DROP = ['Year','Date','index_x', 'index_y', 'index', 'Quarter','Adj Close_y'] + CATEGORICAL + OHLCV
# let's define on more custom numerical features
df_full['ln_volume'] = df_full.Volume.apply(lambda x: np.log(x))
# manually defined features
CUSTOM_NUMERICAL = ['SMA10', 'SMA20', 'growing_moving_average', 'high_minus_low_relative','volatility', 'ln_volume']

# All Supported Ta-lib indicators: https://github.com/TA-Lib/ta-lib-python/blob/master/docs/funcs.md
TECHNICAL_INDICATORS = ['adx', 'adxr', 'apo', 'aroon_1','aroon_2', 'aroonosc',
 'bop', 'cci', 'cmo','dx', 'macd', 'macdsignal', 'macdhist', 'macd_ext',
 'macdsignal_ext', 'macdhist_ext', 'macd_fix', 'macdsignal_fix',
 'macdhist_fix', 'mfi', 'minus_di', 'mom', 'plus_di', 'dm', 'ppo',
 'roc', 'rocp', 'rocr', 'rocr100', 'rsi', 'slowk', 'slowd', 'fastk',
 'fastd', 'fastk_rsi', 'fastd_rsi', 'trix', 'ultosc', 'willr',
 'ad', 'adosc', 'obv', 'atr', 'natr', 'ht_dcperiod', 'ht_dcphase',
 'ht_phasor_inphase', 'ht_phasor_quadrature', 'ht_sine_sine', 'ht_sine_leadsine',
 'ht_trendmod', 'avgprice', 'medprice', 'typprice', 'wclprice']
TECHNICAL_PATTERNS = [g for g in df_full.keys() if g.find('cdl')>=0]
print(f'Technical patterns count = {len(TECHNICAL_PATTERNS)}, examples = {TECHNICAL_PATTERNS[0:5]}')
MACRO = ['gdppot_us_yoy', 'gdppot_us_qoq', 'cpi_core_yoy', 'cpi_core_mom', 'FEDFUNDS',
 'DGS1', 'DGS5', 'DGS10']
NUMERICAL = GROWTH + TECHNICAL_INDICATORS + TECHNICAL_PATTERNS + CUSTOM_NUMERICAL + MACRO
# CHECK: NO OTHER INDICATORS LEFT
OTHER = [k for k in df_full.keys() if k not in OHLCV + CATEGORICAL + NUMERICAL + TO_DROP]
df_full.Ticker.nunique()
# tickers, min-max date, count of daily observations
# print(df_full.groupby(['Ticker'])['Date'].agg(['min','max','count']))

df_full['Month_name'] = df_full['Date'].dt.strftime("%B")
df_full['wom'] = df_full['Date'].apply(lambda d: (d.day-1) // 7 + 1)
df_full['month_wom'] = df_full['Month_name']+'_w'+df_full['wom'].astype(str)
df_full.drop(columns=['wom','Month_name'], inplace=True)
CATEGORICAL = ['Month', 'Weekday', 'Ticker', 'ticker_type', 'month_wom']
# dummy variables are not generated from Date and numeric variables
df_full.loc[:,'Month'] = df_full.Month.dt.strftime('%B')
df_full.loc[:,'Weekday'] = df_full.Weekday.astype(str)
# Generate dummy variables (no need for bool, let's have int32 instead)
df = df_full[df_full.Date>='2000-01-01']

dummy_variables = pd.get_dummies(df[CATEGORICAL], dtype='int32')
# get dummies names in a list
DUMMIES = dummy_variables.keys().to_list()
# Concatenate the dummy variables with the original DataFrame



df_with_dummies = pd.concat([df, dummy_variables], axis=1)
corr_is_positive_growth_30d_future = df_with_dummies[NUMERICAL+DUMMIES+TO_PREDICT].corr()['is_positive_growth_30d_future']
corr_is_positive_growth_30d_future_df = pd.DataFrame(corr_is_positive_growth_30d_future)
# print(corr_is_positive_growth_30d_future_df.info())
# Filter the correlation results to include only the dummy variables generated from month_wom
prefix_to_filter = 'month_wom'
filtered_df = corr_is_positive_growth_30d_future_df[corr_is_positive_growth_30d_future_df.index.str.startswith(prefix_to_filter)]
filtered_df['abs_corr'] = filtered_df['is_positive_growth_30d_future'].abs()
print(filtered_df.sort_values(by='abs_corr', ascending=False).head(10))

# Question 2

def temporal_split(df, min_date, max_date, train_prop=0.7, val_prop=0.15, test_prop=0.15):
    """
    Splits a DataFrame into three buckets based on the temporal order of the 'Date' column.

    Args:
        df (DataFrame): The DataFrame to split.
        min_date (str or Timestamp): Minimum date in the DataFrame.
        max_date (str or Timestamp): Maximum date in the DataFrame.
        train_prop (float): Proportion of data for training set (default: 0.6).
        val_prop (float): Proportion of data for validation set (default: 0.2).
        test_prop (float): Proportion of data for test set (default: 0.2).

    Returns:
        DataFrame: The input DataFrame with a new column 'split' indicating the split for each row.
    """
    # Define the date intervals
    train_end = min_date + pd.Timedelta(days=(max_date - min_date).days * train_prop)
    val_end = train_end + pd.Timedelta(days=(max_date - min_date).days * val_prop)

    # Assign split labels based on date ranges
    split_labels = []
    for date in df['Date']:
        if date <= train_end:
            split_labels.append('train')
        elif date <= val_end:
            split_labels.append('validation')
        else:
            split_labels.append('test')

    # Add 'split' column to the DataFrame
    df['split'] = split_labels

    return df

df_with_dummies = temporal_split(df_with_dummies, df_with_dummies['Date'].min(), df_with_dummies['Date'].max())
new_df = df_with_dummies.copy()
print(new_df.groupby(['split'])['Date'].agg({'min','max','count'}))

new_df['pred3_manual_dgs10_5'] = (new_df['DGS10'] <= 4.825) & (new_df['DGS5'] <= 0.745)
new_df['pred4_manual_dgs10_fedfunds'] = (new_df['DGS10'] > 4.825) & (new_df['FEDFUNDS'] <= 4.795)

# Check the correlation of the new predictions with the target variable
corr_pred3 = new_df['pred3_manual_dgs10_5'].corr(new_df['is_positive_growth_30d_future'])
corr_pred4 = new_df['pred4_manual_dgs10_fedfunds'].corr(new_df['is_positive_growth_30d_future'])
print(f'Correlation of pred3 with is_positive_growth_30d_future: {corr_pred3}')
print(f'Correlation of pred4 with is_positive_growth_30d_future: {corr_pred4}')

new_df['is_correct_pred3'] = (new_df.pred3_manual_dgs10_5 == new_df.is_positive_growth_30d_future)
new_df['is_correct_pred4'] = (new_df.pred4_manual_dgs10_fedfunds == new_df.is_positive_growth_30d_future)

print(new_df.groupby(['split'])[['is_correct_pred3', 'is_correct_pred4']].sum())
print(new_df.groupby(['split'])[['is_correct_pred3', 'is_correct_pred4']].count())

TP = new_df[new_df['split']=='test'].is_correct_pred3.sum()
TPFP = new_df[new_df['split']=='test'].is_correct_pred3.count()
result = TP / TPFP
print(f"{result:.3f}")
filter = (new_df.split=='test') & (new_df.pred0_manual_cci==1)
new_df[filter].is_correct_prediction.value_counts()

# Question 3

new_df['pred0_manual_cci'] = (new_df.cci>200).astype(int)
new_df['pred1_manual_prev_g1'] = (new_df.growth_30d>1).astype(int)
new_df['pred2_manual_prev_g1_and_snp'] = ((new_df['growth_30d'] > 1) & (new_df['growth_snp500_30d'] > 1)).astype(int)

def remove_infinite_values(X):
    """
    Remove infinite values from the input array.

    Parameters:
    - X: Input array (NumPy array or array-like)

    Returns:
    - Array with infinite values removed
    """
    return X[np.isfinite(X).all(axis=1)]

# Example usage:
# Assuming X is your input data
# filtered_X = remove_infinite_values(X)

# Split the data into training and testing sets based on the split date
features_list = NUMERICAL+DUMMIES
to_predict = 'is_positive_growth_30d_future'

train_df = new_df[new_df.split.isin(['train','validation'])].copy(deep=True)
test_df = new_df[new_df.split.isin(['test'])].copy(deep=True)

# ONLY numerical Separate features and target variable for training and testing sets
# need Date and Ticker later when merging predictions to the dataset
X_train = train_df[features_list+[to_predict,'Date','Ticker']]
X_test = test_df[features_list+[to_predict,'Date','Ticker']]

print(f'length: X_train {X_train.shape},  X_test {X_test.shape}')

# Can't have +-inf values . E.g. ln(volume)=-inf when volume==0 => substitute with 0

# Disable SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'

X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

# Need to fill NaNs somehow
X_train.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)

print(f'length: X_train_imputed {X_train.shape},  X_test_imputed {X_test.shape}')

X_train_imputed = X_train # we won't use outliers removal to save more data to train: remove_outliers_percentile(X_train)
X_test_imputed = X_test # we won't use outliers removal to save more data to test: remove_outliers_percentile(X_test)

# same shape
print(f'length: X_train_imputed {X_train_imputed.shape},  X_test_imputed {X_test_imputed.shape}')

y_train = X_train_imputed[to_predict]
y_test = X_test_imputed[to_predict]

# remove y_train, y_test from X_ dataframes
del X_train_imputed[to_predict]
del X_test_imputed[to_predict]

clf = DecisionTreeClassifier(max_depth=10, random_state=42) 
clf.fit(X_train_imputed.drop(['Date','Ticker'],axis=1), y_train)
X_test_imputed.drop(['Date','Ticker'],axis=1), y_test
y_pred = clf.predict(X_test_imputed.drop(['Date','Ticker'],axis=1))
result_df = pd.concat([X_test_imputed.drop(['Date','Ticker'],axis=1), y_test, pd.Series(y_pred, index=X_test_imputed.drop(['Date','Ticker'],axis=1).index, name='pred_')], axis=1)
print(result_df.pred_.value_counts())

PREDICTIONS = [k for k in new_df.keys() if k.startswith('pred')]
# generate columns is_correct_
for pred in PREDICTIONS:
  part1 = pred.split('_')[0] # first prefix before '_'
  new_df[f'is_correct_{part1}'] =  (new_df[pred] == new_df.is_positive_growth_30d_future).astype(int)

IS_CORRECT =  [k for k in new_df.keys() if k.startswith('is_correct_')]