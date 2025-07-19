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
# print(df_full.info())
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
# print(f'Technical patterns count = {len(TECHNICAL_PATTERNS)}, examples = {TECHNICAL_PATTERNS[0:5]}')
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
filtered_df.loc[:,'abs_corr'] = filtered_df['is_positive_growth_30d_future'].abs()
print(filtered_df.sort_values(by='abs_corr', ascending=False).head(5))

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

new_df = temporal_split(df_with_dummies, df_with_dummies['Date'].min(), df_with_dummies['Date'].max()).copy()
print(new_df.groupby(['split'])['Date'].agg({'min','max','count'}))

new_df['pred0_manual_cci'] = (new_df.cci>200).astype(int)
new_df['pred1_manual_prev_g1'] = (new_df.growth_30d>1).astype(int)
new_df['pred2_manual_prev_g1_and_snp'] = ((new_df['growth_30d'] > 1) & (new_df['growth_snp500_30d'] > 1)).astype(int)

new_df['pred3_manual_dgs10_5'] = ((new_df['DGS10'] <= 4) & (new_df['DGS5'] <= 1)).astype(int)
new_df['pred4_manual_dgs10_fedfunds'] = ((new_df['DGS10'] > 4) & (new_df['FEDFUNDS'] <= 4.795)).astype(int)

# # Check the correlation of the new predictions with the target variable
# corr_pred3 = new_df['pred3_manual_dgs10_5'].corr(new_df['is_positive_growth_30d_future'])
# corr_pred4 = new_df['pred4_manual_dgs10_fedfunds'].corr(new_df['is_positive_growth_30d_future'])
# print(f'Correlation of pred3 with is_positive_growth_30d_future: {corr_pred3}')
# print(f'Correlation of pred4 with is_positive_growth_30d_future: {corr_pred4}')

PREDICTIONS = [k for k in new_df.keys() if k.startswith('pred')]
# print(PREDICTIONS)

# generate columns is_correct_
for pred in PREDICTIONS:
  part1 = pred.split('_')[0] # first prefix before '_'
  new_df[f'is_correct_{part1}'] =  (new_df[pred] == new_df.is_positive_growth_30d_future).astype(int)

IS_CORRECT =  [k for k in new_df.keys() if k.startswith('is_correct_')]
# print(IS_CORRECT)

# define "Precision" for ALL predictions on a Test dataset (~4 last years of trading)
for i,column in enumerate(IS_CORRECT):
  prediction_column = PREDICTIONS[i]
  is_correct_column = column
  filter = (new_df.split=='test') & (new_df[prediction_column]==1)
  print(f'Prediction column:{prediction_column} , is_correct_column: {is_correct_column}')
  print(new_df[filter][is_correct_column].value_counts())
  print(new_df[filter][is_correct_column].value_counts()/len(new_df[filter]))

  print('---------')

# Question 3

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
all_df = new_df.copy(deep=True)

# ONLY numerical Separate features and target variable for training and testing sets
# need Date and Ticker later when merging predictions to the dataset
X_train = train_df[features_list+[to_predict,'Date','Ticker']]
X_test = test_df[features_list+[to_predict,'Date','Ticker']]
X_all = all_df[features_list+[to_predict,'Date','Ticker']]

print(f'length: X_train {X_train.shape},  X_test {X_test.shape}, X_all {X_all.shape}')

# Can't have +-inf values . E.g. ln(volume)=-inf when volume==0 => substitute with 0

# Disable SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'

X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
X_all.replace([np.inf, -np.inf], np.nan, inplace=True)

# Need to fill NaNs somehow
X_train.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)
X_all.fillna(0, inplace=True)

print(f'length: X_train_imputed {X_train.shape},  X_test_imputed {X_test.shape}, X_all_imputed {X_all.shape}')

X_train_imputed = X_train # we won't use outliers removal to save more data to train: remove_outliers_percentile(X_train)
X_test_imputed = X_test # we won't use outliers removal to save more data to test: remove_outliers_percentile(X_test)
X_all_imputed = X_all # we won't use outliers removal to save more data to train: remove_outliers_percentile(X_all)

# same shape
print(f'length: X_train_imputed {X_train_imputed.shape},  X_test_imputed {X_test_imputed.shape}, X_all_imputed {X_all_imputed.shape}')

y_train = X_train_imputed[to_predict]
y_test = X_test_imputed[to_predict]
y_all = X_all_imputed[to_predict]

# remove y_train, y_test from X_ dataframes
del X_train_imputed[to_predict]
del X_test_imputed[to_predict]
del X_all_imputed[to_predict]

clf = DecisionTreeClassifier(max_depth=10, random_state=42) 
clf.fit(X_train_imputed.drop(['Date','Ticker'],axis=1), y_train)

new_df['pred5_clf_10'] = clf.predict(X_all_imputed.drop(['Date','Ticker'],axis=1))
new_df['is_correct_pred5'] =  (new_df['pred5_clf_10'] == new_df[to_predict]).astype(int)
new_df['only_pred5_is_correct'] = ((new_df['is_correct_pred5'] ==1) & (new_df['is_correct_pred0']==0) & (new_df['is_correct_pred1']==0) & (new_df['is_correct_pred2']==0) & (new_df['is_correct_pred3']==0) & (new_df['is_correct_pred4']==0)).astype(int)
filter = (new_df.split=='test') & (new_df['only_pred5_is_correct']==1)
print(new_df[filter]['only_pred5_is_correct'].value_counts())

# Question 4

best_max_depth = 0
best_score = 0

for max_depth in range(1,21):
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42) 
    clf.fit(X_train_imputed.drop(['Date','Ticker'],axis=1), y_train)
    y_pred = clf.predict(X_test_imputed.drop(['Date','Ticker'],axis=1))
    TP = np.sum((y_pred == 1) & (y_test == 1))
    FP = np.sum((y_pred == 1) & (y_test == 0))
    precision_score = TP / (TP + FP) if (TP + FP) > 0 else 0
    print(f'Max depth: {max_depth}, Precision score: {precision_score}')
    if precision_score > best_score:
        best_score = precision_score
        best_max_depth = max_depth

print(f'Best max depth: {best_max_depth}, Best precision score: {best_score}')