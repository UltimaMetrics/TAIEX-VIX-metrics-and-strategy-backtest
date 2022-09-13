# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 10:04:50 2022

@author: sigma
"""



import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np
import pandas as pd

import scipy.stats
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


import requests
import datetime
from statsmodels.tsa.stattools import adfuller



import Backtest2 as bt

#import back_testing 



df=pd.read_excel(r'D:\Derivatives Trading\ResearchRecord.xlsx')

df["Date"] = df["Date"].astype("datetime64")
dataframe = df.set_index("Date")
date=df["Date"]
date

#timeseries1 ={'Date':df["Date"], 'Spot':df["Spot"], 'IV':df["Implied Volatility"], 'pnl': df["PNL Index"]}


daily_return_futures=df["Futures"].pct_change()

rolling_skew_futures=df["Futures"].rolling(20).skew()
rolling_skew_futures

rolling_kurt_futures=df["Futures"].rolling(20).kurt()
rolling_kurt_futures

#calculate 20-day historical volatility
rolling_sd_futures_return_20=daily_return_futures.rolling(20).std()
rolling_sd_futures_return_20

volatility_20_day=rolling_sd_futures_return_20*(250**0.5)
volatility_20_day
list(volatility_20_day)
plt.plot(volatility_20_day)

#calculate 5-day historical volatility
rolling_sd_futures_return_5=daily_return_futures.rolling(5).std()
rolling_sd_futures_return_5

volatility_5_day=rolling_sd_futures_return_5*(250**0.5)
volatility_5_day
list(volatility_5_day)
plt.plot(volatility_5_day)


#Alternative method to calculate historical volatility
#source: https://www.learnpythonwithrune.org/calculate-the-volatility-of-historic-stock-prices-with-pandas-and-python/
log_return=np.log(df['Futures']/df['Futures'].shift())
volatility=log_return.std()*(252**0.5)
volatility



# Check for unit root process
def stationarity(data, cutoff=0.05):
    if adfuller(data)[1] < cutoff:
        print('The series is stationary')
        print('p-value = ', adfuller(data)[1])
    else:
        print('The series is NOT stationary')
        print('p-value = ', adfuller(data)[1])
        


timeseries2 ={'date':df["Date"], 'skew':rolling_skew_futures, 'kurt':rolling_kurt_futures, 'IV':df["Implied Volatility"], 'pnl':df["PNL Index"] }


stationarity(df["Futures"])

stationarity(df["Implied Volatility"])

#Negative skewness: frequent small gains and few extreme or significant losses in the time period considered.
#Positive skewness: frequent small losses and few extreme gains

skew=timeseries2["skew"] 
kurt=timeseries2["kurt"]
iv=timeseries2["IV"]
pnl=timeseries2["pnl"]
pnl
date=timeseries2["date"]

return_pnl=pnl.pct_change()

#Statistical distribution
summary_kurt=kurt.describe()
summary_kurt
                

summary_iv=iv.describe()
summary_iv

summary_TX=df["Futures"].describe()
summary_TX



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
iv_train, iv_test, pnl_train, pnl_test = train_test_split(iv, pnl, test_size = 1/3, random_state = 0)


#Regression
reg=LinearRegression()
reg.fit(pnl_train, iv_train)


#Predict the test set result
iv_test

pnl_pred=reg.predict(iv_test)

# Visualising 
plt.scatter(iv_train, pnl_train, color = 'purple')
plt.plot(iv_train, reg.predict(iv_train), color = 'blue')


# Visualising the Test set results
plt.scatter(iv_test, pnl_test, color = 'pink')
plt.plot(iv_train, reg.predict(iv_train), color = 'blue')
plt.title('VIX vs PnL (Test set)')
plt.xlabel('VIX')
plt.ylabel('Pnl')
plt.show()



#Bollinger bands
period=20
iv_20=iv.rolling(period).mean()
std_iv=iv.rolling(period).std()
iv_upper=iv_20+(2*std_iv)
iv_lower=iv_20-(2*std_iv)

BU=iv_upper
BL=iv_lower


# plot Bollinger band for IV
plt.figure(figsize=(15,10))
plt.title('Bollinger Bands chart ')
plt.plot(date, iv)
plt.plot(date, BU, alpha=0.3)
plt.plot(date, BL, alpha=0.3)
plt.plot(date, iv_20, alpha=0.3)
plt.fill_between(date, BU, BL, color='grey', alpha=0.1)

plt.show()


#Buy signal
df["signal"]=np.where(iv<iv_20, 1, np.nan)
#Sell signal
df["signal"]=np.where(iv>iv_20, -1,df["signal"])
#buy/sell next trading day
buy_date=df["signal"].shift()
sell_date=df["signal"].fillna(0)


#Backtesting strategy

close= df["Implied Volatility"]
bt.backtest_dataframe(df)


plt.figure(figsize=(12,5))
plt.xticks(rotation=45)
plt.plot( df['Date'], close)
plt.scatter(df[(df["signal"] == 1)]['buy_date'], df[(df["signal"] == 1)][close], label = 'Buy', marker='^', c='g')
plt.scatter(df[(df["signal"] == -1)]['sell_date'], df[(df["signal"] == -1)][close], label = 'Sell', marker='v', c='r')

plt.title('Price Chart & Historical Trades', fontweight="bold")
plt.legend()
plt.show()



fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(date, iv, 'crimson')
ax2.plot(date, kurt, 'royalblue')
ax1.set_ylabel("TAIEX VIX")
ax2.set_ylabel("Kurtosis")


fig, ax3 = plt.subplots()
ax4 = ax3.twinx()
ax3.plot(date, iv, 'crimson')
ax4.plot(date, skew, 'royalblue')
ax3.set_ylabel("TAIEX VIX")
ax4.set_ylabel("skewness")



fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(date, skew, 'royalblue')
ax2.plot(date, kurt, 'violet')
ax1.set_ylabel("Skewness")
ax2.set_ylabel("Kurtosis")


fig, ax5 = plt.subplots()
ax6 = ax5.twinx()
ax5.plot(date, iv, 'crimson')
ax6.plot(date, pnl, 'royalblue')
ax5.set_ylabel("TAIEX VIX")
ax6.set_ylabel("PnL")


#Regression
iv_ols=sm.add_constant(iv)
model=sm.OLS(pnl, iv_ols).fit()
model.summary()


#Strategy based on z-score 
data_mavg_40_day=iv.rolling(window=40).mean() #moving average of IV

std_iv=iv.rolling(window=40).std()

#z-score
z_score=(iv-data_mavg_40_day)/std_iv
z_score

plt.figure(figsize=(10, 5))
z_score.plot()
plt.axhline(0, color='black')
plt.axhline(2.0, color='red', linestyle='--')
plt.axhline(-2.0, color='green', linestyle='--')
plt.legend(['Z-Score', 'Mean', '+2', '-2'])
plt.show()

#Entry and exit
# Define threshold
#threshold = 2

threshold_for_short_IV=1.7
threshold_for_long_IV=-1.3

# Long entry
long_positions = np.where(z_score < threshold_for_long_IV,1,0)

# Long exit
long_positions = np.where(z_score >= 2.1, 0, long_positions)

# Short entry
short_positions = np.where(z_score >threshold_for_short_IV,-1,0)

# Short exit
short_positions = np.where(z_score <= -1.2, 0, short_positions)

# Combine the positions


df['positions'] = long_positions + short_positions

df['positions'].shift(1)

# Fill NaN values
FNV = iv.fillna(method='ffill')


# Calculate returns
returns_2 = iv - iv.shift(1)
returns_2

# Calculate strategy returns
strategy_returns =  df['positions'].shift(1) * returns_2
strategy_returns 

# Calculate pnl
pnl_calc = strategy_returns.cumsum()
pnl_calc

# Plot the strategy returns
pnl_calc.plot(figsize=(12,7))
plt.xlabel('Date')
plt.ylabel('PnL')
plt.show()
