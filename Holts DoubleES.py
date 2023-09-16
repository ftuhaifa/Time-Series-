# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 11:12:41 2022

@author: ftuha
"""

import pandas as pd
import numpy as np

from pandas.plotting import autocorrelation_plot
from matplotlib import pyplot
import statsmodels.graphics.tsaplots as tsa
from statsmodels.stats.diagnostic import acorr_ljungbox
import pymannkendall as mk
from statsmodels.tsa.stattools import adfuller
#from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt

# dataframe opertations - pandas
import pandas as pd
# plotting data - matplotlib
from matplotlib import pyplot as plt
# time series - statsmodels 
# Seasonality decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import seasonal_decompose 
# holt winters 
# single exponential smoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing   
# double and triple exponential smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import statsmodels.api as sm

import warnings 
warnings.filterwarnings('ignore')

#%matplotlib inline


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.api import Holt


df = pd.read_csv('YearOfDeath01.csv')

df['Year'] = pd.to_datetime(df['Year'], format='%Y')

df['Year'] = df['Year'].dt.date
#airline['Year'] = airline['Year'].date()

# Updating the header
df.columns=["Year","Rate"]

#print(df.head())

df.set_index('Year',inplace=True)



#print(df[20:])

# Split into train and test set
train = df[:80]
test = df[20:]



#train = df[:33]
#test = df[-11:]



print("== train ===")

print(train)

print("== test ==")
print(test)
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
y_hat_avg = test.copy()


result = {}
a = 0.1
b = 0.1
for a1 in range (10):
    b = 0.1
    for b1 in range (10):

        
        
        holt = Holt(np.asarray(df['Rate'])).fit(smoothing_level = a,
                                                   smoothing_slope = b
                                                   )
        result[(a,b)] = holt.aic
        b= b + 0.1
    a = a + 0.1

a,b = min(result, key=result.get)

print(a)
print(b)
print(result)

#smoothing_level = 0.1, smoothing_slope = 0.5
#smoothing_level = 0.4, smoothing_slope = 0.9

fit1 = Holt(np.asarray(train['Rate'])).fit(smoothing_level = 0.4,
                                           smoothing_slope = 0.9
                                           )

print(fit1.summary())

# 33 and 43

prediction = fit1.predict(start= 20 , end= 43)
test ["Prediction"] = fit1.predict(start= 20 , end= 43)

test ["Prediction"]= test ["Prediction"].astype(int)
print(test)



prediction = prediction.astype(int)
print (prediction)

y_hat_avg['Holt_linear'] = fit1.forecast(24)

forcast = fit1.forecast(15)
print(forcast)

from sklearn.metrics import mean_absolute_error,mean_squared_error

print("******************************************************************")
print("******************************************************************")
print(" ")
print("EVALUATION")
print(" ")
print("******************************************************************")
print("******************************************************************")

mae = mean_absolute_error(test['Rate'], prediction)


print ("===========================================================")
mape = mean_absolute_percentage_error(test['Rate'], prediction)

MSE = mean_squared_error(test['Rate'], prediction)
rmse = np.sqrt(mean_squared_error(test['Rate'], prediction))

print ("===========================================================")


print ("The MAE: ", mae)
print ("===========================================================")
print("The MAPE: ", mape)

print ("===========================================================")
print ("The RMSE: ",rmse)





print("==== Ploting =====")

plt.figure(figsize=(16,8))
plt.plot(train['Rate'], label='Train')
plt.plot(test['Rate'], label='Test')
plt.plot(test["Prediction"], label='Prediction')
#plt.plot(y_hat_avg['Holt_linear'], label='Holt_linear')
plt.legend(loc='best')
plt.show()






