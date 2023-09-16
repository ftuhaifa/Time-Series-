# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 15:50:09 2022

@author: ftuha
"""

import pmdarima as pm
from pmdarima.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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


import statsmodels.api as sm

from prophet import Prophet

import warnings 
warnings.filterwarnings('ignore')



from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error



# Load/split your data
df = pd.read_csv('df.csv')





df['ds'] = pd.to_datetime(df['ds'], format='%Y')

#df['ds'] = df['ds'].dt.date
df['y'] = pd.to_numeric(df['y'])

df['y']= df['y'].astype(np.float64)

df.rename(columns = {'ds':'ds', "y":'y'}, inplace = True)
#Converting the Year to Date type
#df = df.set_index('ds')
print (df.head())
print(df.dtypes)

#spliting the data to train and test data
train = df[:33]
test = df[-11:]


# Python
m = Prophet()
m.fit(df)



# Python
future = m.make_future_dataframe(periods=0)
print(future.tail())


# Python
forecast = m.predict(future)
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())



# Python
fig1 = m.plot(forecast)

# Python
fig2 = m.plot_components(forecast)


# Python
from prophet.plot import plot_plotly, plot_components_plotly

plot_plotly(m, forecast)


# Python
plot_components_plotly(m, forecast)


prediction = forecast['yhat']
prediction = prediction [-11:]
print("EVALUATION")
print(" ")
print("******************************************************************")
print("******************************************************************")

mae = mean_absolute_error(test['y'], prediction)


print ("===========================================================")
mape = mean_absolute_percentage_error(test['y'], prediction)

MSE = mean_squared_error(test['y'], prediction)
rmse = np.sqrt(mean_squared_error(test['y'], prediction))

print ("===========================================================")


print ("The MAE: ", mae)
print ("===========================================================")
print("The MAPE: ", mape)

print ("===========================================================")
print ("The RMSE: ",rmse)




#https://facebook.github.io/prophet/docs/quick_start.html
#https://towardsdatascience.com/facebook-prophet-for-time-series-forecasting-in-python-part1-d9739cc79b1d



