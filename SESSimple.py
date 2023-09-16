# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 02:17:03 2022

@author: ftuha
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 01:48:40 2022

@author: ftuha
"""




import numpy as np
import pandas as pd
#import yfinance as yf
import datetime 
import matplotlib.pyplot as plt
from statsmodels.tsa.api import SimpleExpSmoothing

import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error

data = pd.read_csv('YearOfDeath01.csv')
data.head()

print(data.describe())

data['Year'] = pd.to_datetime(data['Year'], format='%Y')

#data['Year'] = data['Year'].dt.date
#df['Rate of death'] = pd.to_numeric(df['Rate of death'])
train_data = data[0:len(data)-33]
test_data = data[len(data)- 11:]


#train_data = data[:80]
#test_data = data[:20]



#print(test_data.head(3))
today = datetime.date.today()
plt.figure(figsize=(10,6))
plt.title('MICROSOFT STOCK FOR LAST 1 MONTH AS ON'+ str(today))
plt.plot(data["Rate"])
plt.show()
data = data["Rate"].tolist()
#CHANGE DATA FOR 30 DAYS FROM THE DATE WHEN YOU ARE RUNNING THIS CODE
#index= pd.date_range(start= "2010-01-01", end= "2010-02-13")

index = pd.date_range(start="1975", end="2019", freq="A")
stock_data = pd.Series(data, index)


print ("==== train train =====")
print(stock_data[0:33])
#stock_data01 = pd.Series(train_data, index)

forecast_timestep = 15

fit_4 = SimpleExpSmoothing(stock_data[0:33], initialization_method="heuristic").fit(smoothing_level= 0.1,optimized=False)

#prediction = fit_4.predict(start=len(train_data), end=len(train_data) + len(test_data)-1)
print(fit_4.summary())

prediction = fit_4.predict(start= 33, end= 43)
#print(fit_4.summary())
print("===prediction===")
#print(prediction01)



print("===Train===")
print(train_data)
print("===Test===")
print(test_data)

print("===prediction===")

prediction = prediction.astype(int)
print(prediction)
forecast4 = fit_4.forecast(forecast_timestep).rename(r'$\alpha=%s$'%fit_4.model.params['smoothing_level'])
plt.figure(figsize=(16,10))
print("====Forecast=======")
print (forecast4)

plt.plot(stock_data, color='black')
#plt.plot(fit_4.fittedvalues, color='blue')
plt.plot(prediction, color='blue')
line4, = plt.plot(forecast4, color='green')
plt.legend([line4], [forecast4.name])
plt.show()

#prediction = fit_4.fittedvalues
#test_data = pd.DataFrame({'Year':stock_data.index, 'Rate':stock_data.values})


prediction = pd.DataFrame({'Year':prediction.index, 'Rate':prediction.values})


#prediction = prediction.to_frame()
prediction['Rate'] = prediction['Rate'].astype(int)
print(prediction)

print("******************************************************************")
print("******************************************************************")
print(" ")
print("EVALUATION")
print(" ")
print("******************************************************************")
print("******************************************************************")

mae = mean_absolute_error(test_data['Rate'], prediction['Rate'])

print(mae)
print ("===========================================================")
mape = mean_absolute_percentage_error(test_data['Rate'], prediction['Rate'])

MSE = mean_squared_error(test_data['Rate'], prediction['Rate'])
rmse = np.sqrt(mean_squared_error(test_data['Rate'], prediction['Rate']))

print ("===========================================================")


print ("The MAE: ", mae)
print ("===========================================================")
print("The MAPE: ", mape)

print ("===========================================================")
print ("The RMSE: ",rmse)

print ("===========================================================")
print ("The MSE: ",MSE)
# Accuracy metrics
import math 
MSE = np.square(np.subtract(test_data['Rate'],prediction['Rate'])).mean()   
   
rsme = math.sqrt(MSE)  
print ("===========================================================")
print("the other RMSE: ", rsme)


#Resources 

#https://medium.com/analytics-vidhya/all-about-it-time-aeries-analysis-exponential-smoothing-example-e62057768bc1
#https://www.statsmodels.org/dev/examples/notebooks/generated/exponential_smoothing.html



#train test split
#https://builtin.com/data-science/train-test-split


#univert data
#https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc44.htm#:~:text=The%20term%20%22univariate%20time%20series,to%20predict%20el%20nino%20effects.