# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 04:30:16 2022

@author: ftuha
"""


"""
Created on Wed Nov 30 12:43:59 2022

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


import statsmodels.api as sm

import warnings 
warnings.filterwarnings('ignore')

#%matplotlib inline

# Load data example
MedNote = pd.read_csv('YearOfDeath01.csv')


print("******************************************************************")
print("******************************************************************")
print("Ploting :")
print("******************************************************************")
print("******************************************************************")


MedNote.head()

# Updating the header
MedNote.columns=["Year of death","Rate of death"]
MedNote.head()
MedNote.describe()
#set the index of the dataframe to this variable using the set_index method
MedNote.set_index('Year of death',inplace=True)



from pylab import rcParams
rcParams['figure.figsize'] = 15, 7
MedNote.plot()

train = MedNote[:80]
test = MedNote[20:]


print ("=== train ===")
print(train)

print("== Test ==")
print(test)
print("******************************************************************")
print("******************************************************************")
print("Checking the stationary of the dataset :")
print("******************************************************************")
print("******************************************************************")
#Checking the stationary of the dataset

#Ho: It is non-stationary
#H1: It is stationary

test_result=adfuller(MedNote["Rate of death"])

def adfuller_test(sales):
    result=adfuller(sales)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )

    if result[1] <= 0.05:
       print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data is stationary")
    else:
       print("weak evidence against null hypothesis,indicating it is non-stationary ")

adfuller_test(train["Rate of death"])
#print(adfuller_test(MedNote["Rate of death"]))


print("******************************************************************")
print("******************************************************************")
print("Let’s try to see the first difference and seasonal difference :")
print("******************************************************************")
print("******************************************************************")



MedNote['Sales First Difference'] = MedNote["Rate of death"] - MedNote["Rate of death"].shift(1)
MedNote['Seasonal First Difference']=MedNote["Rate of death"]-MedNote["Rate of death"].shift(12)


train['Sales First Difference'] = train["Rate of death"] - train["Rate of death"].shift(1)
train['Seasonal First Difference']= train["Rate of death"]- train["Rate of death"].shift(12)

test['Sales First Difference'] = test["Rate of death"] - test["Rate of death"].shift(1)
test['Seasonal First Difference']= test["Rate of death"]- test["Rate of death"].shift(12)

print(MedNote.head())
#MedNote.to_csv("diff.csv")

# Again testing if data is stationary
print("******************************************************************")
print("******************************************************************")
print("Again testing if data is stationary :")
print("******************************************************************")
print("******************************************************************")
adfuller_test(MedNote['Sales First Difference'].dropna())


MedNote['Sales First Difference'].plot()

print("******************************************************************")
print("******************************************************************")
print("Let’s try to see the second difference and seasonal difference :")
print("******************************************************************")
print("******************************************************************")



MedNote['Sales First Difference'] = MedNote["Rate of death"] - MedNote["Rate of death"].shift(1)
MedNote['Seasonal First Difference']=MedNote["Rate of death"]-MedNote["Rate of death"].shift(12)



train['Sales First Difference'] = train["Rate of death"] - train["Rate of death"].shift(1)
train['Seasonal First Difference']= train["Rate of death"]- train["Rate of death"].shift(12)

test['Sales First Difference'] = test["Rate of death"] - test["Rate of death"].shift(1)
test['Seasonal First Difference']= test["Rate of death"]- test["Rate of death"].shift(12)

MedNote['Sales First Difference'] = MedNote["Rate of death"].diff().diff() 
train['Sales First Difference'] = train["Rate of death"].diff().diff() 
test['Sales First Difference'] = test["Rate of death"].diff().diff() 

#MedNote.to_csv("diff.csv")

# Again testing if data is stationary
print("******************************************************************")
print("******************************************************************")
print("Again testing if data is stationary :")
print("******************************************************************")
print("******************************************************************")
adfuller_test(MedNote['Sales First Difference'].dropna())


MedNote['Sales First Difference'].plot()










print("******************************************************************")
print("******************************************************************")
print(" Creaeting auto-correlation  :")
print("******************************************************************")
print("******************************************************************")



pp = autocorrelation_plot(train["Rate of death"])
autocorrelation_plot(train["Rate of death"])
pyplot.show()




#fig = pyplot.figure(figsize=(12,8))
#ax1 = fig.add_subplot(211)
#fig = sm.graphics.tsa.plot_acf(train['Sales First Difference'].dropna(),lags=30,ax=ax1)
#ax2 = fig.add_subplot(212)
#fig = sm.graphics.tsa.plot_pacf(train['Sales First Difference'].dropna(),lags=14,ax=ax2)





#p=1, d=1, q=0 or 1

#‘p’ is the order of the ‘Auto Regressive’ (AR) term. 
#It refers to the number of lags of Y to be used as predictors.

# d =1 because we have done difference only one time to make the time series stationary
#The value of d, therefore, is the minimum number of differencing needed to make the series stationary. 
#And if the time series is already stationary, then d = 0.

print("******************************************************************")
print("******************************************************************")
print(" For non-seasonal data")
print(" p=1, d=1, q=0 or 1")
print("******************************************************************")
print("******************************************************************")





model =sm.tsa.arima.ARIMA(train["Rate of death"],order=(0,2,2))
model_fit = model.fit()
print (model_fit.summary())


print (test.index)

#prediction = model_fit.predict(n_periods = 40)

#prediction0 = pd.DataFrame(model_fit.predict(n_periods = 40), index = test.index)


# Get forecast 500 steps ahead in future
pred_uc = model_fit.get_forecast(steps=10)
# Get confidence intervals of forecasts
pred_ci = pred_uc.conf_int()
print("##############################################################")
print(pred_ci)


prediction = pred_ci ["upper Rate of death"]
prediction = prediction.astype(int)

print("===  prediction ==")
print (prediction[0:])
#MedNote['forecast'] = prediction ["predicted_mean"]
print("******************************************************************")
print("******************************************************************")
print(" ")
print("PREDICTION")
print(" ")
print("******************************************************************")
print("******************************************************************")
#print (prediction)

#MedNote['forecast'] = model_fit.predict(start=28,end=36,dynamic=True, index= None)




#print(MedNote['forecast'])
#MedNote[['Rate of death','forecast']].plot(figsize=(12,8))

#start=90,end=103,dynamic=True







print ("*****************************************************************")
print ("*****************************************************************")

model=sm.tsa.statespace.SARIMAX(train['Rate of death'],order=(2, 2, 1),
                                seasonal_order=(0,0,0,0))
results= model.fit()
#MedNote['forecast']= results.predict(start=2000,end=2017, dynamic=True)

#prediction = pd.DataFrame(arima.predict(n_periods = 20), index=test.index)
#MedNote.set_index('Year of death',inplace=True)
#prediction = pd.DataFrame(model_fit.predict(n_periods = 20), 
#                          index = MedNote.index)
print (results.summary())

prediction01 = pd.DataFrame(results.predict(n_periods = 20), 
                         index = test.index)

print("================================================================")
print("===========================SECOND=====================================")




MedNote['forecast'] = prediction01 ["predicted_mean"]
#start=90,end=103,dynamic=True
MedNote[['Rate of death','forecast']].plot(figsize=(12,8))


print ("*****************************************************************")
print ("*****************************************************************")


#



from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error

#print(test)
#print(prediction)


test.drop('Sales First Difference', inplace=True, axis=1)
test.drop('Seasonal First Difference', inplace=True, axis=1)
train.drop('Sales First Difference', inplace=True, axis=1)
train.drop('Seasonal First Difference', inplace=True, axis=1)
MedNote.drop('Sales First Difference', inplace=True, axis=1)
MedNote.drop('Seasonal First Difference', inplace=True, axis=1)

#prediction.rename(columns = {'predicted_mean':'Rate of death'}, inplace = True)

print("******************************************************************")
print("******************************************************************")
print(" ")
print("EVALUATION")
print(" ")
print("******************************************************************")
print("******************************************************************")
#prediction ["Rate of death"] = prediction ["Rate of death"].astype(int)

#print(prediction)
mae = mean_absolute_error(test[1:], prediction)


print ("===========================================================")
mape = mean_absolute_percentage_error(test[1:], prediction)

#MSE = mean_squared_error(test, prediction)
rmse = np.sqrt(mean_squared_error(test[1:], prediction))

print ("===========================================================")


print ("The MAE: ", mae)
print ("===========================================================")
print("The MAPE: ", mape)

print ("===========================================================")
print ("The RMSE: ",rmse)

print ("===========================================================")
#print ("The MSE: ",MSE)
# Accuracy metrics
import math 
#MSE = np.square(np.subtract(test,prediction)).mean()   
   
#rsme = math.sqrt(MSE)  
print ("===========================================================")
#print("the other RSME: ", rsme)



#from sklearn.metrics import r2_score
#test['death'] = prediction
#r2_score(MedNote['Rate of death'], prediction ["predicted_mean"])



year = 2020
count = 20
for i in range(count):
    
     new_row = {'Year of death':year}
     MedNote = MedNote.append(new_row, ignore_index=True)
     year = year + 1


# Get forecast 500 steps ahead in future
pred_uc = model_fit.get_forecast(steps=14)
# Get confidence intervals of forecasts
pred_ci = pred_uc.conf_int()
print("##############################################################")
print(pred_ci)


MedNote['for'] = pred_ci ["lower Rate of death"]
#start=90,end=103,dynamic=True
MedNote[['Rate of death','for']].plot(figsize=(12,8))
#ax = MedNote.plot(label='observed', figsize=(20, 15))
#pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
#ax.fill_between(pred_ci.index,
#                pred_ci.iloc[:, 0],
#                pred_ci.iloc[:, 1], color='k', alpha=.25)
#ax.set_xlabel('Date')
#ax.set_ylabel('CO2 Levels')

#plt.legend()
#plt.show()
print("##############################################################")
print("##############################################################")
print("##############################################################")





print("##############################################################")
print("##############################################################")
print("##############################################################")

#Another way to find the ARIMA parameters is using an information criteria
#this method is called Akaike Information Criterion (AIC)

import statsmodels.api as sm
result = {}
for p in range(5):
    for q in range(5):
        arma = sm.tsa.ARIMA(train, order=(p,2,q))
        arma_fit = arma.fit()
        result[(p,q)] = arma_fit.aic

p,q = min(result, key=result.get)

print(p)
print(q)




#predictions_f_ms = model_fit.forecast(steps=len(test))
#predictions_p_ms = model_fit.predict(start=len(train), end=len(train)+len(test)-1)

#print(predictions_p_ms)
#https://www.analyticsvidhya.com/blog/2020/10/how-to-create-an-arima-model-for-time-series-forecasting-in-python/

#https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/

#https://www.tensorflow.org/tutorials/structured_data/time_series

#https://towardsdatascience.com/temporal-loops-intro-to-recurrent-neural-networks-for-time-series-forecasting-in-python-b0398963dc1f

#https://www.justintodata.com/arima-models-in-python-time-series-prediction/

#https://www.frontiersin.org/articles/10.3389/fdata.2020.00004/full

#https://ademos.people.uic.edu/Chapter23.html

#https://stackoverflow.com/questions/62783633/how-to-interpret-plots-of-autocorrelation-and-partial-autocorrelation-using-pyth



#https://www.kaggle.com/code/sumi25/understand-arima-and-tune-p-d-q