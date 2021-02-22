#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import datetime
from statsmodels.tsa.arima_model import ARIMA
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from pylab import rcParams
from pandas.plotting import autocorrelation_plot
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})
from matplotlib.pyplot import figure
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from numpy import log
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm
from statsmodels.tools.eval_measures import mse, rmse
import seaborn as sns
import warnings
import math
import scipy.stats as stats
import scipy
from sklearn.preprocessing import scale
from statsmodels.tsa.stattools import adfuller
from numpy import log
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf
from pandas.tseries.offsets import DateOffset


# In[2]:


#Reading my dataset and showing it

covid_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
covid_df = pd.read_csv(covid_url)
covid_df.head()


# # Cleaning my dataset

# In[3]:


# Filtering the Country that I will analyse first

fil_covid=covid_df.loc[covid_df['Country/Region'].isin(['Australia'])]
fil_covid


# In[4]:


# Cleaning the data, taking off the colunms that I didn't need

g_covid=fil_covid.drop(columns=['Lat', 'Long'])
g_covid


# In[5]:


# Transpnding some of my colunms in lines and creating two new colunms called "Date" and "Cases"

covid=g_covid.melt(id_vars=['Province/State', 'Country/Region'], var_name='Date', value_name='Cases')
covid.head()


# In[6]:


covid['Date']=pd.to_datetime(covid['Date'])
covid.head()


# In[7]:


# Grouping by Country and Date

covid=covid.groupby(['Country/Region', 'Date']).sum().reset_index()
covid.head()


# In[8]:


covid.describe()
covid.set_index('Date',inplace=True)


# In[9]:


rcParams['figure.figsize'] = 15, 7
covid.plot()


# # Analysing and Testing my dataset

# In[10]:


test_result=adfuller(covid['Cases'])


# In[11]:


#Verifying if my data is stationary or not to drive me for the next steps 

def adfuller_test(cases):
    result = adfuller(cases)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations']
    for value, label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data is stationary")
    else:
        print("weak evidence against null hypothesis,indicating it is non-stationary ")

adfuller_test(covid['Cases'])


# From the above test, I could get from my data the P-value is greater than the significance level 0.05. I could observe that my data is consistent with the null hypothesis that means it is non-stationary

# In[12]:


import numpy as np, pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import pmdarima as pm
from pmdarima.arima.utils import ndiffs
import datetime


# In[13]:


# Verifying the autocorrelation from my dataset

autocorrelation_plot(covid['Cases'])
plt.show()


# The following test I would like to find which is the right order of differencing that is the minimum differencing required to get a near-stationary.

# In[14]:


fig = plt.figure(figsize=(12,8))
covid['Cases First Difference']=covid['Cases'].diff().dropna()
adfuller_test(covid['Cases First Difference'].dropna())
covid['Cases First Difference'].plot()
covid.head()


# In[15]:


covid['Cases Second Difference']=covid['Cases'].diff().diff().dropna()
adfuller_test(covid['Cases Second Difference'].dropna())
covid['Cases Second Difference'].plot()
covid.head()


# In[16]:


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(covid['Cases First Difference'].dropna(),lags=50,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(covid['Cases First Difference'].dropna(),lags=50,ax=ax2)


# In[17]:


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(covid['Cases Second Difference'].dropna(),lags=50,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(covid['Cases Second Difference'].dropna(),lags=50,ax=ax2)


# The above results shows us that the time series reaches stationarity with two orders of differencing. And these graphs helpd me to define my ARIMA model.

# In[18]:


import pmdarima as pm
from pmdarima.arima.utils import ndiffs


# With my terms defined ARIMA (p,d,q) 
# p is the order of the AR term = 1
# q is the order of the MA term = 2
# d is the number of differencing required to make the time series stationary = 2

# In[19]:


y = covid['Cases']

## Adf Test
ndiffs(y, test='adf')  

# KPSS test
ndiffs(y, test='kpss')  

# PP test:
ndiffs(y, test='pp') 


# In[20]:


# Calculating the ARIMA model to analyse the coef, P-value 

model = ARIMA(covid['Cases'], order=(1,2,2))
model_fit = model.fit(disp=0)
print(model_fit.summary())


# From the above information I chose the order 2 for differencing my data

# In[21]:


# Testing the residuals and verifying their behavior

residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()


# The residual errors seem fine with near zero mean and uniform variance

# In[22]:


#Let’s plot the actuals against the fitted values using plot_predict().

model_fit.plot_predict(dynamic=False)
plt.show()


# ## Training and predicting my dataset

# In[23]:


from statsmodels.tsa.stattools import acf

# Create Training and Test with around 60% of my dataset

train = covid.Cases[:200]
test = covid.Cases[200:]


# In[24]:


# Aplaying ARIMA in main training data

model = ARIMA(train, order=(1, 2, 2))  
fitted = model.fit(disp=-1)  


# In[25]:


# Forecast
fc, se, conf = fitted.forecast(196, alpha=0.05)  # 95% conf


# In[26]:


# Make as pandas series to investigate my trainning data

fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)


# In[27]:


# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()


# Withe the Forecast X Actuals plot, I could observe that my data needs more improvements.
# 
# ## The below information will help me to verify which order in my p,q,d values could I use to get more reliable results

# In[28]:


def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    acf1 = acf(fc-test)[1]                      # ACF1
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 'acf1':acf1, 
            'corr':corr, 'minmax':minmax})

forecast_accuracy(fc, test.values)


# In[29]:


model = pm.auto_arima(covid['Cases'], start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)

print(model.summary())


# In[30]:


model.plot_diagnostics(figsize=(7,5))
plt.show()


# I could verify that my ARIMA test the ideal p,d,q values are 2,2,2

# In conclusion, the next plot I could predict the cases for the next months 

# In[31]:


from pandas.tseries.offsets import DateOffset
results=model.fit()
future_dates=[covid.index[-1]+ DateOffset(x) for x in range(1,730)]
future_datest_df=pd.DataFrame(index=future_dates[1:],columns=covid.columns)
future_datest_df.tail()
future_df=pd.concat([covid,future_datest_df])
future_df['forecast'] = results.predict(start=20, end = 700, dynamic= True)
future_df[['Cases', 'forecast']].plot(figsize=(12, 8))


# In[ ]:




