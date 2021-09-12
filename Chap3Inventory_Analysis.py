import itertools
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import warnings  # do not disturbe mode
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
warnings.filterwarnings('ignore')        # statistics and econometrics
import statsmodels.tsa.api as smt
import statsmodels.api as sm

#----------- Part 1: Import data   ---------------------------------
sales_data= pd.read_excel("D:/Warwick/dissertation/Original_Data/2021-09-06 MASTER DASHBOARD_PreMeet.xlsm",
                          sheet_name='Data',parse_dates=['Received Date'])
#print(sales_data)
sales_data = sales_data[sales_data['Status']=='PAID']
sales_data = sales_data[sales_data['Sub-source']!='Laika Designs']
sales_data = sales_data[sales_data['Sub-source']!='https://happylinencompany.co.uk']
SKU_selected = 'DINOSNORE-HL01-SIN'
ProductRange_selected = 'Dinosnore'
RangeDINOSIN_sales_data=sales_data[sales_data['Product Range2'] == ProductRange_selected]

# define performance metric MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true+0.1))) * 100

#------------Part 2: Sales Forecast (Baseline approach)--------------------------
# Use columns 'Received Date','Main SKU', and 'Quantity' only
simp_RangeDINOSIN_sales_data = RangeDINOSIN_sales_data.loc[:,['Received Date','Main SKU','Quantity']]
# groupby with each order date and each SKU
RangeDINOSIN_sales_qty = simp_RangeDINOSIN_sales_data.groupby(['Received Date','Main SKU'],as_index=False).sum()

# same as above for sales revenue
simp_RangeDINOSIN_sales_rev = RangeDINOSIN_sales_data.loc[:,['Received Date','Main SKU','Transaction GBP']]
RangeDINOSIN_sales_rev = simp_RangeDINOSIN_sales_rev.groupby(['Received Date','Main SKU'],as_index=False).sum()
RangeDINOSIN_sales_rev = RangeDINOSIN_sales_rev.set_index('Received Date')

# Convert dataframe to order date as index, SKU as columns, and values are sales quantities
conv_RangeDINOSIN_sales_qty = pd.pivot(RangeDINOSIN_sales_qty, index='Received Date', columns='Main SKU',values='Quantity')
conv_RangeDINOSIN_sales_qty = conv_RangeDINOSIN_sales_qty.fillna(0)  #fill Nan with zeros

# Convert weekly data to monthly data
monthly_RangeDINOSIN_sales_qty = conv_RangeDINOSIN_sales_qty.resample('M').sum()
# monthly sales qty in 2019, 2020, and 2021
monthly_2019_salesqty = monthly_RangeDINOSIN_sales_qty['2019']
monthly_2020_salesqty = monthly_RangeDINOSIN_sales_qty['2020']
monthly_2021_salesqty = monthly_RangeDINOSIN_sales_qty['2021']

# Baseline value
baseline_qty_2019 = monthly_2019_salesqty['2019-01-31':'2019-08-31'].mean()
baseline_qty_2020 = monthly_2020_salesqty['2020-01-31':'2020-08-31'].mean()
# growth rate
revenue_2019 = RangeDINOSIN_sales_rev['2019']['Transaction GBP'].sum()
revenue_2020 = RangeDINOSIN_sales_rev['2020']['Transaction GBP'].sum()
growth_rate_1920 = revenue_2020/revenue_2019
# difference between baseline value and actual value
seasonality_19 = monthly_2019_salesqty-baseline_qty_2019
seasonality_20 = monthly_2020_salesqty-baseline_qty_2020
# prediction for 2021 based on 2020
pred_2021 = seasonality_20[SKU_selected]\
            +baseline_qty_2020[SKU_selected]*growth_rate_1920
pred_2021.index = pd.date_range(end ='2021-12',freq='M',periods=12)
# prediction for 2020 based on 2019
pred_2020 = seasonality_19[SKU_selected]\
            +baseline_qty_2019[SKU_selected]*growth_rate_1920
pred_2020.index = pd.date_range(end ='2020-12',freq='M',periods=12)
monthly_2021_salesqty.index = pred_2021.index[:8]  #8 means current month
monthly_2020_salesqty.index = pred_2020.index

mape_baseline_20 = mean_absolute_percentage_error(monthly_2020_salesqty.iloc[:,1], pred_2020)
mape_baseline_21 = mean_absolute_percentage_error(monthly_2021_salesqty.iloc[:,1], pred_2021[:8])  # 8 means current month


plt.figure(figsize=(15, 7))
plt.title("Baseline approximation  20-21,Mean Absolute Percentage Error: {0:.2f}%".format(mape_baseline_21),fontsize=20)
plt.plot(pred_2021, color='r', label="baseline_forecast")
plt.axvspan(xmin='2021-07', xmax='2021-12', alpha=0.5, color='lightgrey')
plt.plot(monthly_2021_salesqty[SKU_selected], label="actual quantity")
plt.legend(fontsize=20)
plt.grid(True)

plt.figure(figsize=(15, 7))
plt.title("Baseline approximation 19-20,Mean Absolute Percentage Error: {0:.2f}%".format(mape_baseline_20),fontsize=20)
plt.plot(pred_2020, color='r', label="baseline_forecast")
#plt.axvspan(xmin='Jun', xmax='Dec', alpha=0.5, color='lightgrey')
plt.plot(monthly_2020_salesqty[SKU_selected], label="actual quantity")
plt.legend(fontsize=20)
plt.grid(True)

#-------------------Part 3: Run rate  Approximation   ----------------------------------

def time_slice(time,single,X_lag):
    # This funtion is used to get different time slices
    sample = []
    label = []
    for k in range(len(time) - X_lag - 1):
        t = k + X_lag
        sample.append(single[k:t])
        label.append(single[t])
    return sample,label

# For monthly data, use the past 6 months sales quantity to predict next month's sales quantity
sample,label = time_slice(monthly_RangeDINOSIN_sales_qty.index,monthly_RangeDINOSIN_sales_qty.iloc[:,1],6)
# For weekly data, use the past 26 weeks sales quantity to predict next week's sales quantity
sample2,label2 = time_slice(conv_RangeDINOSIN_sales_qty.index,conv_RangeDINOSIN_sales_qty.iloc[:,1],26)

sample=np.array(sample)
sample2=np.array(sample2)
# prediction without adjusting seasonality
qty_ave_pred = sample[12:].mean(axis=1).round()
# adjusting seasonality with adding the difference
qty_ave_pred2 = sample[12:].mean(axis=1).round()+ label[:(len(sample)-12)]-sample[:(len(sample)-12)].mean(axis=1).round()

# Same method for weekly data
qty_ave_pred_weekly = sample2[52:].mean(axis=1).round()
qty_ave_pred2_weekly = sample2[52:].mean(axis=1).round()+ label2[:(len(sample2)-52)]-sample2[:(len(sample2)-52)].mean(axis=1).round()
# Check MAPE value
mape_ave = mean_absolute_percentage_error(label[12:], qty_ave_pred)
mape_ave2 = mean_absolute_percentage_error(label[12:], qty_ave_pred2)
mape_ave_weekly = mean_absolute_percentage_error(label2[52:], qty_ave_pred_weekly)
mape_ave2_weekly = mean_absolute_percentage_error(label2[52:], qty_ave_pred2_weekly)

date_interval = pd.date_range(start= None, end='2021-06',freq='m',periods=len(qty_ave_pred2))
date_interval2 = pd.date_range(start= None, end='2021-06',freq='7D',periods=len(qty_ave_pred2_weekly))
plt.figure(figsize=(15, 7))
plt.title("run rate without adjusting seasonality_Mean Absolute Percentage Error: {0:.2f}%".format(mape_ave),fontsize = 20)
plt.plot(date_interval,qty_ave_pred, color='r', label="rolling_mean forecast")
#plt.axvspan(xmin=25, xmax=35, alpha=0.5, color='lightgrey')
plt.plot(date_interval,label[12:], label="actual")
plt.legend(fontsize = 20)
plt.grid(True)

plt.figure(figsize=(15, 7))
plt.title("run rate without adjusting seasonality(Weekly data)_Mean Absolute Percentage Error: {0:.2f}%".format(mape_ave),fontsize = 20)
plt.plot(date_interval2,qty_ave_pred_weekly, color='r', label="rolling_mean forecast")
#plt.axvspan(xmin=25, xmax=35, alpha=0.5, color='lightgrey')
plt.plot(date_interval2,label2[52:], label="actual")
plt.legend(fontsize = 20)
plt.grid(True)

fig, ax = plt.subplots(1,1,figsize=(15, 7))
plt.title("run rate adjusting seasonality_Mean Absolute Percentage Error: {0:.2f}%".format(mape_ave2),fontsize = 20)
ax.plot(date_interval,qty_ave_pred2, color='r', label="run_rate forecast")
#plt.axvspan(xmin=25, xmax=35, alpha=0.5, color='lightgrey')
ax.plot(date_interval,label[12:], label="actual")
plt.legend(fontsize = 20)
plt.grid(True)

fig, ax = plt.subplots(1,1,figsize=(15, 7))
plt.title("run rate adjusting seasonality(Weekly data)_Mean Absolute Percentage Error: {0:.2f}%".format(mape_ave2),fontsize = 20)
ax.plot(date_interval2,qty_ave_pred2_weekly, color='r', label="run_rate forecast")
#plt.axvspan(xmin=25, xmax=35, alpha=0.5, color='lightgrey')
ax.plot(date_interval2,label2[52:], label="actual")
plt.legend(fontsize = 20)
plt.grid(True)

# Seasonal decomposition
decompose_result= seasonal_decompose(conv_RangeDINOSIN_sales_qty.iloc[:,1],
                                      model="additive",two_sided=False)
trend = decompose_result .trend
seasonal = decompose_result.seasonal
residual = decompose_result .resid
decompose_result.plot();
# for weekly data
decompose_result2= seasonal_decompose(conv_RangeDINOSIN_sales_qty.iloc[:,1],
                                      model="additive",two_sided=False)

# sales forecast based on run rate method
previous_difference = monthly_RangeDINOSIN_sales_qty.iloc[-12]-monthly_RangeDINOSIN_sales_qty.iloc[-18:-12].mean().round()
current_qty_ave_pred1 = monthly_RangeDINOSIN_sales_qty.tail(6).mean(axis=0).round()+ previous_difference
# weekly
previous_difference_weekly = conv_RangeDINOSIN_sales_qty.iloc[-52]-conv_RangeDINOSIN_sales_qty.iloc[-78:-52].mean().round()
current_qty_ave_pred1_weekly = conv_RangeDINOSIN_sales_qty.tail(26).mean(axis=0).round()+ previous_difference_weekly

#---------------------------------------------------------------
#                 Part 4: Sales Forecast (ARIMA)
#---------------------------------------------------------------
def tsplot(y, lags=None, figsize=(12, 7), style='bmh'):
    """
        Plot time series, its ACF and PACF, calculate Dickey–Fuller test
        y - timeseries
        lags - how many lags to include in ACF, PACF calculation
    """

    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))

        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()
tsplot(conv_RangeDINOSIN_sales_qty.iloc[:,1],lags=60)
#Dickey-Fuller test
def test_stationarity(timeseries):

    plt.figure(figsize=(12, 8))
    plt.plot(timeseries, color='blue')
    plt.show()
    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test statistic', 'p-value', '#Lags Used', 'Number of Obervations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value(%s)' % key] = value
    print(dfoutput)

test_stationarity(conv_RangeDINOSIN_sales_qty.iloc[:,1])
#p-value > 0.05 - This implies that time-series is non-stationary.
#p-value <=0.05 - This implies that time-series is stationary-our sample is stationary

#Remove Seasonality
#with first difference
first_difference = conv_RangeDINOSIN_sales_qty.iloc[:,1] - conv_RangeDINOSIN_sales_qty.iloc[:,1].shift(1)    #也可以使用diff()
first_difference.dropna(inplace=True)
# test stationary
test_stationarity(first_difference)
tsplot(first_difference, lags=60)
# check seasonal decomposition
decompose_result_first_difference = seasonal_decompose(first_difference,model="additive",two_sided=False)
decompose_result_first_difference .plot();

#second difference
second_difference = first_difference-first_difference.shift(1)
second_difference.dropna(inplace=True)
test_stationarity(second_difference)
tsplot(second_difference, lags=60)

#seasonal difference
seasonal_difference = conv_RangeDINOSIN_sales_qty.iloc[:,1] - conv_RangeDINOSIN_sales_qty.iloc[:,1].shift(52)
seasonal_difference.dropna(inplace=True)
test_stationarity(seasonal_difference)
tsplot(seasonal_difference, lags=40)

#seasonal first difference
seasonal_first_difference = first_difference - first_difference.shift(52)
seasonal_first_difference.dropna(inplace=True)
test_stationarity(seasonal_first_difference)
tsplot(seasonal_first_difference, lags=40)

# Create Training and Test
monthly_index = index = int(monthly_RangeDINOSIN_sales_qty.shape[0]*0.8)
monthly_train = monthly_RangeDINOSIN_sales_qty.iloc[:,1][0:index]   # 80% for training
monthly_test = monthly_RangeDINOSIN_sales_qty.iloc[:,1][index:]     # 20% for test

# Fitting ARMA model
arma_mod = ARIMA(seasonal_first_difference, order=(1, 0, 1)).fit()
print(arma_mod.summary())
arma_mod2 = ARIMA(conv_RangeDINOSIN_sales_qty.iloc[:,1], order=(0, 1, 0)).fit()
print(arma_mod2.summary())


#Monthly Data
# Fitting SARIMA model
arima_model1=sm.tsa.statespace.SARIMAX(monthly_train,trend='n',order=(0,1,1),seasonal_order=(0,1,1,12))
arima_model1_fitted=arima_model1.fit()
print(arima_model1_fitted.summary())

# Plot residual errors
#Residual diagnostic
residuals = pd.DataFrame(arima_model1_fitted.resid)
fig1, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()

# SARIMA Forecast
arima_forecast1 = arima_model1_fitted.forecast(len(monthly_test)+3, alpha=0.05)
mape_arima1 = mean_absolute_percentage_error(arima_forecast1[:len(monthly_test)],monthly_test )
# Plot
# sales data for June 2021 is not complete, so the performance can be better with complete June data
plt.figure(figsize=(18, 6))
plt.plot(monthly_train ,label='original training dataset')
plt.plot(arima_model1_fitted.fittedvalues,color = 'orange',label = 'fitted value for the past')
plt.plot(monthly_test, color='red',label='test set')
plt.plot(arima_forecast1, color='darkgreen', label='forecast')
plt.title("ARIMA monthly Forecast of qty, Mean Absolute Percentage Error: {0:.2f}%".format(mape_arima1),fontsize =10)
plt.legend(fontsize = 10)
plt.show()
# Prediction for the next month
current_arima_pred = arima_forecast1[-3]

#Weekly data
weekly_index = index = int(conv_RangeDINOSIN_sales_qty.shape[0]*0.8)
weekly_train = conv_RangeDINOSIN_sales_qty.iloc[:,1][0:index]
weekly_test = conv_RangeDINOSIN_sales_qty.iloc[:,1][index:]

tsplot(seasonal_first_difference, lags=40)
arima_model2=sm.tsa.statespace.SARIMAX(weekly_train,trend='n',order=(0,1,1),seasonal_order=(0,1,1,52))
arima_model2_fitted=arima_model2.fit()
print(arima_model2_fitted.summary())

# Plot residual errors
#Residual diagnostic
residuals2 = pd.DataFrame(arima_model2_fitted.resid)
fig2, ax = plt.subplots(1,2)
residuals2.plot(title="Residuals", ax=ax[0])
residuals2.plot(kind='kde', title='Density', ax=ax[1])
plt.show()

# SARIMA Forecast
arima_forecast2 = arima_model2_fitted.forecast(len(weekly_test)+12, alpha=0.05)
mape_arima2 = mean_absolute_percentage_error(arima_forecast2[:len(weekly_test)], weekly_test)
# Plot
plt.figure(figsize=(18, 6))
plt.plot(weekly_train,label='original training dataset')
plt.plot(arima_model2_fitted.fittedvalues,color = 'orange',label = 'fitted value for the past')
plt.plot(weekly_test,color ='red',label='test set')
plt.plot(arima_forecast2, color='darkgreen', label='forecast')
plt.title("ARIMA weekly Forecast of qty, Mean Absolute Percentage Error: {0:.2f}%".format(mape_arima2),fontsize = 20)
plt.legend(fontsize = 10)
plt.show()
arima_forecast2[-12]  ## Prediction for the next week

arima_model3=sm.tsa.statespace.SARIMAX(conv_RangeDINOSIN_sales_qty.iloc[:,1],trend='n',order=(0,1,1),seasonal_order=(0,1,1,52))
arima_model3_fitted=arima_model3.fit()
print(arima_model3_fitted.summary())

# Grid research to confirm parameters of SARIMA
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
pdq_x_PDQs = [(x[0], x[1], x[2], 52) for x in list(itertools.product(p, d, q))]
a=[]
b=[]
c=[]
wf=pd.DataFrame()
for param in pdq:
    for seasonal_param in pdq_x_PDQs:
        try:
            mod = sm.tsa.statespace.SARIMAX(weekly_train,order=param,seasonal_order=seasonal_param,enforce_stationarity=False,enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{} - AIC:{}'.format(param, seasonal_param, results.aic))
            a.append(param)
            b.append(seasonal_param)
            c.append(results.aic)
        except:
            continue
wf['pdq']=a
wf['pdq_x_PDQs']=b
wf['aic']=c
print(wf[wf['aic']==wf['aic'].min()])

#-----------------------------------------------------------
#                  Part 5:Sales Forecast (SVR)
#-----------------------------------------------------------
# Seasonal decomposition of weekly data
decompose_result_weekly = seasonal_decompose(conv_RangeDINOSIN_sales_qty.iloc[:,1],
                                      model="additive",two_sided=False)
trend_weekly = decompose_result_weekly .trend
seasonal_weekly = decompose_result_weekly .seasonal
residual_weekly = decompose_result_weekly .resid
decompose_result_weekly .plot();

# sales forecast of only one SKU.monthly data will be too small dataset, so use weekly dataset to try.
#same as above run rate, using previous 26 weeks data to predict the next time step
sample_weekly,label_weekly = time_slice(conv_RangeDINOSIN_sales_qty.index,conv_RangeDINOSIN_sales_qty.iloc[:,1],26)
sample_weekly=np.array(sample_weekly)
X_train, X_test, y_train, y_test = train_test_split(sample_weekly, label_weekly, test_size=0.2, random_state=42)

pipeline = Pipeline([('sel', SelectKBest()),  #SelectKBest():feature_selection
                     ('clf', SVR() )])
parameter_space = [{ 'sel__k': [1, 6],
    'clf__C': [0.1, 1, 10, 100],
    'clf__kernel': ['linear','rbf', 'poly']
}]
# grid research for parameter selection
gridsearch_SVR = GridSearchCV(pipeline,
    param_grid = parameter_space,
    scoring='neg_mean_squared_error',
    verbose=1,
    n_jobs=-1)
# 'neg_mean_squared_error' greater is better

gridsearch_SVR.fit(X_train,y_train)
print(gridsearch_SVR.best_estimator_)
print(gridsearch_SVR.best_score_)
# SVR forecasts
qty_svr_pred2 = gridsearch_SVR.predict(X_test).round()
# SVR fitted value
qty_svr_pred1 = gridsearch_SVR.predict(X_train).round()
# MAPE value
mape_svr = mean_absolute_percentage_error(y_test, qty_svr_pred2)

#traing set and test set are not in initial order
plt.figure(figsize=(15, 7))
plt.title('SVR Fitted',fontsize=20)
plt.plot(y_train,color='blue', label="training set")
plt.plot(qty_svr_pred1,color = 'orange', label="fitted value")
plt.legend(fontsize=20)
plt.grid(True)

plt.figure(figsize=(15, 7))
plt.title("SVR_forecast, Mean Absolute Percentage Error: {0:.2f}%".format(mape_svr),fontsize=20)
plt.plot(qty_svr_pred2, color='darkgreen', label="svr_forecast")
plt.plot(y_test,color = 'red', label="test set")
plt.legend(fontsize=20)
plt.grid(True)
# sales forecast for the following week
current_qty_svr_pred = gridsearch_SVR.predict(conv_RangeDINOSIN_sales_qty.iloc[:,1].tail(26).to_numpy().reshape(1,-1))


#-----------------------   Part 6: Inventory Analysis   ----------------------
#Import inventory data
Inventory_Interspan= pd.read_excel("D:/Warwick/dissertation/Original_Data/2021-09-06 MASTER DASHBOARD_PreMeet.xlsm",
                                   sheet_name='InventoryInter',skiprows=1,header=0)
Inventory_FBA= pd.read_excel("D:/Warwick/dissertation/Original_Data/2021-09-06 MASTER DASHBOARD_PreMeet.xlsm",
                             sheet_name='InventoryFBA1',skiprows=1,header=0)
SKUs = pd.read_excel("D:/Warwick/dissertation/Original_Data/2021-09-06 MASTER DASHBOARD_PreMeet.xlsm",
                     sheet_name='SKUs', skiprows=5, header=0)

Inventory_Interspan.set_index(["SKU"], inplace=True);
Inventory_Interspan = Inventory_Interspan.replace(np.nan, 0)
Inventory_FBA.set_index(["SKU"], inplace=True);Inventory_FBA = Inventory_FBA.replace(np.nan, 0)
Inventory_total = Inventory_Interspan+ Inventory_FBA


def run_rate(df,new_value):
    #generate run rate forecasts

    df2 = df.append(new_value,ignore_index=True)
    updated_value = df2.tail(6).mean(axis=0).round()
    previous_diff = df2.iloc[-12]-df2.iloc[-18:-12].mean().round()
    new_prediction = updated_value+previous_diff
    return new_prediction
# sales forecast for the second month based
current_qty_ave_pred2 = run_rate(monthly_RangeDINOSIN_sales_qty,current_qty_ave_pred1)
# sales forecast for the third month based
current_qty_ave_pred3 = run_rate(monthly_RangeDINOSIN_sales_qty.append(current_qty_ave_pred1, ignore_index=True),current_qty_ave_pred2)
# sales forecast for the following quarter
current_qty_quarter = current_qty_ave_pred1+current_qty_ave_pred2+current_qty_ave_pred3

# Inventory analysis based on single month's prediction
def qty_permonth(df_inven,qty_pred):
    qty_pred[qty_pred < 0] = 0
    #combine predicted qty and inventory lecel
    qty_inventory = pd.concat([qty_pred,df_inven], axis=1, join='inner', ignore_index=False)
    qty_inventory.rename(columns={0:'monthly sales qty','Level':'Inventory level'},inplace=True)
    qty_inventory = qty_inventory.fillna(0)
    #adjusting other inventory levels
    qty_inventory['Inventory level'] = [100,1000,300,90,100,0,167,60,68]  #should be actual inventory level

    #Compare three months' qty with inventory level,
    # If current stock months remain < 3 months, then need to place order
    months_remain = (qty_inventory['Inventory level'] // qty_inventory['monthly sales qty']).round()
    qty_inventory.loc[:, 'months_remain'] = months_remain
    qty_inventory.loc[:, 'months remaining flag'] = 3
    qty_inventory['place_order'] = np.where((qty_inventory['months_remain'] - qty_inventory['months remaining flag']) <= 0,'Y', 'N');

    # When the place order flag is 'Y', initially work with monthly predicted sales qty *3 months
    replenish_stock_Y = qty_inventory['monthly sales qty'] * 3
    qty_inventory['replenish_original'] = np.where(qty_inventory['place_order'] == 'Y', replenish_stock_Y, 0)
    return qty_inventory

# Inventory analysis based on three month's prediction
def qty_3month(df_inven,qty_pred):
    qty_pred[qty_pred < 0] = 0
    #combine predicted qty and inventory lecel
    qty_inventory = pd.concat([qty_pred,df_inven], axis=1, join='inner', ignore_index=False)
    qty_inventory.rename(columns={0:'3 months sales qty','Level':'Inventory level'},inplace=True)
    qty_inventory = qty_inventory.fillna(0)
    #adjusting other inventory levels
    qty_inventory['Inventory level'] = [100,1000,400,190,100,0,167,60,68]  #should be actual inventory level

    #Compare three months' qty with inventory level,
    # If current stock months remain < 3 months, then need to place order
    qty_inventory['place_order'] = np.where((qty_inventory['3 months sales qty'] - qty_inventory['Inventory level']) < 0,'N', 'Y');
    # When the place order flag is 'Y', initially work with 3 months sales qty
    replenish_stock_Y = qty_inventory['3 months sales qty']
    qty_inventory['replenish_original'] = np.where(qty_inventory['place_order'] == 'Y', replenish_stock_Y, 0)

    return qty_inventory


RangeDINOSIN_monthly_qty_inventory = qty_permonth(Inventory_total,current_qty_ave_pred1)
RangeDINOSIN_monthly_qty_inventory2 = qty_3month(Inventory_total,current_qty_quarter)

def inventory_analysis(df):
    # Check if the initial quantity reach MOQ 1500pcs for each range
    df['reach_MOQ'] = np.where(df['replenish_original'].sum() < 1500, 'N', 'Y')

    # use half year sales qty to estimate the weights of each SKU
    halfyear_sales_qty = (monthly_RangeDINOSIN_sales_qty.tail(6).sum(axis=0)).sum()
    df['eachSKU_Weight'] = (monthly_RangeDINOSIN_sales_qty.tail(6).sum(axis=0) / halfyear_sales_qty).round(2)
    df['orderSKU_Weight'] = np.where(df['place_order'] == 'Y',df['eachSKU_Weight'].round(2), 0)
    # the updated weights for to be ordered SKU only
    df['finalSKU_Weight'] =np.where(df['place_order'] == 'Y',(df['orderSKU_Weight'] /df['orderSKU_Weight'].sum()).round(2), 0)

    # import packaging information into the chart
    RangeDINOSIN_SKUs = SKUs[SKUs['Product Range'] == ProductRange_selected]
    RangeDINOSIN_SKUs = RangeDINOSIN_SKUs.set_index('Main SKU')
    df['package'] = pd.Series(RangeDINOSIN_SKUs['Quantity per Box'])

    # If the initial quantity can reach MOQ,
    # the final order quantity will be equal to initial quantity plus some quantities to fullfill a Box
    # If the initial quantity can't reach MOQ,
    # the final order quantity will be equal to initial quantity multiply the final weights for that SKU, plus some quantities to fullfill a Box

    fufill_box_reachMOQ = df['package'] - df['replenish_original'] % df['package']

    qtyto_reach_MOQ = df['replenish_original'] + ((1500 - df['replenish_original'].sum())*df['finalSKU_Weight']).round()
    fufill_box_not_reachMOQ = df['package'] - qtyto_reach_MOQ %  df['package']

    df['replenish_final'] =np.where(df['place_order'] == 'N', 0,
                 (np.where(df['reach_MOQ'] == 'Y',df['replenish_original'] + fufill_box_reachMOQ,
                           qtyto_reach_MOQ + fufill_box_not_reachMOQ)))

    return df

Order_suggestion = inventory_analysis(RangeDINOSIN_monthly_qty_inventory)
Order_suggestion2 = inventory_analysis(RangeDINOSIN_monthly_qty_inventory2)
Order_suggestion.to_csv("Inventory Analysis.csv")
Order_suggestion2.to_csv("Inventory Analysis1.csv")







