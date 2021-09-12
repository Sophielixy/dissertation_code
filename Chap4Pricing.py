import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson
import scipy.stats as stats
import matplotlib.ticker as ticker
import statsmodels.api as sm
import seaborn as sns
from scipy.optimize import linprog
from sklearn.neighbors import KernelDensity

#----------------- Part 1:  Import sales data   ---------------------------------------------
sales_data= pd.read_excel("D:/Warwick/dissertation/Original_Data/2021-09-06 MASTER DASHBOARD_PreMeet.xlsm", sheet_name='Data')
#print(sales_data)
sales_data = sales_data[sales_data['Status']=='PAID']
sales_data = sales_data[sales_data['Country']=='United Kingdom']
sales_data = sales_data[sales_data['Sub-source']!='Laika Designs']
sales_data = sales_data[sales_data['Sub-source']!='https://happylinencompany.co.uk']
SKU_selected = 'DINOSNORE-HL01-SIN'
ProductRange_selected = 'Dinosnore'
DINOSIN_sales_data=sales_data[sales_data['Main SKU'] == SKU_selected]
current_date = sales_data.loc[sales_data.index[-1]]['Received Date']
start_date = sales_data.loc[sales_data.index[0]]['Received Date']

simp_RangeDINOSIN_sales_data = DINOSIN_sales_data.loc[:,['Received Date','Main SKU','Quantity']]
DINOSIN_sales_qty = simp_RangeDINOSIN_sales_data.groupby(DINOSIN_sales_data['Received Date']).sum()
DINOSIN_sales_dates = DINOSIN_sales_qty.index

DINOSIN_salesdata_Amazon,DINOSIN_salesdata_AmazonFBA,DINOSIN_salesdata_Direct,DINOSIN_salesdata_Ebay,DINOSIN_salesdata_Etsy,DINOSIN_salesdata_Shopify\
    =  DINOSIN_sales_data.groupby(DINOSIN_sales_data['Source'])
fulldates = pd.date_range(start=start_date, end=current_date,freq='7D')

#---------------------- Part 2: Import price data  --------------------------------------
prices = pd.read_excel("D:/Warwick/dissertation/Original_Data/2021-09-06 MASTER DASHBOARD_PreMeet.xlsm", sheet_name='Price Data',skiprows=1)
DINOSIN_prices=prices[prices['SKU'] == SKU_selected]
# ------Shopify----------
DINOSIN_prices_shopify = DINOSIN_prices[DINOSIN_prices['Channel'] == 'Shopify']
DINOSIN_prices_shopify.set_index(["Date"], inplace=True)
fulldates_prices_trend_shopify = DINOSIN_prices_shopify.reindex(fulldates, fill_value=0)['Price']

# -------Amazon----------
DINOSIN_prices_Amazon = DINOSIN_prices[DINOSIN_prices['Channel'] == 'Amazon']
DINOSIN_prices_Amazon.set_index(["Date"], inplace=True)
fulldates_prices_trend_Amazon = DINOSIN_prices_Amazon.reindex(fulldates, fill_value=0)['Price']

# -------eBay----------
DINOSIN_prices_eBay = DINOSIN_prices[DINOSIN_prices['Channel'] == 'eBay']
DINOSIN_prices_eBay.set_index(["Date"], inplace=True)
fulldates_prices_trend_eBay = DINOSIN_prices_eBay.reindex(fulldates, fill_value=0)['Price']

#------------------------ Part 3: sales quantity from different channels -----------------------------
DINOSIN_salesqty_Amazon = DINOSIN_salesdata_Amazon[1]['Quantity'].groupby(DINOSIN_salesdata_Amazon[1]['Received Date']).sum()
DINOSIN_salesqty_AmazonFBA = DINOSIN_salesdata_AmazonFBA[1]['Quantity'].groupby(DINOSIN_salesdata_AmazonFBA[1]['Received Date']).sum()
DINOSIN_salesqty_Direct = DINOSIN_salesdata_Direct[1]['Quantity'].groupby(DINOSIN_salesdata_Direct[1]['Received Date']).sum()
DINOSIN_salesqty_Ebay = DINOSIN_salesdata_Ebay[1]['Quantity'].groupby(DINOSIN_salesdata_Ebay[1]['Received Date']).sum()
DINOSIN_salesqty_Etsy = DINOSIN_salesdata_Etsy[1]['Quantity'].groupby(DINOSIN_salesdata_Etsy[1]['Received Date']).sum()
DINOSIN_salesqty_Shopify = DINOSIN_salesdata_Shopify[1]['Quantity'].groupby(DINOSIN_salesdata_Shopify[1]['Received Date']).sum()

fulldates_DINOSIN_salesqty_Amazon = DINOSIN_salesqty_Amazon.reindex(fulldates, fill_value=0)
fulldates_DINOSIN_salesqty_AmazonFBA = DINOSIN_salesqty_AmazonFBA.reindex(fulldates, fill_value=0)
fulldates_DINOSIN_salesqty_Ebay = DINOSIN_salesqty_Ebay.reindex(fulldates, fill_value=0)
fulldates_DINOSIN_salesqty_Shopify = DINOSIN_salesqty_Shopify.reindex(fulldates, fill_value=0)


# ------------------------  Part 4: Marginal Cost (MC) Curve of AmazonFBA----------------------------------------
fulldates_unitcost_AmazonFBA = 6.42+fulldates_prices_trend_Amazon*(0.163+0.1667)
MC_df = pd.concat([fulldates_DINOSIN_salesqty_AmazonFBA,fulldates_unitcost_AmazonFBA], axis=1, join='inner', ignore_index=False)
MC_df.rename(columns={'Price':'unit_cost'},inplace=True)
# remove lines with price=0
MC_df = MC_df[MC_df['unit_cost']!=6.42]
MC_df['total_cost'] = MC_df['Quantity']*MC_df['unit_cost']
# except the last row
MC_df1= MC_df.iloc[:MC_df.shape[0]-1,:]
# except the first row
MC_df2= MC_df.iloc[1:MC_df.shape[0],:]
MC_df2.reset_index(inplace=True)
MC_df1.reset_index(inplace=True)
MC_df2['Marginal_cost'] = (MC_df2['total_cost']- MC_df1['total_cost'])/(MC_df2['Quantity']- MC_df1['Quantity'])

#------------------------------Part 5: Profit and revenue plot-----------------------------------------------
#DINOSIN_sales_rev_Amazon = DINOSIN_sales_data[DINOSIN_sales_data['Source'].str.contains("AMAZON")]
DINOSIN_sales_data_AmazonFBA = DINOSIN_sales_data[DINOSIN_sales_data['Source']=='AMAZON FBA']
DINOSIN_sales_rev_AmazonFBA = DINOSIN_sales_data_AmazonFBA['Transaction GBP'].groupby(DINOSIN_sales_data['Received Date']).sum()
fulldates_DINOSIN_sales_rev_AmazonFBA = DINOSIN_sales_rev_AmazonFBA.reindex(fulldates, fill_value=0)

def profit_plot(cost,rev,price):

    #combine cost dataframe and revenue dataframe
    Profit_df = pd.concat([cost,rev],axis=1,join='inner',ignore_index=False)
    Profit_df.rename(columns = {'Transaction GBP':'Revenue'},inplace=True)
    #add column profit
    Profit_df['Profit'] = Profit_df['Revenue'] -Profit_df['total_cost']
    # add column price
    Profit_df = pd.concat([Profit_df,price],axis=1,join='inner',ignore_index=False)

    tick_spacing = 10
    fig2, ax3 = plt.subplots(1,1)
    ax3.stackplot(Profit_df.index,Profit_df['Profit'],Profit_df['Revenue'],
                  labels=['Profit','Revenue'])
    ax3.set_title('Profit and Revenue',fontsize=10)
    ax3.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax3.xaxis.set_tick_params(rotation=45,labelsize = 8)
    #price trend
    ax4 = ax3.twinx()
    ax4.plot(Profit_df.index,Profit_df['Price'],'orange')
    ax4.set_ylabel('price')
    plt.legend()
    plt.show()
    return Profit_df

Profit_Amazon = profit_plot(MC_df,fulldates_DINOSIN_sales_rev_AmazonFBA,fulldates_prices_trend_Amazon)
Profit_Amazon.to_csv("Profit_Amazon.csv")

#----------------------   Part 6:  Demand curve and elasticity -------------------------------------
def Demand_Curve(qty,price,channels):

    df = pd.concat([qty,price], axis=1, join='inner', ignore_index=False)
    df = df[df['Price']!=0]
    df2 = df.groupby('Price').sum()/df.groupby('Price').count()
    df3 = df2.iloc[1:,:]

    fig,ax1 = plt.subplots(1,1)
    ax1.plot(df2['Quantity'].values,df2.index.values,'b')
    ax1.set_xlabel('Quantity')
    ax1.set_ylabel('Price')
    plt.title(channels)
    plt.show()
    return df3

#price only explains 7.7% of the variance of qty,not for predict,，low R2 can be accpetable, p value is more important

fulldates_DINOSIN_salesqty_Amazonall = fulldates_DINOSIN_salesqty_Amazon +fulldates_DINOSIN_salesqty_AmazonFBA
#monthly_DINOSIN_sales_qty = fulldates_DINOSIN_salesqty_Amazonall.resample('M').sum()
demand_curve_Amazon = Demand_Curve(fulldates_DINOSIN_salesqty_Amazonall,fulldates_prices_trend_Amazon,'Amazon')
demand_curve_Shopify = Demand_Curve(fulldates_DINOSIN_salesqty_Shopify,fulldates_prices_trend_shopify,'Shopify')
demand_curve_Ebay = Demand_Curve(fulldates_DINOSIN_salesqty_Ebay,fulldates_prices_trend_eBay,'Ebay')

#linear fit with all channels combined
###weight of each channel
sales_qty_allchannels = sum(np.array(DINOSIN_sales_qty))
weight_Amazon = sum(DINOSIN_salesqty_Amazon)/sales_qty_allchannels
weight_AmazonFBA = sum(DINOSIN_salesqty_AmazonFBA)/sales_qty_allchannels
weight_Shopify = sum(DINOSIN_salesqty_Shopify)/sales_qty_allchannels
weight_Ebay = sum(DINOSIN_salesqty_Ebay)/sales_qty_allchannels
weight_Etsy = sum(DINOSIN_salesqty_Etsy)/sales_qty_allchannels

fulldates_DINOSIN_salesqty_All = DINOSIN_sales_qty.reindex(fulldates, fill_value=0)
total_Amazon_qty = sum(fulldates_DINOSIN_salesqty_Amazon)+sum(fulldates_DINOSIN_salesqty_AmazonFBA)
#igore Etsy and Direct, only 101 and 1
weighted_mean_price = (fulldates_prices_trend_Amazon*(weight_Amazon+ weight_AmazonFBA)\
                      +fulldates_prices_trend_shopify*weight_Shopify+fulldates_prices_trend_eBay*(weight_Ebay+weight_Etsy)).round(2)

demand_curve_All = Demand_Curve(fulldates_DINOSIN_salesqty_All,weighted_mean_price,'All Channels')
demand_curve_All.to_csv("demand_curve_All.csv")



# Price Elasticity
def Price_Elasticity(qty,price):

    df = pd.concat([qty,price], axis=1, join='inner', ignore_index=False)
    df = df[df['Price']!=0]
    df.set_index('Price',inplace=True)

    price_elasticity = []
    for i in range(1, df.shape[0]):
        qtydiff = 2 * (df.values[i] - df.values[i - 1]) / (df.values[i] + df.values[i - 1])
        pricediff = 2 * (df.index[i] - df.index[i - 1]) / (df.index[i] + df.index[i - 1])
        price_elasticity.append(qtydiff / pricediff)
    df2 = df.iloc[1:,:]
    df2['Price_elasticity'] = price_elasticity
    df2.replace([np.inf,-np.inf],np.nan,inplace = True)
    df2.dropna(inplace = True)
    df2_fit = sm.OLS(df2.reset_index().loc[:, ('Quantity')], sm.add_constant(df2.reset_index().loc[:, ('Price')])).fit()
    print(df2_fit.summary())
    sns.pairplot(df2.reset_index().loc[:, ('Price', 'Quantity')], y_vars='Price', x_vars='Quantity', kind="reg",
                 height=4, aspect=1)

    return df2

price_elasticity_Amazon = Price_Elasticity(fulldates_DINOSIN_salesqty_Amazonall,fulldates_prices_trend_Amazon)
#price_elasticity_Amazon.to_csv("price_elasticity_Amazon.csv")
price_elasticity_Shopify = Price_Elasticity(fulldates_DINOSIN_salesqty_Shopify,fulldates_prices_trend_shopify)
price_elasticity_Ebay = Price_Elasticity(fulldates_DINOSIN_salesqty_Ebay,fulldates_prices_trend_eBay)
price_elasticity_All = Price_Elasticity(fulldates_DINOSIN_salesqty_All,weighted_mean_price)



#--------------  Part 7: Classical approach    --------------------------------
# This part I calculate the cost by hand using HL listing data sheet

fixed_cost = 6.73*weight_Shopify+6.42*weight_AmazonFBA+6.7*weight_Amazon+7*weight_Ebay+6.9*weight_Etsy
vari_cost = 0.04*weight_Shopify+0.163*weight_AmazonFBA+0.163*weight_Amazon+0.179*weight_Ebay+0.16*weight_Etsy

# maximize profit
#func = -4.9154(1- vari_cost -0.1667)*x**2+(153.0701*(1-vari_cost-0.1667)+4.9154*fixed_cost)*x-153.0701*fixed_cost
a = -4.9154*(1- np.array(vari_cost) -0.1667)
b = 153.0701*(1-np.array(vari_cost)-0.1667)+4.9154*np.array(fixed_cost)
c = -153.0701*np.array(fixed_cost)

plt.style.use('ggplot')  # 使用‘ggplot风格美化图表’

def create_graph(a, b, c, d, e,metric):
    x = np.arange(d, e, 0.01)
    y = a * x ** 2 + b * x + c
    max_x = b/(-2*a)  # max value index
    max_y = a * max_x ** 2 + b * max_x + c
    plt.figure()
    plt.plot(x, y)
    plt.plot(max_x, max_y, 'ks')
    show_max = '[' + str(max_x.round(2)) + ' ' + str(max_y.round(2)) + ']'
    plt.annotate(show_max, xytext=(max_x, max_y), xy=(max_x, max_y))
    plt.xlabel('Price')
    plt.ylabel(metric)
    plt.title(metric+'~ price')
    plt.show()

create_graph(a, b, c, 10, 25,'Margin')

#maximize revenue
a2 = np.array(-4.9154)
b2 = np.array(153.0701)
create_graph(a2, b2, 0, 10, 25,'Revenue')


#---------------------- Part 8: Dynamic Pricing  ------------------------------------------
#(1) check prior
# Using all channels together
# For other SKUs, maybe other prior distribution will be more appropriate
df = pd.concat([fulldates_DINOSIN_salesqty_All, weighted_mean_price], axis=1, join='inner', ignore_index=False)
df = df[df['Price'] != 0]
df_theta = df.groupby('Price').sum() / df.groupby('Price').count()

#distribution of theta
plt.hist(x=df_theta,bins=50,density=True,histtype='stepfilled',color='red',alpha=0.5,label='histogram')

plt.figure()
df_theta.plot(kind='kde',label='distribution of theta')
np.mean(df_theta.values)
np.var(df_theta.values)  #alpha0=5  beta0=1/18
# probability density function
y1 = stats.gamma.pdf(range(225), 5, scale=18)  # "α=5,β=1/18, scale = 1/beta"
plt.plot(range(225),y1,label='prior distribution')  #quite similar
plt.legend()
#cumulative distribution function
y2 = stats.gamma.cdf(range(225), a=5, scale=18)
plt.plot(range(225), y2, "y-", label=(r'$\alpha=, \beta=$'))


#(2)Thompson Sampling

def sample_actual_demand1(price):
    demand = true_slop + true_intercept * price
    return np.random.poisson(demand, 1)[0]   # np.random.poisson(lambda, size) E[x]=lambda


def sample_actual_demand2(price):
    demand = demand_curve_All['Quantity'][demand_curve_All.index == price]
    return np.random.poisson(demand, 1)[0]  # np.random.poisson(lambda, size) E[x]=lambda


# sample mean demands for each price level???
def sample_demands_from_model(p_theta):
    return list(map(lambda v:
                    np.random.gamma(v['alpha'], 1 / v['beta']), p_theta))


# return price that maximizes the profit
def optimal_price_profit(prices, demands, variable_cost=vari_cost, fixed_cost=fixed_cost):
    profit = np.array(demands) * prices * (1 - variable_cost - 0.1667) - np.array(demands) * fixed_cost
    price_index = np.argmax(np.array(profit))
    return price_index, prices[price_index]


def optimal_price_rev(prices, demands, variable_cost=vari_cost, fixed_cost=fixed_cost):
    price_index = np.argmax(np.array(demands * prices))
    return price_index, prices[price_index]

best_price_profit_contidemand = []
best_price_profit_contidemand2 = []
best_price_profit_discrdemand = []
best_price_rev_contidemand = []
best_price_rev_discrdemand = []

for epoch in range (0,200):
    weighted_price2 = weighted_mean_price[weighted_mean_price!=0]
    weighted_price2 = weighted_price2[weighted_price2!=13.95]

    # parameters
    prices = np.array(weighted_price2)
    alpha_0 = 5  # parameter of the prior distribution
    beta_0 = 1/18  # parameter of the prior distribution

    # parameters of the true (unknown) demand model
    true_slop = -4.9154
    true_intercept = 153.0701

    # prior distribution for each price
    p_theta = []
    for p in prices:
        p_theta.append({'price': p, 'alpha': alpha_0, 'beta': beta_0})

    # simulation loop
    #------------------a. maximize profit,linear demand-------------------------------------
    #for t in range(0, 1500):
    #    demands = sample_demands_from_model(p_theta)
    #    price_index_t, price_t = optimal_price_profit(prices, demands)
#
    #    # offer the selected price and observe demand
    #    demand_t = sample_actual_demand1(price_t)
#
    #    # update model parameters
    #    v = p_theta[price_index_t]
    #    v['alpha'] = v['alpha'] + demand_t
    #    v['beta'] = v['beta'] + 1
    #best_price_profit_contidemand.append(v['price'])
    #best_price_profit_contidemand2.append(v['price'])
    #------------------b.maximize profit,discrete demand----------------------------------------
    for t in range(0, 1500):
        demands = sample_demands_from_model(p_theta)
        price_index_t, price_t = optimal_price_profit(prices, demands)

        # offer the selected price and observe demand
        demand_t = sample_actual_demand2(price_t)

        # update model parameters
        m = p_theta[price_index_t]
        m['alpha'] = m['alpha'] + demand_t
        m['beta'] = m['beta'] + 1
    best_price_profit_discrdemand.append(m['price'])
    #---------------------------------c.maximize rev, linear demand----------------------------------
    #for t in range(0, 1500):
    #    demands = sample_demands_from_model(p_theta)
    #    price_index_t, price_t = optimal_price_rev(prices, demands)
#
    #    # offer the selected price and observe demand
    #    demand_t = sample_actual_demand1(price_t)
#
    #    # update model parameters
    #    u = p_theta[price_index_t]
    #    u['alpha'] = u['alpha'] + demand_t
    #    u['beta'] = u['beta'] + 1
    #best_price_rev_contidemand.append(u['price'])
    #--------------------------------------d.#maximize rev,discrete demand-------------------------------
    #for t in range(0, 1500):
    #    demands = sample_demands_from_model(p_theta)
    #    price_index_t, price_t = optimal_price_rev(prices, demands)
#
    #    # offer the selected price and observe demand
    #    demand_t = sample_actual_demand2(price_t)
#
    #    # update model parameters
    #    n = p_theta[price_index_t]
    #    n['alpha'] = n['alpha'] + demand_t
    #    n['beta'] = n['beta'] + 1
    #best_price_rev_discrdemand.append(n['price'])

fig,ax =plt.subplots()
ax.hist(best_price_profit_contidemand2,20)
plt.xticks(best_price_profit_contidemand2)
ax.xaxis.set_tick_params(rotation=45,labelsize = 8)
ax.set_ylabel('Number of times each price is selected as optimal price')
plt.title('maximize profit with linear demand ')
plt.show()

fig,ax2 =plt.subplots()
ax2.hist(best_price_profit_discrdemand,20)
#ax2.set_ylim(10, 22)
plt.xticks(best_price_profit_discrdemand)
ax2.xaxis.set_tick_params(rotation=45,labelsize = 8)
plt.title('maximize profit with discrete demand ')
plt.show()

fig,ax3 =plt.subplots()
ax3.hist(best_price_rev_contidemand,20)
#ax2.set_ylim(10, 22)
plt.xticks(best_price_rev_contidemand)
ax3.xaxis.set_tick_params(rotation=45,labelsize = 8)
plt.title('maximize revenue with linear demand ')

fig,ax4 =plt.subplots()
ax4.hist(best_price_rev_discrdemand,20)
#ax2.set_ylim(10, 22)
plt.xticks(best_price_rev_discrdemand)
ax4.xaxis.set_tick_params(rotation=45,labelsize = 8)
plt.title('maximize revenue with discrete demand ')


#posterior distribution of theta
prior = np.random.gamma(alpha_0, 1/beta_0,225)
#posterior = pd.Series(np.random.gamma( v['alpha'], 1/v['beta'],225))
posterior = pd.Series(np.random.gamma( m['alpha'], 1/m['beta'],225))
#posterior = pd.Series(np.random.gamma( u['alpha'], 1/u['beta'],225))
#posterior = pd.Series(np.random.gamma( n['alpha'], 1/n['beta'],225))


#-------------------------------Part 9: Multiple products' Pricing-------------------------------
#(1)Import substitutable products
Product_Category = 'Childrens Bedding'
Size = 'SIN'
RangeDINOSIN_sales_data=sales_data[sales_data['Product Category'] == Product_Category]
RangeDINOSIN_sales_data=RangeDINOSIN_sales_data[RangeDINOSIN_sales_data['Source'].str.contains('AMAZON')]
RangeDINOSIN_sales_data=RangeDINOSIN_sales_data[RangeDINOSIN_sales_data['Main SKU'].str.contains(Size)]

simp_RangeDINOSIN_sales_data = RangeDINOSIN_sales_data.loc[:,['Received Date','Main SKU','Quantity']]
RangeDINOSIN_sales_qty = simp_RangeDINOSIN_sales_data.groupby(['Received Date','Main SKU'],as_index=False).sum()
simp_RangeDINOSIN_sales_rev = RangeDINOSIN_sales_data.loc[:,['Received Date','Main SKU','Transaction GBP']]
RangeDINOSIN_sales_rev = simp_RangeDINOSIN_sales_rev.groupby(['Received Date','Main SKU'],as_index=False).sum()
RangeDINOSIN_sales_rev = RangeDINOSIN_sales_rev.set_index('Received Date')

conv_RangeDINOSIN_sales_qty = pd.pivot(RangeDINOSIN_sales_qty, index='Received Date', columns='Main SKU',values='Quantity')
conv_RangeDINOSIN_sales_qty = conv_RangeDINOSIN_sales_qty.fillna(0)
# all the substitutable products list
conv_RangeDINOSIN_sales_qty.columns

current_date = sales_data.loc[sales_data.index[-1]]['Received Date']
start_date = sales_data.loc[sales_data.index[0]]['Received Date']
fulldates = pd.date_range(start=start_date, end=current_date,freq='7D')
# all prices of Amazon
AMAZON_Prices = prices[prices['Channel']=='Amazon']
# Amazon prices of all substitutable products
SIN_price = pd.DataFrame()
for i in range(len(conv_RangeDINOSIN_sales_qty.columns)):
    new_df = AMAZON_Prices[prices['SKU'] == conv_RangeDINOSIN_sales_qty.columns.to_list()[i]]
    new_df.set_index(["Date"], inplace=True)
    new_df_2= pd.concat([conv_RangeDINOSIN_sales_qty.iloc[:,i],new_df['Price']],axis=1, join='inner', ignore_index = False)
    new_df_3 = new_df_2.groupby('Price').sum() / new_df_2.groupby('Price').count()
    SIN_price = pd.concat([SIN_price,new_df_3],axis=1,join='outer',ignore_index=False)

SIN_price = SIN_price.fillna(0)
#AMAZON+AMAZONFBA account for 73%

# Pricing on Multiple Products
# prices - k-dimensional vector of allowed price levels
# demands - n x k matrix of demand predictions
# c - required sum of prices
# where k is the number of price levels, n is number of products

prices = np.array(SIN_price.index.to_list())
demands = SIN_price.T
# setting the total amount of all substitutable products
c = 170
n,k = demands.shape
# prepare inputs
r = np.multiply(np.tile(prices, n), np.array(demands).reshape(1, k * n))
vari_cost = [i * (0.163+0.1667) for i in prices]
# costs are checked by hand
fixed_cost = [7.35,6.42,6.42,6.42,7.11,6.42,6.19,6.32,6.67,6.25]  #costs are checked by hand
costs = []
for i in range(n):
    costs.append(fixed_cost[i]+np.array(vari_cost))
margin = np.multiply( np.array(demands).reshape(1, k * n),(np.tile(prices, n)-np.array(costs).reshape(1,k*n)))

A = np.array([[
    1 if j >= k * (i) and j < k * (i + 1) else 0
    for j in range(k * n)
] for i in range(n)])
#A = np.append(A, np.tile(prices, n), axis=0)
A=np.vstack((A,np.tile(prices, n)))
b = [np.append(np.ones(n), c)]

# solve the linear programing
# maximize revenue
res1 = linprog(-r.flatten(),
              A_eq=A,
              b_eq=b,
              bounds=(0, 1))

result = np.array(res1.x).reshape(n, k)
rev_optimal_price = np.array(res1.x).reshape(n, k)*prices

#maximize profit
res2 = linprog(-margin.flatten(),
              A_eq=A,
              b_eq=b,
              bounds=(0, 1))

result2 = np.array(res2.x).reshape(n, k)
margin_optimal_price = np.array(res2.x).reshape(n, k)*prices
