import os,glob
import pandas as pd
import numpy as np
from statsmodels.graphics.gofplots import ProbPlot
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import math

#---------------  Part 1:Combine all inventory excels together   ----------------------------
path = "D:/Warwick/dissertation/Original_Data/Inventory/"
all_files = glob.glob(os.path.join(path, "*.csv")) #make list of paths

initial = pd.read_csv("D:/Warwick/dissertation/Original_Data/Inventory/StockApril2019.csv")
initial = initial.set_index('product_variant_sku')
for file in all_files:
    # Getting the file name without extension
    file_name = os.path.splitext(os.path.basename(file))[0]
    df = pd.read_csv(file)
    df.dropna(axis=0, how='any', inplace=True)
    df = df.set_index('product_variant_sku')
    df.rename(columns={'ending_quantity': file_name},inplace=True)
    df_full = pd.concat([initial,df.iloc[:,-1]], axis=1, join='outer', ignore_index=False)
    initial = df_full

initial.to_csv('D:/Warwick/dissertation/Original_Data/Inventory_full.csv', mode='a', index=True,header=True)

# I have modified the saved inventory sheet
Inventory_level = pd.read_csv('D:/Warwick/dissertation/Original_Data/Full_Inventory_data.csv')
Inventory_level.fillna(0,inplace=True)
Inventory_level.drop(['product_title','product_variant_title'],axis=1,inplace = True)

#inventory trend of product range dinosnore
RangeDINOS_Inventory_level = Inventory_level[Inventory_level['product_variant_sku'].str.contains("DINOSNORE")]
#inventory trend of DINOSNORE-HL01-SIN
DINOSNORE_Inventory_level = Inventory_level[Inventory_level['product_variant_sku'] =='DINOSNORE-HL01-SIN']
#inventory trend of SKUs except DINOSNORE-HL01-SIN in product range dinosnore
Others_RangeDINOS_Inventory_level =RangeDINOS_Inventory_level.drop(54,axis=0)

RangeDINOS_Inventory_level.set_index('product_variant_sku',inplace=True)
DINOSNORE_Inventory_level.set_index('product_variant_sku',inplace=True)
Others_RangeDINOS_Inventory_level.set_index('product_variant_sku',inplace=True)
DINOSNORE_Inventory_level = DINOSNORE_Inventory_level.T
Others_RangeDINOS_Inventory_level = Others_RangeDINOS_Inventory_level.T
RangeDINOS_Inventory_level = RangeDINOS_Inventory_level.T

#---------------------- Part 2:  Import price data   ---------------------------
prices = pd.read_excel("D:/Warwick/dissertation/Original_Data/2021-06-21 MASTER DASHBOARD_PostMeet.xlsm", sheet_name='Price Data',skiprows=1)
DINOSINORE_prices=prices[prices['SKU'] == 'DINOSNORE-HL01-SIN']
DINOSINORE_prices = DINOSINORE_prices.set_index('Date')
#Remove channels with Marketplace
DINOSINORE_prices=DINOSINORE_prices[~DINOSINORE_prices['Channel'].str.contains("Market")]

def prices_chanel(df,channel):
    df = df[df['Channel'] == channel]
    df = df.resample('M').mean().round(2)
    # Both start from 2019-01
    df = df.iloc[2:,:]
    return df

monthly_DINOSINORE_prices_Amazon = prices_chanel(DINOSINORE_prices,'Amazon')
monthly_DINOSINORE_prices_eBay = prices_chanel(DINOSINORE_prices,'eBay')
monthly_DINOSINORE_prices_Shopify = prices_chanel(DINOSINORE_prices,'Shopify')

DINOSNORE_Inventory_level.index = monthly_DINOSINORE_prices_Amazon.index

#--------------------- Part 3: Plot of inventory and price trend of different channels --------------------
tick_spacing = 20
fig, [ax1, ax3, ax5] = plt.subplots(nrows=3, ncols=1)
plt.subplots_adjust(wspace =0, hspace =1)
#Amazon
ax1.plot(DINOSNORE_Inventory_level.index,DINOSNORE_Inventory_level,'b',label='Inventory')
ax1.set_ylim(0,5000)
ax1.set_title('Inventory Level and Amazon price trend of best selling SKU ',fontsize=20)
ax1.set_ylabel('Inventory Level')
ax1.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax1.xaxis.set_tick_params(rotation=45,labelsize = 8)
#price trend
ax2 = ax1.twinx()
ax2.set_ylim((DINOSINORE_prices['Price'].min()-1),(DINOSINORE_prices['Price'].max()+1))
ax2.plot(monthly_DINOSINORE_prices_Amazon.index,monthly_DINOSINORE_prices_Amazon,'r',label='Amazon Price')
ax2.set_ylabel('price')
plt.legend()
#Shopify
ax3.plot(DINOSNORE_Inventory_level.index,DINOSNORE_Inventory_level,'b',label='Inventory')
ax3.set_ylim(0,5000)
ax3.set_title('Inventory Level and Shopify price trend of best selling SKU ',fontsize=20)
ax3.set_ylabel('Inventory Level')
ax3.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax3.xaxis.set_tick_params(rotation=45,labelsize = 8)
ax4 = ax3.twinx()
ax4.set_ylim((DINOSINORE_prices['Price'].min()-1),(DINOSINORE_prices['Price'].max()+1))
ax4.plot(monthly_DINOSINORE_prices_Amazon.index,monthly_DINOSINORE_prices_Shopify,'r',label='Shopify Price')
ax4.set_ylabel('price')
plt.legend()
# eBay
ax5.plot(DINOSNORE_Inventory_level.index,DINOSNORE_Inventory_level,'b',label='Inventory')
ax5.set_ylim(0,5000)
ax5.set_title('Inventory Level and eBay price trend of best selling SKU ',fontsize=20)
ax5.set_ylabel('Inventory Level')
ax5.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax5.xaxis.set_tick_params(rotation=45,labelsize = 8)
ax6 = ax5.twinx()
ax6.set_ylim((DINOSINORE_prices['Price'].min()-1),(DINOSINORE_prices['Price'].max()+1))
ax6.plot(monthly_DINOSINORE_prices_Amazon.index,monthly_DINOSINORE_prices_eBay,'r',label='eBay Price')
ax6.set_ylabel('price')
plt.legend()
plt.show()


#---------------------   Below are my trials on relationship of price and inventory   --------------------------
#regression on price~SKU_inventory
price_otherSKUs_inventory0 = pd.concat([DINOSNORE_Inventory_level,monthly_DINOSINORE_prices_Amazon], axis=1, join='inner', ignore_index=False)
sns.pairplot(price_otherSKUs_inventory0, x_vars=price_otherSKUs_inventory0.columns, y_vars='Price',kind="reg", height=5, aspect=1)
price_otherSKUs_inventory_fit0 = sm.OLS(price_otherSKUs_inventory0.iloc[:,-1], sm.add_constant(price_otherSKUs_inventory0.iloc[:,0])).fit()
price_otherSKUs_inventory_fit0.summary()


Others_RangeDINOS_Inventory_level.index = monthly_DINOSINORE_prices_Amazon.index
#regression on price~all other skus
price_otherSKUs_inventory = pd.concat([Others_RangeDINOS_Inventory_level,monthly_DINOSINORE_prices_Amazon], axis=1, join='inner', ignore_index=False)
sns.pairplot(price_otherSKUs_inventory, x_vars=price_otherSKUs_inventory.columns[4:9], y_vars='Price',kind="reg", height=5, aspect=1)
price_otherSKUs_inventory_fit = sm.OLS(price_otherSKUs_inventory.iloc[:,-1], sm.add_constant(price_otherSKUs_inventory.iloc[:,:8])).fit()
price_otherSKUs_inventory_fit.summary()

#regression on price~all SKUs inventory in the range
RangeDINOS_Inventory_level.index = monthly_DINOSINORE_prices_Amazon.index
price_otherSKUs_inventory1 = pd.concat([RangeDINOS_Inventory_level,monthly_DINOSINORE_prices_Amazon], axis=1, join='inner', ignore_index=False)
sns.pairplot(price_otherSKUs_inventory1, x_vars=price_otherSKUs_inventory1.columns[:10], y_vars='Price',kind="reg", height=5, aspect=1)
price_otherSKUs_inventory_fit1 = sm.OLS(price_otherSKUs_inventory1.iloc[:,-1], sm.add_constant(price_otherSKUs_inventory1.iloc[:,:10])).fit()
price_otherSKUs_inventory_fit1.summary()

