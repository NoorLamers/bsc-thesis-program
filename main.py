from functions import *
from plot import *
from functions_multdiv import *

#all models in this program
models = ['gbm', 'escrowed', 'diracdelta', 'adjusted strike', 'VA1', 'VA2', 'VA3', 'LEH']

#plots all stock paths
pltallstockpath(models, random_seed=2)

#initializes a random stock used for the adjusted strike model
stock =  Stock(stockprice=100, 
              volatility=0.3, 
              riskfreerate=0.05, 
              dividendamounts= 10, 
              dividendtimes=0.5, 
              strikeprice=100, 
              maturity=1, 
              steps=100, 
              model='adjusted strike')

#initializes a random stock used for the adjusted strike model with multiple dividends
stock_multdiv =  Stock_multdiv(stockprice=100, 
              volatility=0.3, 
              riskfreerate=0.05, 
              dividendamounts= np.array([5, 5, 5, 5]), 
              dividendtimes=np.array([0.5, 1.5, 2.5, 3.5]), 
              strikeprice=100, 
              maturity=4, 
              steps=100, 
              model='adjusted strike')


