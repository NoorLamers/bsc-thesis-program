import numpy as np
from scipy.stats import norm
import math
       
class Stock:    
    def __init__(self, stockprice, volatility, riskfreerate, dividendamounts, dividendtimes, strikeprice, maturity, steps, model):
        
                 self.S0 = stockprice
                 self.sigma = volatility
                 self.r = riskfreerate
                 self.D = dividendamounts
                 self.t_D = dividendtimes
                 self.n = steps
                 self.model = model
                 self.K = strikeprice
                 self.T = maturity
                 self.dt = self.T / self.n
                 
                 self.S0_esc = None
                 self.Kplus = None
                 self.X_n = None
                 self.X_f = None
                 
                 self.compute_constants()
                 
# function that per model computes constants needed to calculate the corresponding otoin price (or stock price behaviour)
# computes constants such as the escrowed stock price, adjusted volatilities and adjsted strike
    def compute_constants(self):
        
        if self.model == 'escrowed':
            
            self.S0_esc = self.S0 - np.exp(-self.r * self.t_D) * self.D
            
        elif self.model == 'adjusted strike':
            
            self.Kplus = np.exp(self.r* (self.T -self.t_D)) * self.D
            
        elif self.model == 'VA1':
            
            self.S0_esc = self.S0 - np.exp(-self.r * self.t_D) * self.D
            self.sigma = self.S0 / self.S0_esc * self.sigma
            
        elif self.model == 'VA2':
            
            self.S0_esc = self.S0 - np.exp(-self.r * self.t_D) * self.D
            
            sigma_squared = ( ((self.S0/self.S0_esc) * self.sigma )**2 * self.t_D + self.sigma **2 * (self.T - self.t_D) ) / self.T
            self.sigma = np.sqrt(sigma_squared)
            
        elif self.model == 'VA3':
            
            self.S0_esc = self.S0 - np.exp(-self.r * self.t_D) * self.D

            s = np.log(self.S0_esc)
            k = np.log(self.K * np.exp(-self.r * self.T))
            a = (s-k) / (self.sigma * np.sqrt(self.T)) + self.sigma * np.sqrt(self.T) / 2
            b = a + self.sigma * np.sqrt(self.T) / 2
            
            sigma_squared = self.sigma ** 2 + self.sigma * np.sqrt(math.pi / (2 * self.T)) * ( 4 * np.exp(a**2 /2 - s) * np.exp(-self.r * self.t_D) * self.D * ( norm.cdf(a) - norm.cdf(a - self.sigma * self.t_D / np.sqrt(self.T))) + np.exp(b**2 /2 - 2*s) * np.exp(-self.r * 2 * self.t_D) * self.D * self.D * (norm.cdf(b) - norm.cdf(b - 2 * self.sigma * self.t_D / np.sqrt(self.T))))
            self.sigma = np.sqrt(sigma_squared)      
        
        elif self.model == 'LEH':
            
            self.X_n = (self.T - self.t_D) / self.T * self.D * np.exp(-self.r* self.t_D)
            self.X_f = self.t_D / self.T * self.D * np.exp(-self.r* self.t_D)
           
# function that returns the stock price behaviour of a certain model 
# random seed is used especially when plotting different stock price paths over time
    def path(self, random_seed=None):
        
        if random_seed is not None:
            np.random.seed(random_seed)
            
        # generate array of random variables, and an array of time increments
        dW_array = np.cumsum(np.random.normal( 0, 1, self.n) * np.sqrt(self.dt))
        dt_array = np.linspace(0, self.T, self.n)

        
        if self.model == 'gbm': 
            
            path = self.S0 * np.exp((self.r - 0.5 * self.sigma**2 ) * dt_array + self.sigma * np.sqrt(dt_array) * dW_array)
        
        elif self.model == 'escrowed':

            path = self.S0_esc * np.exp((self.r- 0.5 * self.sigma**2 ) * dt_array + self.sigma * np.sqrt(dt_array) * dW_array)        
              
        elif self.model == 'adjusted strike':
            
            path = self.S0 * np.exp((self.r - 0.5 * self.sigma**2 ) * dt_array + self.sigma * np.sqrt(dt_array) * dW_array) - np.exp(self.r * (self.T - self.t_D)) * self.D
 
    #diracdelta model represents          
        elif self.model == 'diracdelta':
            path = np.zeros(self.n)
            drop_index = int(self.t_D / self.T * self.n)
            
            path[:drop_index] = self.S0 * np.exp((self.r - 0.5 * self.sigma**2) * dt_array[:drop_index] + self.sigma * np.sqrt(dt_array[:drop_index]) * dW_array[:drop_index])

            drop = path[drop_index - 1] - self.D
            path[drop_index] = drop      

            path[drop_index + 1:] = drop * np.exp((self.r - 0.5 * self.sigma**2) * dt_array[:self.n - drop_index - 1:] + self.sigma * np.sqrt(dt_array[: self.n - drop_index - 1:]) * dW_array[drop_index + 1:])
               
        elif self.model == 'VA1':
            
            path = self.S0_esc * np.exp((self.r- 0.5 * self.sigma**2 ) * dt_array + self.sigma * np.sqrt(dt_array) * dW_array)
        
        elif self.model == 'VA2':
            
            path = self.S0_esc * np.exp((self.r- 0.5 * self.sigma**2 ) * dt_array + self.sigma * np.sqrt(dt_array) * dW_array)
               
        elif self.model == 'VA3':
     
            path = self.S0_esc * np.exp((self.r- 0.5 * self.sigma**2 ) * dt_array + self.sigma * np.sqrt(dt_array) * dW_array)
        
        elif self.model == 'LEH':
        
            path = (self.S0 - self.X_n) * np.exp((self.r- 0.5 * self.sigma**2 ) * dt_array +self.sigma * np.sqrt(dt_array) * dW_array) - self.X_f * np.exp(self.r* dt_array)
            
        return path   

    
# function to simulate the black scholes option price for a call option with dividend d
# the function adjusts the original black scholes function accordingly in the case of all models other than the original black scholes
def BS_call(stock_instance):    
     
     K = stock_instance.K
     S0 = stock_instance.S0

     if stock_instance.model == 'adjusted strike':
        K += stock_instance.Kplus
        
     elif stock_instance.model in ['escrowed', 'VA1', 'VA2', 'VA3']:
         S0 = stock_instance.S0_esc
    
     elif stock_instance.model == 'LEH':
         S0 -= stock_instance.X_n
         K += stock_instance.X_f * np.exp(stock_instance.r * stock_instance.T)
     
     c1 = (np.log( S0 / K ) + ( stock_instance.r + 0.5 * stock_instance.sigma**2 ) * stock_instance.T) / (stock_instance.sigma * np.sqrt(stock_instance.T))
     c2 = c1 - stock_instance.sigma * np.sqrt(stock_instance.T)
     
     return ( S0 * norm.cdf(c1) - K * np.exp(-stock_instance.r * stock_instance.T) * norm.cdf(c2) )
    

# function for simulating a monte carlo price for a call option, given a stock's path
def MC_call_eu(stock_instance, m): 
    
    option_prices = np.zeros(m)
        
    for i in range(m):
        path = stock_instance.path()
        option_payoff = np.maximum(path[-1] - stock_instance.K, 0)
        option_price = np.exp(-stock_instance.r * stock_instance.T) * option_payoff
        option_prices[i] = option_price

    return np.mean(option_prices)



