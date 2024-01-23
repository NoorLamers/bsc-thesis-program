import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import math
       
class Stock_multdiv:
    
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
            
            self.S0_esc = self.S0 - np.sum(np.exp(-self.r * self.t_D) * self.D)
            
        elif self.model == 'adjusted strike':
            
            self.Kplus = np.sum(np.exp(self.r* (self.T -self.t_D)) * self.D)
            
        elif self.model == 'VA1':
            
            self.S0_esc = self.S0 - np.sum(np.exp(-self.r * self.t_D) * self.D)
            self.sigma = self.S0 / self.S0_esc * self.sigma
            
        elif self.model == 'VA2':

            self.S0_esc = self.S0 - np.sum(np.exp(-self.r * self.t_D) * self.D)
            
            sigma_squared = self.sigma**2 / self.T * ( (self.S0 / (self.S0_esc))**2 * self.t_D[0] + (self.T - self.t_D[-1]) )
            for j in range(1, len(self.t_D)):
                
                d = np.sum(np.exp(-self.r * self.t_D[j:]) * self.D[j:])
                sigma_squared += self.sigma ** 2 / self.T * (self.S0 / (self.S0 - d)) ** 2 * (self.t_D[j] - self.t_D[j-1])

            self.sigma = np.sqrt(sigma_squared)
            
        elif self.model == 'VA3':
            
            self.S0_esc = self.S0 - np.sum(np.exp(-self.r * self.t_D) * self.D)
            
            s = np.log(self.S0_esc)
            k = np.log(self.K * np.exp(-self.r * self.T))
            a = (s-k) / (self.sigma * np.sqrt(self.T)) + self.sigma * np.sqrt(self.T) / 2
            b = a + self.sigma * np.sqrt(self.T) / 2
            
            sum1 = 0
            sum2 = 0
            
            for i in range(len(self.D)):
                t_i = self.t_D[i]
                D_i = self.D[i]
                sum1 += np.exp(-self.r * t_i) * D_i * ( norm.cdf(a) - norm.cdf(a - self.sigma * t_i / np.sqrt(self.T)))
            sum1 = sum1 * 4 * np.exp(a**2 /2 - s)
            
            for i in range(len(self.D)):
                t_i = self.t_D[i]
                D_i = self.D[i]
                for j in range(len(self.D)):
                    t_j = self.t_D[j]
                    D_j = self.D[j]
                    
                    sum2 +=  np.exp(-self.r * (t_i + t_j)) * D_i * D_j * (norm.cdf(b) - norm.cdf(b - 2 * self.sigma * np.minimum(t_i, t_j) / np.sqrt(self.T)))
            sum2 = sum2* np.exp(b**2 /2 - 2*s)
    
            sigma_squared = self.sigma ** 2 + self.sigma * np.sqrt(math.pi / (2 * self.T)) * (sum1 + sum2)
            
            self.sigma = np.sqrt(sigma_squared)      
        
        elif self.model == 'LEH':
            
            self.X_n = np.sum((self.T - self.t_D) / self.T * self.D * np.exp(-self.r* self.t_D))
            self.X_f = np.sum(self.t_D / self.T * self.D * np.exp(-self.r* self.t_D))
           
 # function that returns the stock price behaviour of a certain model 
    def path(self):
        
        # generate array of random variables, and an array of time increments
        dW_array = np.cumsum(np.random.normal( 0, 1, self.n) * np.sqrt(self.dt))
        dt_array = np.linspace(0, self.T, self.n)
        

        # path array represents the effect of the drift term and diffusion term over time
        if self.model == 'gbm': 
            
            path = self.S0 * np.exp((self.r - 0.5 * self.sigma**2 ) * dt_array + self.sigma * np.sqrt(dt_array) * dW_array)
        
        elif self.model == 'escrowed':

            path = self.S0_esc * np.exp((self.r- 0.5 * self.sigma**2 ) * dt_array + self.sigma * np.sqrt(dt_array) * dW_array)        
              
        elif self.model == 'adjusted strike':
            
            path = self.S0 * np.exp((self.r - 0.5 * self.sigma**2 ) * dt_array + self.sigma * np.sqrt(dt_array) * dW_array) - np.exp(-self.r * (self.T - self.t_D)) * self.D
        
        elif self.model == 'diracdelta':
            
            paths = self.S0 * np.exp((self.r - 0.5 * self.sigma**2 ) * dt_array + self.sigma * np.sqrt(dt_array) * dW_array)
            pathd = np.zeros(self.n)
            dividend_indices = np.searchsorted(dt_array, self.t_D)
            
            # Calculate the indices where dividends start and end
            div_start = np.insert(dividend_indices, 0, 0)
            div_end = np.append(dividend_indices, self.n)
            
            # Set values before the first dividend
            pathd[:dividend_indices[0]] = 0
            
            # Set values between dividends
            for i in range(len(dividend_indices)):
                pathd[div_start[i]:div_end[i]] = np.sum(self.D[:i])
            
            # Set values after the last dividend
            pathd[dividend_indices[-1]:] = np.sum(self.D)
            
            path = paths - pathd

                            
        elif self.model == 'VA1':
            
            path = self.S0_esc * np.exp((self.r- 0.5 * self.sigma**2 ) * dt_array + self.sigma * np.sqrt(dt_array) * dW_array)
        
        elif self.model == 'VA2':
            
            path = self.S0_esc * np.exp((self.r- 0.5 * self.sigma**2 ) * dt_array + self.sigma * np.sqrt(dt_array) * dW_array)
               
        elif self.model == 'VA3':
     
            path = self.S0_esc * np.exp((self.r- 0.5 * self.sigma**2 ) * dt_array + self.sigma * np.sqrt(dt_array) * dW_array)
        
        elif self.model == 'LEH':
        
            path = (self.S0 - self.X_n) * np.exp((self.r- 0.5 * self.sigma**2 ) * dt_array +self.sigma * np.sqrt(dt_array) * dW_array) - self.X_f * np.exp(self.r* dt_array)
            
        return path         
















