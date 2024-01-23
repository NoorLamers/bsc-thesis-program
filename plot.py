from functions import *
import matplotlib.pyplot as plt
import numpy as np
import time
import random

# function that plots the absolute error of using monte carlo over increasing simulations
def pltMCerror(simulations):
    
    stock = Stock(stockprice = 100, 
                  volatility = 0.3, 
                  riskfreerate = 0.1, 
                  dividendamounts = 0, 
                  dividendtimes = 0, 
                  strikeprice = 100, 
                  maturity = 1, 
                  steps = 100, 
                  model = 'gbm')
    
    BS_price = BS_call(stock)
    
    # computes the values for both the primary and secondary y-axis
    error_terms = np.zeros(len(simulations))
    simulation_times = np.zeros(len(simulations))
    
    for i, m in enumerate(simulations):
        start_time = time.time()
        error_terms[i] = abs(MC_call_eu(stock, m) - BS_price)
        end_time = time.time()
        simulation_times[i] = end_time - start_time

    plt.figure(figsize=(10, 6))

    plt.plot(simulations, error_terms, marker='o', linestyle='', label='Error Term')
    plt.ylim(0, max(error_terms) * 1.1)
    
    plt.xlabel('Number of Simulations (log scale)')
    plt.ylabel('Error Term')
    plt.legend(loc='upper left')
    
    ax2 = plt.gca().twinx()
    ax2.plot(simulations, simulation_times, marker='x', linestyle='--', color='red', label='Simulation Time')
    ax2.set_ylabel('Simulation Time', color='black')
    ax2.set_ylim(0, max(simulation_times) * 1.1)
    ax2.legend(loc='upper right')
    
    plt.xscale('log')
    plt.grid()
    plt.show()

# function used to calculate the average error and corresponding computation time of monte carlo simulation
def average_error_and_time(simulations, num_simulations):
    
    stock = Stock(stockprice = 100, 
                  volatility = 0.3, 
                  riskfreerate = 0.1, 
                  dividendamounts = 0, 
                  dividendtimes = 0, 
                  strikeprice = 100, 
                  maturity = 1, 
                  steps = 100, 
                  model = 'gbm')
    
    BS_price = BS_call(stock)
    
    avg_error_terms = np.zeros(len(simulations))
    avg_simulation_times = np.zeros(len(simulations))

    for j in range(num_simulations):
        error_terms = np.zeros(len(simulations))
        simulation_times = np.zeros(len(simulations))

        for i, m in enumerate(simulations):
            start_time = time.time()
            error_terms[i] = abs(MC_call_eu(stock, m) - BS_price)
            end_time = time.time()
            simulation_times[i] = end_time - start_time

        avg_error_terms += error_terms
        avg_simulation_times += simulation_times
        
    # calculate the average error term and average simulation time over the number of simulations
    avg_error_terms /= num_simulations
    avg_simulation_times /= num_simulations
    
    print("Average Error Terms:", avg_error_terms)
    print("Average Simulation Times:", avg_simulation_times)

# function that plots stock paths over time, random seed is used for better comparison of the paths
def pltallstockpath(models, random_seed=None):
    
    plt.figure(figsize=(10, 6))
       
    for m in models:
        stock = Stock(stockprice = 100, 
                      volatility = 0.3, 
                      riskfreerate = 0.05, 
                      dividendamounts = 10, 
                      dividendtimes = 0.5, 
                      strikeprice = 100, 
                      maturity = 1, 
                      steps = 100, 
                      model = m)
        path = stock.path(random_seed=random_seed)
        time_steps = np.linspace(0, stock.T, len(path))
 
        # set colours are used for the different models
        if m == 'gbm':
            clr = 'steelblue'
        elif m == 'escrowed':
            clr = 'sandybrown'
        elif m == 'adjusted strike':
            clr = 'palevioletred'
        elif m == 'VA1':
            clr = 'gold'
        elif m == 'VA2':
            clr = 'yellowgreen'
        elif m == 'VA3':
            clr = 'olivedrab'
        elif m == 'LEH':
            clr = 'darkslateblue'
        elif m == 'diracdelta':
            clr = 'darkorange'
            
        plt.plot(time_steps, path, label=m, color=clr)
        
    #sometimes the code line below is used to add a line representing the ex-date
    #plt.axvline(x=stock.t_D, color='grey', ls=':', lw=2, label = "ex-date")
        
    plt.xlabel('Time (Years)')
    plt.ylabel('Stock Price')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.yticks(np.arange(int(min(path) - 15), int(max(path) + 30), 10))
    plt.xlim(0, stock.T)

    plt.tight_layout()
    plt.show()
 
 # function plots option values for given maturirties for various models 
def pltoptionval():
    
    maturity = [1,2,3,4,5,6,7,8,9,10]
    esc = [0.06, 0.76, 2.13, 3.85, 5.76, 7.78, 9.75, 11.74, 13.97, 15.59]
    strike = [1.13, 4.37, 8.02, 11.67, 15.14, 18.46, 21.61, 24.59, 27.4, 30.6]
    VA1 = [1.76, 5.58, 9.40, 12.92, 16.14, 19.06, 21.7, 24.15, 26.37, 28.39]
    VA2 = [1.76, 3.18, 4.81, 6.58, 8.41, 10.26, 12.11, 13.94, 15.73, 17.48]
    VA3 = [1.15, 2.84, 4.63, 6.48, 8.41, 10.26, 12.11, 13.42, 15.73, 17.48]
    LEH = [1.13, 2.29, 3.86, 5.64, 7.54, 9.48, 11.42, 13.33, 15.21, 17.04]
    
    
    plt.plot(maturity, esc, 'o', color='sandybrown', label='escrowed')
    plt.plot(maturity, strike, 'o', color= 'palevioletred', label='adjusted strike')
    plt.plot(maturity, VA1, 'o', color= 'gold', label='VA1')
    plt.plot(maturity, VA2, 'o', color= 'yellowgreen', label='VA2')
    plt.plot(maturity, VA3, 'o', color= 'olivedrab', label='VA3')
    plt.plot(maturity, LEH, 'o', color= 'darkslateblue', label='LEH')
    
    
    plt.xlabel('Maturity')
    plt.ylabel('Option Value')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    plt.show()
        
    
