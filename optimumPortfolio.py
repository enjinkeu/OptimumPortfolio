from lib2to3.pygram import Symbols
from pycoingecko import CoinGeckoAPI
import pandas as pd
from datetime import datetime
import warnings
import numpy as np
from tqdm import tqdm
warnings.filterwarnings('ignore')
import scipy.optimize as sc
np.set_printoptions(suppress=True)
pd.options.display.float_format = '{:.4f}'.format



def getRecentData(ListofTickerNames , numberOFDays):
    """ 
        This functions pulls data through the coingecko API.
        It returns recent token price data as a dataframe with 
        the list of tickernames as list the number of Days of data

    """
    
    #instatiate the coingecko api class object
    cg = CoinGeckoAPI()

    #get a list of coin and metadata
    CoinList = cg.get_coins_list()

    #create a list of coingecko symbols, IDs
    symbolList = [item['symbol'] for item in CoinList]
    # ListofTickerNames = ['Ethereum','Polygon','Bitcoin','Avalanche','Cosmos','Chainlink','Polkadot','The Graph','Solana','Dai']
    
    listOfIDs=[item['id'] for item  in CoinList if item['name'] in ListofTickerNames]

    #create dictionnaries for fast data retrieval
    dct ={item :cg.get_coin_ohlc_by_id(id=item,vs_currency='usd',days=numberOFDays) for item in listOfIDs}
    IDDict={item['id'] :item['name'] for item in CoinList if item['id'] in listOfIDs}
    SymbolDict = {item['id'] :item['symbol'] for item in CoinList if item['id'] in listOfIDs}

    #use list to get 'Timestamp','Price','CoingeckoID','Name','Symbol'
    l=[]
    for item in listOfIDs:
        for i in range(len(dct[item])):
            l.append([dct[item][i][0],dct[item][i][1],dct[item][i][2],dct[item][i][3],dct[item][i][4],item,IDDict[item],SymbolDict[item]])
            

    #use list to create the dataframe to store the price data
    df = pd.DataFrame(l , columns=['Timestamp','Open','High','Low','Close','CoingeckoID','Name','Symbol'])
    df['date']=pd.to_datetime(df['Timestamp']/1000,unit='s')
    df.set_index('date',inplace = True)

    return df



def getHistoricalPriceData(ListofTickerNames, numberOFDays='max'):
    """
        This functions pulls data through the coingecko API.
        It returns historical token price data as a dataframe with 
        the list of tickernames as list the number of Days of data

    """
    #instatiate the coingecko api class object
    cg = CoinGeckoAPI()

    #get a list of coin and metadata
    CoinList = cg.get_coins_list()

    #create a list of coingecko symbols, IDs
    symbolList = [item['symbol'] for item in CoinList]
    # ListofTickerNames = ['Ethereum','Polygon','Bitcoin','Avalanche','Cosmos','Chainlink','Polkadot','The Graph','Solana','Dai']
    
    listOfIDs=[item['id'] for item  in CoinList if item['name'] in ListofTickerNames]

    #create dictionnaries for fast data retrieval
    dct ={item :cg.get_coin_market_chart_by_id(id=item,vs_currency='usd',days=numberOFDays) for item in listOfIDs}
    IDDict={item['id'] :item['name'] for item in CoinList if item['id'] in listOfIDs}
    SymbolDict = {item['id'] :item['symbol'] for item in CoinList if item['id'] in listOfIDs}

    #use list to get 'Timestamp','Price','CoingeckoID','Name','Symbol'
    l=[]
    for item in listOfIDs:
        for i in range(len(dct[item]['prices'])):
            l.append([dct[item]['prices'][i][0],dct[item]['prices'][i][1],item,IDDict[item],SymbolDict[item]])

    #use list to create the dataframe to store the price data
    df = pd.DataFrame(l , columns=['Timestamp','Price','CoingeckoID','Name','Symbol'])
    
    #create the date column and set the index
    df['date']=pd.to_datetime(df['Timestamp']/1000,unit='s')
    df.set_index('date',inplace = True)

    #create the daily returns column
    df['DailyReturns'] = df['Price'].pct_change()
    

    #remove nulls
    df.dropna(inplace = True)
    df.sort_index(inplace = True)
    return df


def getMoments(df):
    """
        This function gets the token price dataframe
        and returns mu (mean) and covariance matrix.
        Since tokens are traded daily, we use the default value of 365
    """

    pivoted_df = df.pivot_table(columns='Symbol',values='DailyReturns',index='date').dropna()
    mu = pivoted_df.mean()
    cov = pivoted_df.cov()

    return mu,cov,pivoted_df



def portfolio_annualised_performance(weights, mu, cov,NumberOfDays=365):
    """ 
        this function aims to calculate portfolio annualized mean return and volatility 
    """
    returns = np.sum(mu * weights ) *NumberOfDays
    std = np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * np.sqrt(NumberOfDays)
    return  returns,std


def portfolioReturn(weights, mu, cov):
    """
        this function aims to return portfolio returns
    """
    return portfolio_annualised_performance(weights, mu, cov,NumberOfDays=365)[0]

def portfolioVolatility(weights, mu, cov):
    """
        this function finds the portfolio volatity of a portfolio
    """
    return portfolio_annualised_performance(weights, mu, cov,NumberOfDays=365)[1]


def getEfficientPortfolio(mu, cov, returnTarget, constraintSet=(0,1)):
    """For each returnTarget, we want to optimize the portfolio for minimum variance"""
    num_assets = len(mu)
    args = (mu, cov)

    constraints = ({'type':'eq', 'fun': lambda x: portfolioReturn(x, mu, cov) - returnTarget},
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(num_assets))
    optimization = sc.minimize(portfolioVolatility, num_assets*[1./num_assets], args=args, method = 'SLSQP', bounds=bounds, constraints=constraints)
    return optimization

def getNegativeSharpeRatio(weights, mu, cov, riskFreeRate = 0):
    """ 
        this funcion helps compute the negative Sharpe Ratios
        in order to use the function scipy.optimize.minimize
        to find the minimum of a function
    """
    Returns, Std = portfolio_annualised_performance(weights, mu, cov,365)
    return - (Returns - riskFreeRate)/Std

