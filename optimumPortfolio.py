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
#pd.options.display.float_format = '{:.2f}'.format


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


def minimizeVolatility(mu, cov, constraintSet=(0,1)):
    """Minimize the portfolio volatility by altering the 
     weights/allocation of assets in the portfolio"""
    numAssets = len(mu)
    args = (mu, cov)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    result = sc.minimize(portfolioVolatility, numAssets*[1./numAssets], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def maxSharpeRatio( mu, cov,   riskFreeRate=0, constraintSet=(0,1)):
    """
        this function is a optimizing function designed to find the miminum of a function:
        in this instance it looks at a distribution of negative sharpe ratio and find the lowest value
        in other words the maximimum sharpe ratio
    """

    num_assets = len(mu)
    args = (mu, cov, riskFreeRate)
    constraints = ({'type':'eq','fun': lambda x:np.sum(x)-1})
    bound = constraintSet
    bounds = tuple( bound for asset in range(num_assets))
    result = sc.minimize(getNegativeSharpeRatio, num_assets*[1./num_assets], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def getAllocations (mu, cov,  riskFreeRate=0, numOfDaystoAnnualize=365, constraintSet=(0,1)):
    """ function to output allocation for either minimum volatility portfolio 
        or maximum sharpe ratio portfolio
    """
    import numpy as np
    np.set_printoptions(suppress=True)

    #get minimum volatility portfoltio
    minVolOpt = minimizeVolatility(mu,cov)
    minVol_returns, minVol_std = portfolio_annualised_performance(minVolOpt['x'],mu,cov)
    # minVol_returns, minVol_std = round(minVol_returns*100,2), round(minVol_std*100,2)
    minVol_allocation = pd.DataFrame( minVolOpt['x'], index = mu.index, columns=['allocation'])
    minVol_allocation.allocation = [ round(item*100,3) for item in minVol_allocation.allocation]

    #get maximum sharpe ratio portfolio
    maxSRopt = maxSharpeRatio(mu,cov,riskFreeRate)
    maxSharpeRatio_returns, maxSharpeRatio_std = portfolio_annualised_performance(maxSRopt['x'],mu,cov)
    # maxSharpeRatio_returns, maxSharpeRatio_std = round(maxSharpeRatio_returns*100,2), round(maxSharpeRatio_std*100,2)
    maxSR_allocation = pd.DataFrame( maxSRopt['x'], index = mu.index, columns=['allocation'])
    maxSR_allocation.allocation =[ round(item*100,3) for item in maxSR_allocation.allocation ]

    return {
        'Symbols':mu.index.tolist(),
        'MinVolWeights' : list(minVolOpt['x']), 'MinVolReturns' : minVol_returns, 
        'MinVolStd' : minVol_std, 'MinVolAllocation': minVol_allocation,
        'MaxSharpeWeights' :list( maxSRopt['x']), 'MaxSharpeReturns' : maxSharpeRatio_returns, 'MaxSharpeStd' : maxSharpeRatio_std,
        'MaxSharpeAllocation' : maxSR_allocation
    }

def getEfficientFrontierList( mu, cov, allocationdict, riskFreeRate=0, constraintSet=(0,1), linspaceStep=30):
    """
        this function to produce the optimum Sharpe Ratio portfolio, miminum volatility portfolio and efficient frontier
    """
   
    #find efficient frontier using the numpy array with evenly spaced  random sample of
    # numbers between minimum volatility and maximum share ratio returns
    #and insert into efficientList

    efficientPortfolioList =[]
    effcient_weights =[]
    targetReturns = np.linspace(allocationdict['MinVolReturns'], allocationdict['MaxSharpeReturns'],linspaceStep)
    

    for target in targetReturns:
        efficientPortfolioList.append(getEfficientPortfolio(mu, cov, target)['fun'])
        w=list(getEfficientPortfolio(mu, cov, target)['x'])
        w=[round(item*100,3) for item in w]
        effcient_weights.append(w)
    
    e = {'EfficientVolatility': efficientPortfolioList,
         'TargetReturns' : list(targetReturns), 
         'EfficientWeights': list(effcient_weights),
         'Symbols':allocationdict['Symbols'] }

    df = pd.DataFrame([dict(zip(e['Symbols'],w)) for w in e['EfficientWeights']]).join(pd.DataFrame({'Returns':e['TargetReturns'],'Volatility':e['EfficientVolatility']}))

    df['SharpeRatio'] = (df.Returns - riskFreeRate) / df.Volatility

  
    return df



def getPortfolioScenarios(df, mu , cov, num_assets, num_portfolios, RiskFreeRate=0,days=365):
    """
        This function generates a dataframe of multiple portfolios
    """
    #get empty list of portfolio of returns , volatility and weights
    port_returns=[]
    port_volatility=[]
    port_weights=[]

    #compute the portfolios
    print('/*******************************************************************')
    print('Compute mean returns and volatility for multiple portfolio scenarios')
    print('*******************************************************************/')

    np.random.seed(75)
    for port in tqdm(range(num_portfolios)):
        weights=np.random.random(num_assets)
        weights=weights/np.sum(weights)
        port_weights.append(weights)
        returns= np.dot(weights,mu)
        port_returns.append(returns)

        var=cov.mul(weights,axis=0).mul(weights,axis=1).sum().sum()
        sd=np.sqrt(var)*np.sqrt(days)
        
        port_volatility.append(sd)

    #create dictionary to store all the retuns and their volatility from all the scenarios
    data ={'returns':port_returns,'Volatility':port_volatility}
    for counter,ticker in enumerate(df.columns.to_list()):
        data[ticker+' weight'] = [w[counter] for w in port_weights]

    #create data frame from dictionnary
    portfolio=pd.DataFrame(data)
    portfolio['SharpeRatio']=portfolio.returns-RiskFreeRate/portfolio.Volatility
    
    

    return portfolio 