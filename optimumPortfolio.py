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
