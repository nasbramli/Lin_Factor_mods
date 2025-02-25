import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings("ignore")

#Function 1 - CAPM Market Model: Obtain alpha, beta given Rm-Rf and Ri-Rf
def get_CAPM_loading(excess_market, excess_industry):
    """
    Given excess_market[Rm-Rf] & excess_returns[Ri-Rf] ALL Decimals -> Output alpha, beta
    Equation: Ri - Rf = αi + βi(Rm - Rf) + εi
    """
    MarketModel_table = pd.DataFrame(index=excess_industry.columns)
    
    for i in excess_industry.columns:
        model = LinearRegression()
        model.fit(excess_market.to_numpy().reshape(-1,1) , excess_industry[f"{i}"].to_numpy().reshape(-1,1))
        MarketModel_table.loc[f"{i}","Alpha_CAPM"] = model.intercept_
        MarketModel_table.loc[f"{i}","Beta_MarketRisk_CAPM"] = model.coef_

    return MarketModel_table

#Function 2 - Given X:[Rm-Rf,SMB,HML] & Y:excess_returns --> Give 3 factor loadings
def get_FF3_loading(ff_factors, excess_returns):
    '''
    Given X: factor_df(Rm-Rf, SMB, HML), y: monthly excess_returns for each sector --> Generate 3 factor loadings  
    '''
    factor_loadings = pd.DataFrame(index=[excess_returns.columns], columns=["Alpha_FamaFrench", "Beta_MarketRisk_FamaFrench", "Beta_SMB_FamaFrench", "Beta_HML_FamaFrench"])
    
    for i in excess_returns.columns:
        model = LinearRegression()
        model.fit(ff_factors, excess_returns[f"{i}"])
        
        factor_loadings.loc[f"{i}", "Alpha_FamaFrench"], factor_loadings.loc[f"{i}","Beta_MarketRisk_FamaFrench"], factor_loadings.loc[f"{i}", "Beta_SMB_FamaFrench"], factor_loadings.loc[f"{i}", "Beta_HML_FamaFrench"]= (
            model.intercept_ ,model.coef_[0], model.coef_[1], model.coef_[2]
        )
    return factor_loadings

#Function 3
def sharpe_ratio(excess_returns):
    """
    Given excess_returns [Ri-Rf] --> Output sharpe
    """
    sharpe = (
        excess_returns.mean(axis=0)
        /
        excess_returns.std()
        )
    return sharpe

#Function 4
def sortino_ratio(excess_target):
    """
    Given excess_target[Ri-Rt] --> Output sortino
    """
    adj_excess_target = excess_target.copy() 
    adj_excess_target[adj_excess_target>0] = 0 #>0 discard only take those negative
    semi_deviation = (adj_excess_target**2).mean(axis=0).apply(np.sqrt)
    sortino = (
        excess_target.mean(axis=0)
        /
        semi_deviation       
    )
    return sortino

#Function 5 
def treynor_ratio(excess_returns, CAPM):
    """
    Given excess_returns[Ri-Rf], CAPM df from lfmk.get_CAPM_loading() --> Output treynor ratio
    """
    treynor = (
        excess_returns.mean(axis=0)
        /
        CAPM.Beta_MarketRisk_CAPM.values
    )
    return treynor

#Function 6
def jensens_alpha(excess_returns, excess_market, CAPM):
    jensen = (
        excess_returns.mean(axis=0)
        -
        CAPM.Beta_MarketRisk_CAPM*excess_market.mean()        
    )
    return jensen





