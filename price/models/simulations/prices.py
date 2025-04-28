'''
Price Simulation Models. 
Calibration of Jump Diffusion, and Mean reversion Simulation models 

'''

import numpy as np 
import pandas as pd 
import sqlite3 as sql 
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.stats import norm
from scipy.optimize import minimize

@dataclass
class SimulationParams:
    """Dataclass to hold simulation parameters."""
    S0: float  # Initial stock price
    r: float = 0.0375  # Risk-free rate
    days: int = 10  # Number of days to simulate
    mu: float = 0.0  # Drift (expected return)
    sigma: float = 0.2  # Volatility
    number_of_sims: int = 1000  # Number of simulations
    N: int = 100  # Number of time steps (for GBM)
    lam: float = 0.005  # Jump intensity (for jump models)
    mj: float = 0.01  # Mean jump size (for Poisson jump)
    sj: float = 0.01  # Jump size volatility (for Poisson jump)
    m: float = 0.02  # Mean of log-jump size (for Merton jump)
    v: float = 0.003  # Variance of log-jump size (for Merton jump)
    kappa: float = 2.0  # Mean reversion speed (for Heston)
    theta: float = 0.04  # Long-term variance (for Heston)
    v_0: float = 0.04  # Initial variance (for Heston)
    rho: float = -0.7  # Correlation between Brownian motions (for Heston)
    xi: float = 0.1  # Volatility of volatility (for Heston)

def gbm(params: SimulationParams):
    """
    Geometric Brownian Motion simulation.

    :param params: SimulationParams object containing S0, r, days, mu, sigma, number_of_sims, N
    :return: Simulated stock price paths as numpy array
    """
    S0 = params.S0
    mu = params.mu
    sigma = params.sigma
    number_of_sims = params.number_of_sims
    N = params.N
    T = params.days / 252  # Total time in years
    dt = T / N  # Time step

    # Simulate GBM paths
    St = np.exp(
        (mu - sigma ** 2 / 2) * dt
        + sigma * np.random.normal(0, np.sqrt(dt), size=(number_of_sims, N)).T
    )
    St = np.vstack([np.ones(number_of_sims), St])
    St = S0 * St.cumprod(axis=0)
    return St

def poisson_jump(params: SimulationParams):
    """
    Poisson Jump Diffusion simulation.

    :param params: SimulationParams object containing S0, r, days, mu, sigma, number_of_sims, lam, mj, sj
    :return: Simulated stock price paths as numpy array
    """
    S0 = params.S0
    r = params.r
    days = params.days
    sigma = params.sigma
    number_of_sims = params.number_of_sims
    lam = params.lam
    mj = params.mj
    sj = params.sj

    dt = 1 / 252  # Daily time step
    S = np.zeros((days + 1, number_of_sims))
    S[0] = S0
    for t in range(1, days + 1):
        Z = np.random.normal(size=number_of_sims)
        S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
        jumps = np.random.poisson(lam * dt, number_of_sims)
        jump_sizes = np.random.normal(mj, sj, number_of_sims)
        S[t] = S[t] * (1 + jumps * jump_sizes)
    return S

def merton_jump(params: SimulationParams):
    """
    Merton Jump Diffusion simulation.

    :param params: SimulationParams object containing S0, r, days, mu, sigma, number_of_sims, lam, m, v
    :return: Simulated stock price paths as numpy array
    """
    S0 = params.S0
    r = params.r
    days = params.days
    sigma = params.sigma
    number_of_sims = params.number_of_sims
    lam = params.lam
    m = params.m
    v = params.v

    dt = 1 / 252  # Daily time step
    size = (days, number_of_sims)
    poi_rv = np.multiply(np.random.poisson(lam * dt, size=size),
                         np.random.normal(m, v, size=size)).cumsum(axis=0)
    geo = np.cumsum(((r - sigma ** 2 / 2 - lam * (m + v ** 2 / 2)) * dt
                     + sigma * np.sqrt(dt) * np.random.normal(size=size)), axis=0)
    S = np.exp(geo + poi_rv) * S0
    S = np.vstack([np.ones(number_of_sims) * S0, S])
    return S

def heston_path(params: SimulationParams):
    """
    Heston Stochastic Volatility Model simulation.

    :param params: SimulationParams object containing S0, r, days, mu, sigma, number_of_sims, kappa, theta, v_0, rho, xi
    :return: Simulated stock price paths as numpy array
    """
    S0 = params.S0
    r = params.r
    days = params.days
    number_of_sims = params.number_of_sims
    kappa = params.kappa
    theta = params.theta
    v_0 = params.v_0
    rho = params.rho
    xi = params.xi

    dt = 1 / 252  # Daily time step
    size = (number_of_sims, days)
    prices = np.zeros((days + 1, number_of_sims))
    sigs = np.zeros((days + 1, number_of_sims))
    prices[0] = S0
    sigs[0] = v_0
    cov_mat = np.array([[1, rho], [rho, 1]])

    for t in range(1, days + 1):
        WT = np.random.multivariate_normal(np.array([0, 0]), cov=cov_mat, size=number_of_sims) * np.sqrt(dt)
        prices[t] = prices[t - 1] * np.exp((r - 0.5 * sigs[t - 1]) * dt + np.sqrt(sigs[t - 1]) * WT[:, 0])
        sigs[t] = np.abs(sigs[t - 1] + kappa * (theta - sigs[t - 1]) * dt + xi * np.sqrt(sigs[t - 1]) * WT[:, 1])

    return prices

def est_vol(df, lookback=10):
    """
    Yang-Zhang volatility estimator.

    :param df: DataFrame with OHLC data
    :param lookback: Lookback period for estimation
    :return: Estimated annualized volatility as pandas Series
    """

    df.columns = df.columns.str.capitalize()
    o = df.Open
    h = df.High
    l = df.Low
    c = df.Close


    k = 0.34 / (1.34 + (lookback + 1) / (lookback - 1))
    cc = np.log(c / c.shift(1))
    ho = np.log(h / o)
    lo = np.log(l / o)
    co = np.log(c / o)
    oc = np.log(o / c.shift(1))
    oc_sq = oc ** 2
    cc_sq = cc ** 2
    rs = ho * (ho - co) + lo * (lo - co)
    close_vol = cc_sq.rolling(window=lookback).sum() * (1.0 / (lookback - 1.0))
    open_vol = oc_sq.rolling(window=lookback).sum() * (1.0 / (lookback - 1.0))
    window_rs = rs.rolling(window=lookback).sum() * (1.0 / (lookback - 1.0))
    result = (open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * np.sqrt(252)
    result[:lookback - 1] = np.nan
    return result

def heston_calibration(df, params: SimulationParams):
    """
    Calibrate Heston model parameters to historical data.

    :param df: Historical stock price data as pandas DataFrame
    :param params: SimulationParams object
    :return: Optimized Heston parameters as numpy array
    """
    def heston_error(opt_params, df, params):
        df.columns = df.columns.str.capitalize()
        kappa, theta, v_0, rho, xi = opt_params
        params.kappa = kappa
        params.theta = theta
        params.v_0 = v_0
        params.rho = rho
        params.xi = xi
        S_sim = heston_path(params)
        log_returns_sim = np.log(S_sim[1:] / S_sim[:-1])
        mean_sim = np.mean(log_returns_sim)
        var_sim = np.var(log_returns_sim)
        log_returns_hist = np.log(df.Close / df.Close.shift(1)).dropna()
        mean_hist = np.mean(log_returns_hist)
        var_hist = np.var(log_returns_hist)
        error = (mean_sim - mean_hist) ** 2 + (var_sim - var_hist) ** 2
        return error

    initial_params = [params.kappa, params.theta, params.v_0, params.rho, params.xi]
    result = minimize(heston_error, initial_params, args=(df, params), method='Nelder-Mead')
    return result.x

def simulate_stock(stock, df, method='gbm', **kwargs):
    """
    Simulate stock prices using the specified method.

    :param stock: Stock ticker as string
    :param df: Historical stock data as pandas DataFrame
    :param method: Simulation method ('gbm', 'poisson_jump', 'merton_jump', 'heston_path')
    :param kwargs: Additional parameters for SimulationParams
    :return: Simulated stock price paths as numpy array
    """
    # Estimate parameters from historical data]
    df.columns = df.columns.str.capitalize()
    S0 = df.Close.iloc[-1]
    sigma = est_vol(df).iloc[-1]
    mu = df.Close.pct_change().mean()

    # Create SimulationParams object
    params = SimulationParams(S0=S0, mu=mu, sigma=sigma, **kwargs)

    methods = {
        'gbm': gbm,
        'poisson_jump': poisson_jump,
        'merton_jump': merton_jump,
        'heston_path': heston_path
    }

    if method not in methods:
        raise ValueError(f"Invalid method: {method}")

    if method == 'heston_path':
        # Calibrate Heston parameters
        optimized_params = heston_calibration(df, params)
        params.kappa, params.theta, params.v_0, params.rho, params.xi = optimized_params
        print(f"Calibrated Heston parameters: kappa={params.kappa:.4f}, theta={params.theta:.4f}, v_0={params.v_0:.4f}, rho={params.rho:.4f}, xi={params.xi:.4f}")

    # Simulate using the selected method
    S = methods[method](params)
    return S


def set_up_params(df: pd.DataFrame, stock: str, **kwargs):
    """
    Set up simulation parameters based on historical data.

    :param df: DataFrame with historical stock data
    :param stock: Stock ticker
    :param kwargs: Additional parameters for SimulationParams
    :return: SimulationParams object
    """
    S0 = df.Close.iloc[-1]
    sigma = est_vol(df).iloc[-1]
    mu = df.Close.pct_change().mean()

    # Create SimulationParams object
    params = SimulationParams(S0=S0, mu=mu, sigma=sigma, **kwargs)
    return params

def optimize_and_simulate(stock, df, method='heston_path', verbose = True, **kwargs):
    """
    Optimize parameters and simulate stock prices.

    :param stock: Stock ticker as string
    :param df: Historical stock data as pandas DataFrame
    :param method: Simulation method ('gbm', 'poisson_jump', 'merton_jump', 'heston_path')
    :param kwargs: Additional parameters for SimulationParams
    :return: Simulated stock price paths as numpy array
    """
    # Estimate parameters from historical data
    df.columns = df.columns.str.lower()
    S0 = df.close.iloc[-1]
    sigma = df.close.diff().std() * np.sqrt(252)
    mu = df.close.diff().mean()

    # Create SimulationParams object
    params = SimulationParams(S0=S0, mu=mu, sigma=sigma, **kwargs)

    if method == 'heston_path':
        # Calibrate Heston parameters
        optimized_params = heston_calibration(df, params)
        params.kappa, params.theta, params.v_0, params.rho, params.xi = optimized_params
        if verbose:
            print(f"""
                    \nCalibrated Heston parameters for {stock} Days: {params.days}:\n
                    (Mean reversion speed) kappa={params.kappa:.4f}, 
                    (Long term variance) theta={params.theta:.4f}
                    (Initial Variance)v_0={params.v_0:.4f}, 
                    (Correlation between W_t) rho={params.rho:.4f}, 
                    (Volatility of Volatility) xi={params.xi:.4f}
                    """)    

    # Simulate stock prices
    simulated_prices = simulate_stock(stock, df, method=method, **kwargs)
    return simulated_prices

    
if __name__ == "__main__":
    # Example usage (uncomment and replace with actual data)
    # df = pd.read_csv('stock_data.csv')  # Historical data with OHLC columns
    # S_sim = simulate_stock('AAPL', df, method='heston_path', days=30, number_of_sims=500)    
    import sys 
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[4])) 
    from main import Pipeline
    
    p = Pipeline()
    price_db = p.Pricedb.daily_db
    stocks = ['spy','qqq','iwm', 'dia']
    # for stock in stocks:plot_paths(stock, price_db)

    df = pd.read_sql(f"SELECT * FROM {stocks[0]}", price_db, index_col = 'Date', parse_dates = ['Date'])
    S_sim = simulate_stock(stocks[0], df, method='heston_path', days=100, number_of_sims=500)
    print(df.shape, S_sim.shape)

    df = p.Pricedb.get_multi_frame('amzn', ma = 'sma', start_date = "2024-01-01")
    x = optimize_and_simulate(
                stock = 'amzn', 
                df = df, 
                method = 'heston_path',
                verbose = True
                )
    print(df.shape, x.shape)

    

    
    