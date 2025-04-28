import matplotlib.pyplot as plt 
import mplfinance as mpf
import pandas as pd 
from typing import List, Dict

def plot_candles(stock:str, df:pd.DataFrame, title:str = None, save:bool = False, show:bool = True) -> None:
    mpf.plot(df, type='candle', style='charles', title=title)