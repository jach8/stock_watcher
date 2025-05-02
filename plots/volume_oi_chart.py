import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from .utils import human_format


def price_volume_oi_chart(df, pdf, fig, ax, stock = None, start_date = None, end_date = None):
    """
    Plot the price, volume, and open interest of a stock.
    Parameters:
    df (pd.DataFrame): Dataframe containing the options stats data.
    pdf (pd.DataFrame): Dataframe containing the price data.
    fig (matplotlib.figure.Figure): Figure object to plot on.
    ax (matplotlib.axes.Axes): Axes object to plot on.
    stock (str): Stock symbol.
    """

    # Align the dataframes
    # pdf = pdf.loc[df.index]

    # High Option Volume days
    high_option_vol_days = df[df.total_vol > df.total_vol.quantile(0.95)]
    high_option_vol_days = pdf.loc[high_option_vol_days.index]

    # High Stock volume days
    high_volume_days = pdf[pdf.volume > pdf.volume.quantile(0.95)]
    high_volume_days = pdf.loc[high_volume_days.index]

    # High Call Volume days
    high_call_vol_days = df[df.call_vol > df.call_vol.quantile(0.95)]
    high_call_vol_days = pdf.loc[high_call_vol_days.index]
    
    # High Put Volume days
    high_put_vol_days = df[df.put_vol > df.put_vol.quantile(0.95)]
    high_put_vol_days = pdf.loc[high_put_vol_days.index]
    
     # Check if the index is a datetime index   
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if not isinstance(pdf.index, pd.DatetimeIndex):
        pdf.index = pd.to_datetime(pdf.index)
    
    if start_date is not None:
        df = df.loc[start_date:]
        pdf = pdf.loc[start_date:]
    if end_date is not None:
        df = df.loc[:end_date]
        pdf = pdf.loc[:end_date]

    high_option_vol_days = high_option_vol_days.loc[start_date:end_date]
    high_volume_days = high_volume_days.loc[start_date:end_date]
    high_call_vol_days = high_call_vol_days.loc[start_date:end_date]
    high_put_vol_days = high_put_vol_days.loc[start_date:end_date]
    
    
    ax = ax.flatten()
    ax[0].plot(pdf.index, pdf.close, label = 'Close', color = 'blue')

    try:
        ax[0].scatter(high_option_vol_days.index, high_option_vol_days.close, color = 'red', label = 'Option Volume', s = 20)
        ax[0].scatter(high_volume_days.index, high_volume_days.close, color = 'black', label = 'Stock Volume', s = 30, marker = '+')
        ax[0].scatter(high_call_vol_days.index, high_call_vol_days.close, color = 'green', label = 'Call Volume', s = 50, marker = '^')
        ax[0].scatter(high_put_vol_days.index, high_put_vol_days.close, color = 'red', label = 'Put Volume', s = 50, marker = 'v')
    except:
        pass

    ax[0].legend()
    ax[1].plot(pdf.index, pdf.volume, color = 'green')

    ax[2].plot(df.index, df.total_vol, color = 'orange', alpha = 0.5)
    ax[2].plot(df.total_vol.rolling(20).mean(), color = 'orange', alpha = 0.9)
    ax[2].hlines(y =df.total_vol.quantile(0.95), xmin = df.index[0], xmax = df.index[-1], color = 'blue', linestyle = '--', label = '95% Quantile')
    
    ax[3].plot(df.index, df.total_oi, color = 'red', alpha = 0.5)
    ax[3].plot(df.total_oi.rolling(20).mean(), color = 'red', alpha = 0.9)
    ax[3].hlines(y =df.total_oi.quantile(0.95), xmin = df.index[0], xmax = df.index[-1], color = 'blue', linestyle = '--', label = '95% Quantile')


    if stock is None:
        stock = 'Close'
    titles = [f'${stock.upper()}', 'Volume', 'Options Volume', 'Open Interest']
    for i, title in enumerate(titles):
        if i > 0:
            yticks = ax[i].get_yticks()
            ytick_labels = [human_format(int(tick)) for tick in yticks]
            ax[i].set_yticklabels(ytick_labels)
        ax[i].set_title(title)
        ax[i].grid()

    fig.tight_layout()
    fig.autofmt_xdate(rotation = 0 )
    return ax
