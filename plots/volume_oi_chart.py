import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from .utils import human_format


def price_volume_oi_chart(df, pdf, fig, ax, stock = None):
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
    pdf = pdf.loc[df.index]

    ax = ax.flatten()

    ax[0].plot(pdf.index, pdf.close, label = 'Close', color = 'blue')
    # Scatter plot high volume days
    high_option_vol_days = df[df.total_vol > df.total_vol.quantile(0.95)]
    high_option_vol_days = pdf.loc[high_option_vol_days.index]
    ax[0].scatter(high_option_vol_days.index, high_option_vol_days.close, color = 'red', label = 'High Option Volume', s = 20)

    # Scatter plot high volume days
    high_volume_days = pdf[pdf.volume > pdf.volume.quantile(0.95)]
    high_volume_days = pdf.loc[high_volume_days.index]
    ax[0].scatter(high_volume_days.index, high_volume_days.close, color = 'black', label = 'High Volume', s = 30, marker = '+')

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
