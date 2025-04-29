import matplotlib.pyplot as plt
import matplotlib.animation as animation 
import datetime as dt 

def compare_current_yields(df):
    """
    Compare current yields with historical yields.
    """
    pdf = df.copy().sort_index()
    cols = '1d 1m 3m 6m 1y 2y 3y 5y 7y 10y 20y 30y'.split()
    pdf.columns = cols

    today = pdf.index[-1]
    one_month_ago = pdf.loc[dt.datetime.now() - dt.timedelta(days=30):].index[0]
    six_months_ago = pdf.loc[dt.datetime.now() - dt.timedelta(days=180):].index[0]
    one_year_ago = pdf.loc[dt.datetime.now() - dt.timedelta(days=365):].index[0]
    two_years_ago = pdf.loc[dt.datetime.now() - dt.timedelta(days=730):].index[0]


    fig, ax = plt.subplots(figsize=(5.5, 7))
    # pdf.loc[today].plot(ax=ax, label = pdf.loc[today].name.strftime('%-m/%-d/%Y') + ' (today)')
    ax.plot(pdf.columns, pdf.loc[today], marker = 'o', linestyle= '-', label = pdf.loc[today].name.strftime('%-m/%-d/%Y') + ' (today)')
    ax.plot(pdf.columns, pdf.loc[one_month_ago], marker = 'o', linestyle= ':', label = pdf.loc[one_month_ago].name.strftime('%-m/%-d/%Y') + ' (1 month ago)')
    ax.plot(pdf.columns, pdf.loc[six_months_ago], marker = 'o', linestyle= '--', label = pdf.loc[six_months_ago].name.strftime('%-m/%-d/%Y') + ' (6 months ago)')
    ax.plot(pdf.columns, pdf.loc[one_year_ago], marker = 'o', linestyle= '-.', label = pdf.loc[one_year_ago].name.strftime('%-m/%-d/%Y') + ' (1 year ago)')

    ax.legend()
    ax.grid()
    ax.set_title('Yield Curve')
    ax.set_ylabel('Yield (%)')
    ax.set_xlabel('Maturity')
    return fig, ax 


def animate_yield_curve(df):
    fig, ax = plt.subplots(figsize=(5.5, 7))
    pdf = df.copy().sort_index()
    def update(i):
        ax.clear()
        ax.plot(pdf.columns, pdf.iloc[i], marker = 'o', linestyle= '-')
        # Annotate each point with the respective yield
        for j, txt in enumerate(pdf.iloc[i]):
            ax.annotate(f'{txt:.2f}', (pdf.columns[j], pdf.iloc[i][j]), textcoords="offset points", xytext=(0,10), ha='center')

        ax.set_title(pdf.index[i].strftime('%-m/%-d/%Y'))
        ax.set_ylabel('Yield (%)')
        ax.set_xlabel('Maturity')
        ax.set_ylim(0, 7.5)
        ax.grid()

    ani = animation.FuncAnimation(fig, update, frames=len(pdf), interval=1000)
    ani.save('yield_curve_animation.gif', writer='pillow')
    plt.show()
    return ani


if __name__ == '__main__':

    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from main import Manager 
    m = Manager()
    bonds = m.Bonds.bond_df().set_index('DATE')
    fig, ax = compare_current_yields(bonds)
    fig.show()
