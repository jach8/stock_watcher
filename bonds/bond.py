import sqlite3 as sql
import pandas as pd
import datetime as dt
import os 
import sys
from tqdm import tqdm 
from pandas_datareader import data as web


def get_fred_data(start_date, code, name = None):
    data = web.DataReader(code, 'fred', start_date)
    if name != None:
        data.columns = [name]
    return data

class bonds():
    def __init__(self, connections):
        self.bonds_db = sql.connect(connections['bonds_db'])
    
    def get_bond_data(self):
        start_date = dt.datetime(1900, 1, 1)
        fred_code = ['DFF', 'DGS1MO', 'DGS3MO', 'DGS6MO', 'DGS1', 'DGS2', 'DGS3', 'DGS5', 'DGS7', 'DGS10', 'DGS20', 'DGS30']
        bonds = ['overnight','1-month' ,'3-month', '6-month', '1-year', '2-year', '3-year', '5-year', '7-year', '10-year', '20-year', '30-year']
        bond_data = []

        pbar = tqdm(fred_code, desc = 'Downloading Bond Data')
        for i in pbar:
            bond_data.append(web.DataReader(i, 'fred', start_date))
            pbar.set_description(f'Downloading {bonds[fred_code.index(i)]} data')

        out = pd.concat(bond_data, axis = 1).dropna()
        out.index = pd.to_datetime(out.index)
        cols = '1d 1m 3m 6m 1y 2y 3y 5y 7y 10y 20y 30y'.split()
        out.columns = cols 
        return out
        
    def update_bonds(self):
        us_bonds = self.get_bond_data()
        us_bonds.columns = [x.replace(" ", "") for x in us_bonds.columns]
        us_bonds.to_sql('us_bonds', self.bonds_db, if_exists = 'replace')
        print('Bonds Updated')
    
    def bond_df(self):
        return pd.read_sql('select * from us_bonds', self.bonds_db, parse_dates=['DATE']).sort_index()
    
if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from bin.main import get_path 
    
    c = get_path()
    b = bonds(c)
    b.update_bonds()
    print(b.bond_df())
    