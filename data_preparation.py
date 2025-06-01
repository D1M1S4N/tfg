import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
import pandas_ta

def preprocess_financial_data(df):
    """
    Preprocesar datos financieros para su uso con transformers
    """
    # Asegurar que el índice es datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.set_index('time', inplace=True)
        
    # Calcular características adicionales
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close']).diff()
    
    # Crear características técnicas básicas
    df['sma_20'] = df['close'].rolling(window=21).mean()
    df['sma_50'] = df['close'].rolling(window=55).mean()
    df['rsi'] = pandas_ta.rsi(df['close'], length=13)
    
    # Normalizar datos
    for col in df.columns:
        if col not in ['time', 'volume']:
            df[f'{col}_norm'] = (df[col] - df[col].mean()) / df[col].std()
    
    return df

def test_stationarity(df, column='log_returns'):
    """Comprobar estacionariedad de las series temporales"""
    result = adfuller(df[column].dropna())
    print(f'Prueba ADF para {column}:')
    print(f'Estadístico ADF: {result[0]}')
    print(f'p-value: {result[1]}')
    
    return result[1] < 0.05  # True si es estacionaria