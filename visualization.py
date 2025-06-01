import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def plot_predictions(true_values, predictions, asset_name, start_date=None):
    """Visualizar predicciones vs valores reales"""
    plt.figure(figsize=(14, 7))
    
    # Si tenemos fechas, usarlas como índice
    if start_date:
        index = pd.date_range(start=start_date, periods=len(true_values), freq='D')
    else:
        index = range(len(true_values))
    
    plt.plot(index, true_values, label='Valor real', color='blue')
    plt.plot(index, predictions, label='Predicción', color='red', linestyle='--')
    
    plt.title(f'Predicciones vs Valores Reales - {asset_name}')
    plt.xlabel('Fecha')
    plt.ylabel('Valor normalizado')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if start_date:
        plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plt.show()

def plot_attention_weights(attention_weights, seq_length, feature_names=None):
    """Visualizar pesos de atención del transformer"""
    plt.figure(figsize=(12, 10))
    
    # Promediar los pesos de atención de las diferentes cabezas
    avg_weights = attention_weights.mean(axis=1).squeeze()
    
    # Etiquetas para características
    if feature_names is None:
        feature_names = [f"t-{seq_length-i}" for i in range(seq_length)]
    
    # Crear mapa de calor
    sns.heatmap(avg_weights, annot=False, cmap='viridis', 
                xticklabels=feature_names, yticklabels=feature_names)
    
    plt.title('Pesos de Atención Promedio')
    plt.tight_layout()
    plt.show()

def plot_multi_step_forecasts(true_values, predictions, asset_name, horizon=5):
    """Visualizar predicciones de múltiples horizontes temporales"""
    plt.figure(figsize=(15, 10))
    
    for i in range(min(5, len(true_values))):  # Mostrar solo 5 ejemplos
        plt.subplot(5, 1, i+1)
        
        # Valores reales
        plt.plot(range(horizon), true_values[i], 'b-o', label='Real')
        
        # Predicciones
        plt.plot(range(horizon), predictions[i], 'r--x', label='Predicción')
        
        plt.title(f'Muestra {i+1}')
        plt.ylabel('Valor')
        plt.grid(alpha=0.3)
        
        if i == 0:
            plt.legend()
    
    plt.suptitle(f'Predicciones Multi-Horizonte - {asset_name}')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()