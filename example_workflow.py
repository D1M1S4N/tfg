import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch

# Importar módulos personalizados
from data_preparation import preprocess_financial_data
from dataset import FinancialTimeSeriesDataset
from financial_transformer import FinancialTransformer
from train_eval import train_financial_transformer, evaluate_predictions
from visualization import plot_predictions, plot_multi_step_forecasts
from baseline_models import compare_models, visualize_model_comparison

# Cargar datos de MT5 (suponiendo que ya tienes un DataFrame)
# df = pd.read_csv('data_from_mt5.csv')
# df = preprocess_financial_data(df)

# Simulación de datos para ejemplo (reemplazar con tus datos reales de MT5)
def simulate_financial_data(n_samples=1000):
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    price = 100 + np.cumsum(np.random.normal(0, 1, n_samples)) * 0.1
    volume = np.random.lognormal(10, 1, n_samples)
    
    df = pd.DataFrame({
        'time': dates,
        'open': price + np.random.normal(0, 0.5, n_samples),
        'high': price + np.random.uniform(0.5, 1.5, n_samples),
        'low': price - np.random.uniform(0.5, 1.5, n_samples),
        'close': price + np.random.normal(0, 0.5, n_samples),
        'volume': volume
    })
    
    df.set_index('time', inplace=True)
    return df

# Generar o cargar datos
df = simulate_financial_data(1500)  # Reemplazar con tus datos reales
df = preprocess_financial_data(df)

# Definir características a utilizar
features = ['close_norm', 'open_norm', 'high_norm', 'low_norm', 'volume', 'returns', 'sma_20', 'sma_50', 'rsi']
target_column = 'close_norm'

# Parámetros
seq_length = 60  # 60 días de historia
target_horizon = 5  # Predecir 5 días hacia adelante
batch_size = 32

# Preparar dataset
dataset = FinancialTimeSeriesDataset(
    df.dropna(), 
    seq_length=seq_length, 
    target_horizon=target_horizon, 
    features=features
)

# Dividir en train, validation y test
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size]
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

# Inicializar modelo
input_dim = len(features)
model = FinancialTransformer(
    input_dim=input_dim,
    d_model=128,
    nhead=8,
    num_layers=4,
    dim_feedforward=512,
    dropout=0.1,
    target_len=target_horizon
)

# Entrenar modelo
train_losses, val_losses = train_financial_transformer(
    model, train_loader, val_loader, epochs=50, lr=0.0005
)

# Visualizar curvas de aprendizaje
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Learning Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Evaluar modelo
metrics = evaluate_predictions(model, test_loader)
print("Métricas de evaluación:")
for metric_name, value in metrics.items():
    print(f"{metric_name}: {value:.6f}")

# Generar predicciones para visualización
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.eval()

# Tomar algunas muestras del conjunto de prueba
sample_X, sample_y = next(iter(test_loader))
sample_X = sample_X.to(device)

with torch.no_grad():
    predictions = model(sample_X).cpu().numpy()
    
# Visualizar predicciones multi-horizonte
plot_multi_step_forecasts(
    sample_y.numpy(), 
    predictions, 
    asset_name="EUR/USD", 
    horizon=target_horizon
)

# Comparar con modelos baseline
# Preparar datos para comparación
X_train, y_train = [], []
X_val, y_val = [], []
X_test, y_test = [], []

for X, y in train_loader:
    X_train.append(X.numpy())
    y_train.append(y.numpy())
    
for X, y in val_loader:
    X_val.append(X.numpy())
    y_val.append(y.numpy())
    
for X, y in test_loader:
    X_test.append(X.numpy())
    y_test.append(y.numpy())

X_train = np.vstack(X_train)
y_train = np.vstack(y_train)
X_val = np.vstack(X_val)
y_val = np.vstack(y_val)
X_test = np.vstack(X_test)
y_test = np.vstack(y_test)

# Comparar modelos
results = compare_models(
    X_train, y_train, 
    X_val, y_val,
    X_test, y_test, 
    model, 
    target_horizon=target_horizon
)

# Visualizar comparación
visualize_model_comparison(results)