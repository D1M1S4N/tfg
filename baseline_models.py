from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class LSTM_Baseline(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=5):
        super(LSTM_Baseline, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        # Usamos solo la última salida para predicción
        final_out = lstm_out[:, -1, :]
        output = self.fc(final_out)
        
        return output

def train_lstm(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, lr=0.001):
    """Entrenar modelo LSTM baseline"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Convertir datos a tensores
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    # Dataset y DataLoader
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    
    # Criterio y optimizador
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Entrenamiento
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # Validación
        model.eval()
        with torch.no_grad():
            X_val_tensor, y_val_tensor = X_val_tensor.to(device), y_val_tensor.to(device)
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
        
        # Guardar pérdidas
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Actualizar learning rate
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    return model, train_losses, val_losses

def compare_models(X_train, y_train, X_val, y_val, X_test, y_test, transformer_model, target_horizon=5):
    """Comparar transformer con modelos baseline"""
    results = {}
    
    # Preparar datos para modelos sklearn
    X_train_2d = X_train.reshape(X_train.shape[0], -1)  # Aplanar para modelos tradicionales
    X_test_2d = X_test.reshape(X_test.shape[0], -1)
    
    # Modelo lineal
    print("Entrenando modelo lineal...")
    lr_model = LinearRegression()
    lr_model.fit(X_train_2d, y_train)
    lr_preds = lr_model.predict(X_test_2d)
    results['Linear'] = {
        'mse': mean_squared_error(y_test, lr_preds),
        'rmse': np.sqrt(mean_squared_error(y_test, lr_preds)),
        'mae': mean_absolute_error(y_test, lr_preds)
    }
    
    # Random Forest
    print("Entrenando Random Forest...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_2d, y_train)
    rf_preds = rf_model.predict(X_test_2d)
    results['RandomForest'] = {
        'mse': mean_squared_error(y_test, rf_preds),
        'rmse': np.sqrt(mean_squared_error(y_test, rf_preds)),
        'mae': mean_absolute_error(y_test, rf_preds)
    }
    
    # LSTM
    print("Entrenando LSTM...")
    input_dim = X_train.shape[2]
    lstm_model = LSTM_Baseline(input_dim=input_dim, output_dim=target_horizon)
    
    # Entrenar LSTM
    lstm_model, _, _ = train_lstm(
        lstm_model, 
        X_train, y_train, 
        X_val, y_val, 
        epochs=30, 
        batch_size=32
    )
    
    # Evaluar LSTM
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lstm_model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        lstm_preds = lstm_model(X_test_tensor).cpu().numpy()
    
    results['LSTM'] = {
        'mse': mean_squared_error(y_test, lstm_preds),
        'rmse': np.sqrt(mean_squared_error(y_test, lstm_preds)),
        'mae': mean_absolute_error(y_test, lstm_preds)
    }
    
    # Transformer (asumimos que ya está entrenado)
    print("Evaluando Transformer...")
    transformer_model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        transformer_preds = transformer_model(X_test_tensor).cpu().numpy()
    
    results['Transformer'] = {
        'mse': mean_squared_error(y_test, transformer_preds),
        'rmse': np.sqrt(mean_squared_error(y_test, transformer_preds)),
        'mae': mean_absolute_error(y_test, transformer_preds)
    }
    
    # Imprimir resultados
    print("\nResultados de comparación de modelos:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name.upper()}: {value:.6f}")
    
    return results

def visualize_model_comparison(results):
    """Visualizar comparación de modelos en gráfico de barras"""
    import matplotlib.pyplot as plt
    
    models = list(results.keys())
    metrics = ['rmse', 'mae']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(models))
    width = 0.35
    
    # Barras para RMSE
    rmse_values = [results[model]['rmse'] for model in models]
    ax.bar(x - width/2, rmse_values, width, label='RMSE')
    
    # Barras para MAE
    mae_values = [results[model]['mae'] for model in models]
    ax.bar(x + width/2, mae_values, width, label='MAE')
    
    ax.set_ylabel('Error')
    ax.set_title('Comparación de métricas de error por modelo')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    
    plt.tight_layout()
    plt.show()