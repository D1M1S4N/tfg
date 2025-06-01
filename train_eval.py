import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def train_financial_transformer(model, train_loader, val_loader, epochs=50, lr=0.0005):
    """Entrenar modelo transformer para datos financieros"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Criterio y optimizador
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
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
            
            train_loss += loss.item()
            
        # Validación
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = model(X_val)
                val_loss += criterion(outputs, y_val).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    return train_losses, val_losses

def evaluate_predictions(model, test_loader, scaler=None):
    """Evaluar modelo con métricas financieras y de error"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test = X_test.to(device)
            outputs = model(X_test).cpu().numpy()
            
            all_preds.append(outputs)
            all_targets.append(y_test.numpy())
    
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # Desnormalizar si es necesario
    if scaler:
        all_preds = scaler.inverse_transform(all_preds)
        all_targets = scaler.inverse_transform(all_targets)
    
    # Métricas de error
    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_preds)
    
    # Métricas financieras
    direction_accuracy = np.mean(np.sign(all_preds[:, 1:] - all_preds[:, :-1]) == 
                               np.sign(all_targets[:, 1:] - all_targets[:, :-1]))
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'direction_accuracy': direction_accuracy
    }