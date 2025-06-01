import torch
from torch.utils.data import Dataset, DataLoader

class FinancialTimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length=60, target_horizon=5, features=None):
        """
        Dataset para series temporales financieras compatible con transformers
        
        Args:
            data (pd.DataFrame): Datos financieros preprocesados
            seq_length (int): Longitud de la secuencia de entrada
            target_horizon (int): Horizonte de predicción
            features (list): Lista de características a utilizar
        """
        self.data = data
        self.seq_length = seq_length
        self.target_horizon = target_horizon
        self.features = features if features is not None else ['close_norm', 'volume', 'rsi']
        
        # Preparar datos en formato secuencial
        self._prepare_sequences()
    
    def _prepare_sequences(self):
        """Preparar secuencias de entrada y objetivos"""
        self.sequences = []
        self.targets = []
        
        for i in range(len(self.data) - self.seq_length - self.target_horizon):
            # Extraer secuencia
            seq = self.data.iloc[i:i+self.seq_length][self.features].values
            
            # Objetivo (puede ser precio futuro, retornos, etc.)
            target = self.data.iloc[i+self.seq_length:i+self.seq_length+self.target_horizon]['close_norm'].values
            
            self.sequences.append(seq)
            self.targets.append(target)
            
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor(self.targets[idx])