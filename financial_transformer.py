import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class FinancialTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1, target_len=5):
        super(FinancialTransformer, self).__init__()
        
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Capa de predicción para generar múltiples pasos futuros
        self.prediction_head = nn.Linear(d_model, target_len)
        
    def forward(self, src, src_mask=None):
        # src shape: [batch_size, seq_len, input_dim]
        
        # Proyectar entradas a dimensión del modelo
        src = self.input_embedding(src)  # [batch_size, seq_len, d_model]
        
        # Añadir codificación posicional
        src = self.pos_encoder(src)
        
        # Pasar por el encoder transformer
        output = self.transformer_encoder(src, src_mask)  # [batch_size, seq_len, d_model]
        
        # Usar la última salida de secuencia para predicción
        final_encoding = output[:, -1, :]  # [batch_size, d_model]
        
        # Generar predicciones multi-horizonte
        predictions = self.prediction_head(final_encoding)  # [batch_size, target_len]
        
        return predictions