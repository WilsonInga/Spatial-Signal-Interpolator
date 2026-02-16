"""
Arquitectura de red neuronal para interpolación de señales espaciales
Implementa una red profunda con regularización avanzada
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialSignalInterpolator(nn.Module):
    """
    Red neuronal profunda para interpolar señales en el espacio 3D
    
    Arquitectura:
    - Entrada: 3 coordenadas (x, y, z)
    - Capas ocultas con BatchNorm y Dropout
    - Salida: 250 puntos de señal temporal
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dims: tuple = (256, 512, 256),
        output_dim: int = 250,
        dropout_prob: float = 0.3,
        use_batch_norm: bool = True
    ):
        """
        Inicializa la arquitectura de la red
        
        Args:
            input_dim: Dimensión de entrada (coordenadas espaciales)
            hidden_dims: Tupla con tamaños de capas ocultas
            output_dim: Dimensión de salida (longitud de señal)
            dropout_prob: Probabilidad de dropout
            use_batch_norm: Si usar batch normalization
        """
        super(SpatialSignalInterpolator, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_batch_norm = use_batch_norm
        
        # Primera capa: input -> hidden[0]
        self.encoder_layer1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0]) if use_batch_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout_prob)
        
        # Segunda capa: hidden[0] -> hidden[1]
        self.encoder_layer2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1]) if use_batch_norm else nn.Identity()
        self.dropout2 = nn.Dropout(dropout_prob)
        
        # Tercera capa: hidden[1] -> hidden[2]
        self.decoder_layer1 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.bn3 = nn.BatchNorm1d(hidden_dims[2]) if use_batch_norm else nn.Identity()
        self.dropout3 = nn.Dropout(dropout_prob)
        
        # Capa de salida: hidden[2] -> output
        self.output_layer = nn.Linear(hidden_dims[2], output_dim)
        
        # Inicialización de pesos usando Xavier
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializa los pesos de las capas lineales con Xavier"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, spatial_coords: torch.Tensor) -> torch.Tensor:
        """
        Propagación hacia adelante
        
        Args:
            spatial_coords: Tensor de forma (batch_size, 3) con coordenadas
            
        Returns:
            Tensor de forma (batch_size, 250) con señales interpoladas
        """
        # Encoder: comprimir información espacial
        x = self.encoder_layer1(spatial_coords)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.encoder_layer2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Decoder: expandir a señal temporal
        x = self.decoder_layer1(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        # Capa de salida (sin activación para regresión)
        signal_output = self.output_layer(x)
        
        return signal_output
    
    def count_parameters(self) -> int:
        """
        Cuenta el número total de parámetros entrenables
        
        Returns:
            Número de parámetros
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_architecture_summary(self) -> str:
        """
        Genera un resumen de la arquitectura
        
        Returns:
            String con descripción de la arquitectura
        """
        summary = []
        summary.append("=" * 60)
        summary.append("ARQUITECTURA DE RED NEURONAL")
        summary.append("=" * 60)
        summary.append(f"Entrada: {self.input_dim} coordenadas espaciales (x, y, z)")
        summary.append(f"Salida: {self.output_dim} puntos de señal temporal")
        summary.append(f"Parámetros totales: {self.count_parameters():,}")
        summary.append(f"Batch Normalization: {'Sí' if self.use_batch_norm else 'No'}")
        summary.append("=" * 60)
        
        return "\n".join(summary)


class EarlyStopping:
    """
    Implementa early stopping para evitar overfitting
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-6, verbose: bool = True):
        """
        Args:
            patience: Épocas a esperar antes de detener
            min_delta: Mínima mejora considerada significativa
            verbose: Si imprimir mensajes
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, validation_loss: float, model: nn.Module) -> bool:
        """
        Verifica si se debe detener el entrenamiento
        
        Args:
            validation_loss: Pérdida en validación actual
            model: Modelo a guardar si mejora
            
        Returns:
            True si se debe detener el entrenamiento
        """
        if self.best_loss is None:
            self.best_loss = validation_loss
            self.best_model_state = model.state_dict()
            return False
        
        # Verificar si hay mejora significativa
        if validation_loss < self.best_loss - self.min_delta:
            self.best_loss = validation_loss
            self.best_model_state = model.state_dict()
            self.counter = 0
            if self.verbose:
                print(f"    → Mejor modelo actualizado (Loss: {validation_loss:.6f})")
        else:
            self.counter += 1
            if self.verbose:
                print(f"    → Sin mejora ({self.counter}/{self.patience})")
            
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False
    
    def load_best_model(self, model: nn.Module):
        """Carga el mejor modelo guardado"""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
