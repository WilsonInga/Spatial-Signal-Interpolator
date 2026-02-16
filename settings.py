"""
Archivo de configuración del proyecto de interpolación de señales espaciales
Contiene parámetros de rutas, hiperparámetros y configuraciones de entrenamiento
"""

import os
from dataclasses import dataclass
from typing import Tuple

@dataclass
class PathConfig:
    """Configuración de rutas del proyecto"""
    raw_data_dir: str = "datos_originales"
    augmented_data_dir: str = "datos_aumentados"
    training_data_dir: str = "datos_entrenamiento"
    models_dir: str = "modelos_guardados"
    results_dir: str = "resultados"
    
    def create_directories(self):
        """Crea los directorios necesarios si no existen"""
        for directory in [self.models_dir, self.results_dir]:
            os.makedirs(directory, exist_ok=True)


@dataclass
class ModelConfig:
    """Configuración del modelo y entrenamiento"""
    # Arquitectura
    input_dimensions: int = 3
    hidden_layer_1: int = 256
    hidden_layer_2: int = 512
    hidden_layer_3: int = 256
    output_length: int = 250
    
    # Regularización
    dropout_rate: float = 0.3
    use_batch_norm: bool = True
    
    # Entrenamiento
    learning_rate: float = 0.0005
    weight_decay: float = 5e-5
    batch_size: int = 32
    num_epochs: int = 100
    train_split: float = 0.85
    
    # Early stopping
    patience: int = 15
    min_delta: float = 1e-6
    
    # Guardado
    checkpoint_name: str = "mejor_modelo_interpolador.pth"
    
    @property
    def validation_split(self) -> float:
        return 1.0 - self.train_split


@dataclass
class DataConfig:
    """Configuración de procesamiento de datos"""
    signal_length: int = 250
    interpolation_steps: int = 10
    coordinate_range: Tuple[float, float] = (-1.0, 1.0)
    normalize_signals: bool = True
    

# Instancias globales de configuración
paths = PathConfig()
model_params = ModelConfig()
data_params = DataConfig()
