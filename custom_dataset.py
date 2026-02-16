"""
Dataset personalizado para manejar datos de señales espaciales
Implementa transformaciones y normalización
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, Optional


class SpatialSignalDataset(Dataset):
    """
    Dataset para señales capturadas en posiciones espaciales
    
    Maneja normalización, transformaciones y acceso eficiente a datos
    """
    
    def __init__(
        self,
        coordinates: torch.Tensor,
        signals: torch.Tensor,
        normalization_factor: Optional[float] = None,
        apply_normalization: bool = True
    ):
        """
        Inicializa el dataset
        
        Args:
            coordinates: Tensor (N, 3) con posiciones espaciales
            signals: Tensor (N, signal_length) con señales
            normalization_factor: Factor de normalización (se calcula si es None)
            apply_normalization: Si aplicar normalización
        """
        assert len(coordinates) == len(signals), \
            "Coordenadas y señales deben tener el mismo tamaño"
        
        self.coordinates = coordinates
        self.signals = signals
        self.apply_normalization = apply_normalization
        
        # Calcular o usar factor de normalización
        if apply_normalization:
            if normalization_factor is None:
                self.norm_factor = self._calculate_normalization_factor()
            else:
                self.norm_factor = normalization_factor
            
            # Normalizar señales
            self.signals = self.signals / self.norm_factor
        else:
            self.norm_factor = 1.0
    
    def _calculate_normalization_factor(self) -> float:
        """
        Calcula el factor de normalización (máximo valor absoluto)
        
        Returns:
            Factor de normalización
        """
        return float(torch.max(torch.abs(self.signals)))
    
    def __len__(self) -> int:
        """Retorna el número de muestras"""
        return len(self.coordinates)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Obtiene una muestra del dataset
        
        Args:
            idx: Índice de la muestra
            
        Returns:
            Tupla (coordenadas, señal)
        """
        return self.coordinates[idx], self.signals[idx]
    
    def get_normalization_factor(self) -> float:
        """Retorna el factor de normalización usado"""
        return self.norm_factor
    
    def denormalize_signal(self, normalized_signal: torch.Tensor) -> torch.Tensor:
        """
        Desnormaliza una señal
        
        Args:
            normalized_signal: Señal normalizada
            
        Returns:
            Señal en escala original
        """
        return normalized_signal * self.norm_factor
    
    def get_coordinate_range(self) -> dict:
        """
        Obtiene el rango de coordenadas en el dataset
        
        Returns:
            Diccionario con rangos min/max por dimensión
        """
        coords_np = self.coordinates.cpu().numpy()
        
        return {
            'x': (coords_np[:, 0].min(), coords_np[:, 0].max()),
            'y': (coords_np[:, 1].min(), coords_np[:, 1].max()),
            'z': (coords_np[:, 2].min(), coords_np[:, 2].max())
        }
    
    def get_statistics(self) -> dict:
        """
        Calcula estadísticas del dataset
        
        Returns:
            Diccionario con estadísticas
        """
        return {
            'num_samples': len(self),
            'signal_length': self.signals.shape[1],
            'normalization_factor': self.norm_factor,
            'mean_signal_amplitude': float(torch.mean(self.signals)),
            'std_signal_amplitude': float(torch.std(self.signals)),
            'coordinate_ranges': self.get_coordinate_range()
        }


def prepare_dataset_from_dict(
    data_dict: dict,
    signal_length: int = 250,
    device: str = 'cpu',
    apply_normalization: bool = True
) -> SpatialSignalDataset:
    """
    Prepara un dataset a partir de un diccionario de datos
    
    Args:
        data_dict: Diccionario {(x,y,z): señal}
        signal_length: Longitud de señal a usar
        device: Dispositivo para tensores ('cpu' o 'cuda')
        apply_normalization: Si normalizar señales
        
    Returns:
        Dataset preparado
    """
    coords_list = []
    signals_list = []
    
    for (x, y, z), signal in data_dict.items():
        coords_list.append([x, y, z])
        # Truncar señal a la longitud especificada
        signals_list.append(signal[:signal_length])
    
    # Convertir a tensores
    coordinates_tensor = torch.tensor(
    np.array(coords_list), # Convertimos a NumPy primero
    dtype=torch.float32,
    device=device
    )

    signals_tensor = torch.tensor(
    np.array(signals_list), # Convertimos a NumPy primero
    dtype=torch.float32,
    device=device
    )
    
    # Crear dataset
    dataset = SpatialSignalDataset(
        coordinates=coordinates_tensor,
        signals=signals_tensor,
        apply_normalization=apply_normalization
    )
    
    print(f"\n✓ Dataset creado con {len(dataset)} muestras")
    print(f"  - Dimensión de coordenadas: {coordinates_tensor.shape}")
    print(f"  - Dimensión de señales: {signals_tensor.shape}")
    
    if apply_normalization:
        print(f"  - Factor de normalización: {dataset.get_normalization_factor():.6f}")
    
    return dataset
