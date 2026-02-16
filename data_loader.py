"""
Módulo para cargar y procesar señales de receptores espaciales
Maneja la lectura de archivos, validación de datos y estructuración
"""

import os
import re
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np
from pathlib import Path


class SpatialSignalLoader:
    """
    Clase para cargar señales capturadas por receptores en posiciones espaciales
    """
    
    def __init__(self, base_directory: str):
        """
        Inicializa el cargador de señales
        
        Args:
            base_directory: Ruta al directorio con archivos de datos
        """
        self.base_directory = Path(base_directory)
        self.signals_dict: Dict[Tuple[float, float, float], np.ndarray] = {}
        self._validate_directory()
    
    def _validate_directory(self):
        """Valida que el directorio existe y contiene archivos"""
        if not self.base_directory.exists():
            raise FileNotFoundError(
                f"El directorio {self.base_directory} no existe"
            )
        
        txt_files = list(self.base_directory.glob("*.txt"))
        if not txt_files:
            raise ValueError(
                f"No se encontraron archivos .txt en {self.base_directory}"
            )
    
    def _extract_coordinates(self, filename: str) -> Optional[Tuple[float, float, float]]:
        """
        Extrae las coordenadas espaciales del nombre del archivo
        
        Args:
            filename: Nombre del archivo (e.g., 'DatReceptor_1_0_-1.txt')
            
        Returns:
            Tupla con coordenadas (x, y, z) o None si el formato es inválido
        """
        # Patrón para extraer coordenadas del formato DatReceptor_x_y_z.txt
        pattern = r'DatReceptor_([-\d.]+)_([-\d.]+)_([-\d.]+)\.txt'
        match = re.match(pattern, filename)
        
        if match:
            x, y, z = match.groups()
            return (float(x), float(y), float(z))
        return None
    
    def _read_signal_file(self, filepath: Path) -> np.ndarray:
        """
        Lee un archivo de señal y retorna el vector de amplitudes
        
        Args:
            filepath: Ruta completa al archivo
            
        Returns:
            Array numpy con las amplitudes de la señal
        """
        try:
            # Leer archivo con separadores flexibles
            df = pd.read_csv(
                filepath,
                sep=r'\s+',
                header=None,
                names=['tiempo', 'amplitud'],
                engine='python'
            )
            return df['amplitud'].values
        
        except Exception as e:
            print(f"Error leyendo {filepath.name}: {e}")
            return np.array([])
    
    def load_all_signals(self) -> Dict[Tuple[float, float, float], np.ndarray]:
        """
        Carga todas las señales del directorio
        
        Returns:
            Diccionario con coordenadas como llave y señales como valores
        """
        files_processed = 0
        
        for filepath in self.base_directory.glob("*.txt"):
            coordinates = self._extract_coordinates(filepath.name)
            
            if coordinates is None:
                print(f"Advertencia: formato de nombre incorrecto para {filepath.name}")
                continue
            
            signal = self._read_signal_file(filepath)
            
            if signal.size > 0:
                self.signals_dict[coordinates] = signal
                files_processed += 1
        
        print(f"✓ Cargados {files_processed} archivos de señales")
        return self.signals_dict
    
    def get_signal_statistics(self) -> Dict[str, any]:
        """
        Calcula estadísticas de las señales cargadas
        
        Returns:
            Diccionario con estadísticas generales
        """
        if not self.signals_dict:
            return {}
        
        all_signals = np.array(list(self.signals_dict.values()))
        
        return {
            'num_receptores': len(self.signals_dict),
            'longitud_señal': all_signals.shape[1] if all_signals.ndim > 1 else len(all_signals),
            'amplitud_maxima': np.max(all_signals),
            'amplitud_minima': np.min(all_signals),
            'amplitud_promedio': np.mean(all_signals),
            'desviacion_estandar': np.std(all_signals)
        }
    
    def validate_signal_length(self) -> bool:
        """
        Verifica que todas las señales tengan la misma longitud
        
        Returns:
            True si todas tienen la misma longitud
        """
        if not self.signals_dict:
            return False
        
        lengths = [len(signal) for signal in self.signals_dict.values()]
        return len(set(lengths)) == 1


def load_spatial_data(directory_path: str) -> Dict[Tuple[float, float, float], np.ndarray]:
    """
    Función de conveniencia para cargar datos espaciales
    
    Args:
        directory_path: Ruta al directorio con archivos
        
    Returns:
        Diccionario con señales cargadas
    """
    loader = SpatialSignalLoader(directory_path)
    signals = loader.load_all_signals()
    
    # Mostrar estadísticas
    stats = loader.get_signal_statistics()
    print("\n=== Estadísticas de datos cargados ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    return signals
