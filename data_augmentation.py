"""
Generador de datos sintéticos mediante interpolación trilineal
Aumenta el dataset original para mejorar el entrenamiento
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from data_loader import SpatialSignalLoader


class TrilinearAugmentor:
    """
    Clase para generar datos sintéticos usando interpolación trilineal
    """
    
    def __init__(
        self,
        original_signals: Dict[Tuple[float, float, float], np.ndarray],
        grid_resolution: int = 10
    ):
        """
        Inicializa el aumentador de datos
        
        Args:
            original_signals: Diccionario con señales originales
            grid_resolution: Número de puntos intermedios por dimensión
        """
        self.original_signals = original_signals
        self.grid_resolution = grid_resolution
        self.synthetic_signals = {}
        self.signal_length = len(next(iter(original_signals.values())))
    
    def _interpolate_point(
        self,
        xd: float, yd: float, zd: float,
        corner_values: np.ndarray
    ) -> float:
        """
        Realiza interpolación trilineal para un punto específico
        
        Args:
            xd, yd, zd: Coordenadas normalizadas [0, 1] dentro del cubo
            corner_values: Array con 8 valores de las esquinas del cubo
            
        Returns:
            Valor interpolado
        """
        c000, c100, c010, c110, c001, c101, c011, c111 = corner_values
        
        # Fórmula de interpolación trilineal
        interpolated = (
            c000 * (1 - xd) * (1 - yd) * (1 - zd) +
            c100 * xd * (1 - yd) * (1 - zd) +
            c010 * (1 - xd) * yd * (1 - zd) +
            c110 * xd * yd * (1 - zd) +
            c001 * (1 - xd) * (1 - yd) * zd +
            c101 * xd * (1 - yd) * zd +
            c011 * (1 - xd) * yd * zd +
            c111 * xd * yd * zd
        )
        
        return interpolated
    
    def _get_cube_corners(self, x0: int, y0: int, z0: int) -> list:
        """
        Obtiene las coordenadas de las 8 esquinas de un cubo
        
        Args:
            x0, y0, z0: Coordenadas de la esquina inferior del cubo
            
        Returns:
            Lista con las 8 esquinas
        """
        x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
        
        return [
            (x0, y0, z0), (x1, y0, z0), (x0, y1, z0), (x1, y1, z0),
            (x0, y0, z1), (x1, y0, z1), (x0, y1, z1), (x1, y1, z1)
        ]
    
    def _cube_exists(self, corners: list) -> bool:
        """Verifica si todas las esquinas del cubo tienen datos"""
        return all(corner in self.original_signals for corner in corners)
    
    def generate_augmented_data(self) -> Dict[Tuple[float, float, float], np.ndarray]:
        """
        Genera datos sintéticos por interpolación
        
        Returns:
            Diccionario con señales sintéticas
        """
        print(f"\n=== Generando datos sintéticos ===")
        print(f"Resolución de grid: {self.grid_resolution} puntos por dimensión")
        
        # Definir cubos unitarios en la malla
        cube_origins = [
            (-1, -1, -1), (-1, -1, 0), (-1, 0, -1), (-1, 0, 0),
            (0, -1, -1), (0, -1, 0), (0, 0, -1), (0, 0, 0)
        ]
        
        # Grid de interpolación normalizado [0, 1]
        interp_grid = np.linspace(0, 1, self.grid_resolution)
        
        points_generated = 0
        
        for origin in cube_origins:
            x0, y0, z0 = origin
            corners = self._get_cube_corners(x0, y0, z0)
            
            # Verificar que el cubo tenga todos los datos
            if not self._cube_exists(corners):
                continue
            
            # Obtener señales de las esquinas
            corner_signals = [
                self.original_signals[corner] for corner in corners
            ]
            
            # Interpolar puntos dentro del cubo
            for xd in interp_grid:
                for yd in interp_grid:
                    for zd in interp_grid:
                        # Calcular coordenadas globales
                        x_global = x0 + xd
                        y_global = y0 + yd
                        z_global = z0 + zd
                        
                        coords = (
                            round(x_global, 3),
                            round(y_global, 3),
                            round(z_global, 3)
                        )
                        
                        # Evitar duplicar puntos originales
                        if coords in self.original_signals:
                            continue
                        
                        # Evitar duplicar puntos sintéticos
                        if coords in self.synthetic_signals:
                            continue
                        
                        # Interpolar señal punto por punto
                        synthetic_signal = np.zeros(self.signal_length)
                        
                        for i in range(self.signal_length):
                            corner_vals = np.array([
                                signal[i] for signal in corner_signals
                            ])
                            
                            synthetic_signal[i] = self._interpolate_point(
                                xd, yd, zd, corner_vals
                            )
                        
                        self.synthetic_signals[coords] = synthetic_signal
                        points_generated += 1
        
        print(f"✓ Generados {points_generated} puntos sintéticos")
        print(f"✓ Total de datos: {len(self.original_signals) + points_generated}")
        
        return self.synthetic_signals
    
    def save_synthetic_data(self, output_directory: str):
        """
        Guarda los datos sintéticos en archivos
        
        Args:
            output_directory: Directorio de salida
        """
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nGuardando datos sintéticos en {output_directory}...")
        
        for coords, signal in self.synthetic_signals.items():
            x, y, z = coords
            
            # Crear DataFrame con formato original
            df = pd.DataFrame({
                'tiempo': np.arange(self.signal_length),
                'amplitud': signal
            })
            
            # Nombre de archivo
            filename = f"DatReceptor_{x}_{y}_{z}.txt"
            filepath = output_path / filename
            
            # Guardar archivo
            df.to_csv(
                filepath,
                sep='\t',
                header=False,
                index=False,
                float_format='%.8f'
            )
        
        print(f"✓ Archivos guardados exitosamente")


def augment_spatial_data(
    input_directory: str,
    output_directory: str,
    grid_resolution: int = 10
):
    """
    Función principal para aumentar datos espaciales
    
    Args:
        input_directory: Directorio con datos originales
        output_directory: Directorio para datos aumentados
        grid_resolution: Resolución del grid de interpolación
    """
    # Cargar datos originales
    print("Cargando datos originales...")
    loader = SpatialSignalLoader(input_directory)
    original_data = loader.load_all_signals()
    
    # Crear aumentador
    augmentor = TrilinearAugmentor(
        original_signals=original_data,
        grid_resolution=grid_resolution
    )
    
    # Generar datos sintéticos
    augmentor.generate_augmented_data()
    
    # Guardar datos sintéticos
    augmentor.save_synthetic_data(output_directory)
    
    print("\n=== Proceso de aumento de datos completado ===")


if __name__ == "__main__":
    # Ejemplo de uso
    from settings import paths, data_params
    
    augment_spatial_data(
        input_directory=paths.raw_data_dir,
        output_directory=paths.augmented_data_dir,
        grid_resolution=data_params.interpolation_steps
    )
