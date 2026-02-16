"""
Script de utilidades con funciones auxiliares para el proyecto
Incluye herramientas de análisis, conversión y validación
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple, List
import json


class ProjectUtils:
    """Clase con utilidades generales del proyecto"""
    
    @staticmethod
    def analyze_data_distribution(data_dict: Dict) -> Dict:
        """
        Analiza la distribución espacial de los receptores
        
        Args:
            data_dict: Diccionario con datos
            
        Returns:
            Diccionario con análisis estadístico
        """
        coords = np.array(list(data_dict.keys()))
        
        analysis = {
            'total_receptors': len(coords),
            'x_range': (coords[:, 0].min(), coords[:, 0].max()),
            'y_range': (coords[:, 1].min(), coords[:, 1].max()),
            'z_range': (coords[:, 2].min(), coords[:, 2].max()),
            'center_of_mass': tuple(coords.mean(axis=0)),
            'std_deviation': tuple(coords.std(axis=0))
        }
        
        return analysis
    
    @staticmethod
    def validate_signal_consistency(data_dict: Dict) -> Tuple[bool, str]:
        """
        Valida que todas las señales sean consistentes
        
        Args:
            data_dict: Diccionario con señales
            
        Returns:
            Tupla (válido, mensaje)
        """
        if not data_dict:
            return False, "Diccionario vacío"
        
        # Verificar longitudes
        lengths = [len(signal) for signal in data_dict.values()]
        if len(set(lengths)) > 1:
            return False, f"Longitudes inconsistentes: {set(lengths)}"
        
        # Verificar valores numéricos
        for coords, signal in data_dict.items():
            if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
                return False, f"Valores inválidos en {coords}"
        
        return True, "Todas las señales son válidas"
    
    @staticmethod
    def export_predictions_to_json(
        predictions: Dict[Tuple, np.ndarray],
        output_file: str
    ):
        """
        Exporta predicciones a formato JSON
        
        Args:
            predictions: Diccionario con predicciones
            output_file: Ruta del archivo de salida
        """
        # Convertir a formato serializable
        json_data = {}
        
        for coords, signal in predictions.items():
            key = f"({coords[0]:.3f}, {coords[1]:.3f}, {coords[2]:.3f})"
            json_data[key] = {
                'coordinates': list(coords),
                'signal': signal.tolist() if isinstance(signal, np.ndarray) else list(signal)
            }
        
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"✓ Predicciones exportadas a {output_file}")
    
    @staticmethod
    def calculate_signal_energy(signal: np.ndarray) -> float:
        """
        Calcula la energía de una señal
        
        Args:
            signal: Array con la señal
            
        Returns:
            Energía de la señal
        """
        return float(np.sum(signal ** 2))
    
    @staticmethod
    def calculate_signal_power(signal: np.ndarray) -> float:
        """
        Calcula la potencia promedio de una señal
        
        Args:
            signal: Array con la señal
            
        Returns:
            Potencia promedio
        """
        return float(np.mean(signal ** 2))
    
    @staticmethod
    def find_peak_location(signal: np.ndarray) -> Tuple[int, float]:
        """
        Encuentra el pico máximo de una señal
        
        Args:
            signal: Array con la señal
            
        Returns:
            Tupla (índice, valor) del pico
        """
        max_idx = np.argmax(np.abs(signal))
        max_val = signal[max_idx]
        
        return int(max_idx), float(max_val)
    
    @staticmethod
    def compute_decay_rate(signal: np.ndarray, threshold: float = 0.05) -> int:
        """
        Calcula el índice donde la señal decae por debajo del umbral
        
        Args:
            signal: Array con la señal
            threshold: Umbral de decaimiento
            
        Returns:
            Índice de decaimiento
        """
        normalized = np.abs(signal) / np.max(np.abs(signal))
        decay_indices = np.where(normalized < threshold)[0]
        
        if len(decay_indices) > 0:
            return int(decay_indices[0])
        return len(signal)
    
    @staticmethod
    def interpolate_coordinates(
        coord1: Tuple[float, float, float],
        coord2: Tuple[float, float, float],
        num_points: int = 10
    ) -> List[Tuple[float, float, float]]:
        """
        Interpola linealmente entre dos coordenadas
        
        Args:
            coord1: Primera coordenada
            coord2: Segunda coordenada
            num_points: Número de puntos intermedios
            
        Returns:
            Lista de coordenadas interpoladas
        """
        c1 = np.array(coord1)
        c2 = np.array(coord2)
        
        interpolated = []
        for alpha in np.linspace(0, 1, num_points):
            point = c1 * (1 - alpha) + c2 * alpha
            interpolated.append(tuple(point))
        
        return interpolated


class ModelAnalyzer:
    """Clase para analizar modelos entrenados"""
    
    def __init__(self, checkpoint_path: str):
        """
        Inicializa el analizador
        
        Args:
            checkpoint_path: Ruta al checkpoint del modelo
        """
        self.checkpoint = torch.load(
            checkpoint_path,
            map_location='cpu',
            weights_only=False
        )
    
    def get_model_summary(self) -> Dict:
        """
        Obtiene resumen del modelo
        
        Returns:
            Diccionario con información del modelo
        """
        summary = {
            'architecture': self.checkpoint.get('model_config', {}),
            'normalization_factor': self.checkpoint.get('normalization_factor', None),
            'training_results': self.checkpoint.get('training_results', {}),
            'final_metrics': self.checkpoint.get('final_metrics', {})
        }
        
        return summary
    
    def print_detailed_info(self):
        """Imprime información detallada del modelo"""
        print("\n" + "=" * 70)
        print("INFORMACIÓN DETALLADA DEL MODELO")
        print("=" * 70)
        
        summary = self.get_model_summary()
        
        print("\n📐 ARQUITECTURA:")
        arch = summary['architecture']
        for key, value in arch.items():
            print(f"  - {key}: {value}")
        
        print(f"\n🔢 NORMALIZACIÓN:")
        print(f"  - Factor: {summary['normalization_factor']:.6f}")
        
        print("\n📊 MÉTRICAS FINALES:")
        metrics = summary['final_metrics']
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if 'Parameters' in key:
                    print(f"  - {key}: {value:,}")
                else:
                    print(f"  - {key}: {value:.6f}")
        
        print("\n⏱️ RESULTADOS DE ENTRENAMIENTO:")
        results = summary['training_results']
        for key, value in results.items():
            if 'time' in key.lower():
                print(f"  - {key}: {value/60:.2f} minutos")
            else:
                print(f"  - {key}: {value}")
        
        print("=" * 70 + "\n")


def create_test_grid(
    bounds: Tuple[float, float] = (-1.0, 1.0),
    resolution: int = 5
) -> List[Tuple[float, float, float]]:
    """
    Crea una malla 3D de puntos de prueba
    
    Args:
        bounds: Límites (min, max) del espacio
        resolution: Número de puntos por dimensión
        
    Returns:
        Lista de coordenadas 3D
    """
    coords = np.linspace(bounds[0], bounds[1], resolution)
    
    grid_points = []
    for x in coords:
        for y in coords:
            for z in coords:
                grid_points.append((float(x), float(y), float(z)))
    
    return grid_points


def compare_signals_metrics(
    signal1: np.ndarray,
    signal2: np.ndarray
) -> Dict[str, float]:
    """
    Compara dos señales y calcula métricas
    
    Args:
        signal1: Primera señal
        signal2: Segunda señal
        
    Returns:
        Diccionario con métricas de comparación
    """
    # Asegurar misma longitud
    min_len = min(len(signal1), len(signal2))
    s1 = signal1[:min_len]
    s2 = signal2[:min_len]
    
    # Calcular métricas
    mse = float(np.mean((s1 - s2) ** 2))
    mae = float(np.mean(np.abs(s1 - s2)))
    max_error = float(np.max(np.abs(s1 - s2)))
    
    # Correlación
    correlation = float(np.corrcoef(s1, s2)[0, 1])
    
    # Error relativo
    relative_error = mae / (np.mean(np.abs(s1)) + 1e-10)
    
    return {
        'MSE': mse,
        'RMSE': np.sqrt(mse),
        'MAE': mae,
        'Max_Error': max_error,
        'Correlation': correlation,
        'Relative_Error': relative_error
    }


if __name__ == "__main__":
    # Ejemplo de uso
    print("Módulo de utilidades cargado correctamente")
    print("\nFunciones disponibles:")
    print("  - ProjectUtils: Utilidades generales")
    print("  - ModelAnalyzer: Análisis de modelos")
    print("  - create_test_grid: Crear malla de prueba")
    print("  - compare_signals_metrics: Comparar señales")
