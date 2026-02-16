"""
Script principal para realizar inferencias con el modelo entrenado
Permite predecir señales en coordenadas arbitrarias del espacio
"""

import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

from settings import paths, model_params, data_params
from data_loader import load_spatial_data
from neural_network import SpatialSignalInterpolator
from visualizacion import visualize_prediction


class SignalPredictor:
    """
    Clase para realizar predicciones con el modelo entrenado
    """
    
    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        """
        Inicializa el predictor
        
        Args:
            checkpoint_path: Ruta al modelo guardado
            device: Dispositivo a usar ('cpu' o 'cuda')
        """
        self.checkpoint_path = Path(checkpoint_path)
        
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo en: {checkpoint_path}\n"
                "Por favor, ejecuta primero entrenar_modelo.py"
            )
        
        # Configurar dispositivo
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Dispositivo seleccionado: {self.device}")
        
        # Cargar checkpoint
        self._load_checkpoint()
    
    def _load_checkpoint(self):
        """Carga el modelo y sus metadatos"""
        print(f"\nCargando modelo desde: {self.checkpoint_path}")
        
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False
        )
        
        # Extraer configuración del modelo
        model_config = checkpoint.get('model_config', {})
        
        # Crear modelo con la arquitectura guardada
        self.model = SpatialSignalInterpolator(
            input_dim=model_config.get('input_dim', 3),
            hidden_dims=model_config.get('hidden_dims', (256, 512, 256)),
            output_dim=model_config.get('output_dim', 250),
            dropout_prob=model_config.get('dropout_prob', 0.3),
            use_batch_norm=model_config.get('use_batch_norm', True)
        )
        
        # Cargar pesos
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Guardar factor de normalización
        self.normalization_factor = checkpoint.get('normalization_factor', 1.0)
        
        print(f"✓ Modelo cargado exitosamente")
        print(f"✓ Factor de normalización: {self.normalization_factor:.6f}")
        
        # Mostrar métricas del modelo si están disponibles
        if 'final_metrics' in checkpoint:
            metrics = checkpoint['final_metrics']
            print("\nMétricas del modelo:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    if 'Parameters' in key:
                        print(f"  - {key}: {value:,}")
                    else:
                        print(f"  - {key}: {value:.6f}")
    
    def predict(
        self,
        coordinates: Tuple[float, float, float],
        return_normalized: bool = False
    ) -> np.ndarray:
        """
        Predice la señal para unas coordenadas dadas
        
        Args:
            coordinates: Tupla (x, y, z) con las coordenadas
            return_normalized: Si retornar señal normalizada
            
        Returns:
            Array numpy con la señal predicha
        """
        # Validar coordenadas
        if not self._validate_coordinates(coordinates):
            print("Advertencia: Coordenadas fuera del rango de entrenamiento")
        
        # Preparar tensor de entrada
        coords_tensor = torch.tensor(
            [coordinates],
            dtype=torch.float32,
            device=self.device
        )
        
        # Realizar predicción
        with torch.no_grad():
            prediction = self.model(coords_tensor)
        
        # Convertir a numpy
        signal = prediction.cpu().numpy().squeeze()
        
        # Desnormalizar si es necesario
        if not return_normalized:
            signal = signal * self.normalization_factor
        
        return signal
    
    def _validate_coordinates(self, coords: Tuple[float, float, float]) -> bool:
        """
        Valida que las coordenadas estén en el rango esperado
        
        Args:
            coords: Coordenadas a validar
            
        Returns:
            True si están en el rango válido
        """
        x, y, z = coords
        min_val, max_val = data_params.coordinate_range
        
        return all(min_val <= c <= max_val for c in [x, y, z])
    
    def predict_batch(
        self,
        coordinates_list: list,
        return_normalized: bool = False
    ) -> np.ndarray:
        """
        Predice señales para múltiples coordenadas
        
        Args:
            coordinates_list: Lista de tuplas con coordenadas
            return_normalized: Si retornar señales normalizadas
            
        Returns:
            Array numpy (n_samples, signal_length)
        """
        coords_tensor = torch.tensor(
            coordinates_list,
            dtype=torch.float32,
            device=self.device
        )
        
        with torch.no_grad():
            predictions = self.model(coords_tensor)
        
        signals = predictions.cpu().numpy()
        
        if not return_normalized:
            signals = signals * self.normalization_factor
        
        return signals
    
    def evaluate_on_real_data(
        self,
        test_data: dict,
        n_samples: int = 5
    ):
        """
        Evalúa el modelo en datos reales
        
        Args:
            test_data: Diccionario con datos reales
            n_samples: Número de muestras a evaluar
        """
        print(f"\nEvaluando modelo en {n_samples} muestras reales...")
        
        coords_list = list(test_data.keys())[:n_samples]
        errors = []
        
        for coords in coords_list:
            real_signal = test_data[coords][:model_params.output_length]
            predicted_signal = self.predict(coords, return_normalized=False)
            
            # Calcular error
            mse = np.mean((real_signal - predicted_signal) ** 2)
            mae = np.mean(np.abs(real_signal - predicted_signal))
            
            errors.append((coords, mse, mae))
            
            print(f"  Coord {coords}: MSE={mse:.6f}, MAE={mae:.6f}")
        
        avg_mse = np.mean([e[1] for e in errors])
        avg_mae = np.mean([e[2] for e in errors])
        
        print(f"\n  Promedio - MSE: {avg_mse:.6f}, MAE: {avg_mae:.6f}")


def main():
    """Función principal de inferencia"""
    
    print("\n" + "=" * 70)
    print("SISTEMA DE PREDICCIÓN DE SEÑALES ESPACIALES")
    print("=" * 70)
    
    # Ruta al modelo
    model_path = Path(paths.models_dir) / model_params.checkpoint_name
    
    # Cargar datos originales para comparación
    print("\nCargando datos de referencia...")
    reference_data = load_spatial_data(paths.raw_data_dir)
    
    # Inicializar predictor
    predictor = SignalPredictor(str(model_path))
    
    # Definir coordenadas para predicción
    test_coordinates = [
        (0.3, -0.2, 0.5),
        (0.5, 0.5, 0.0),
        (-0.5, 0.0, 0.5),
    ]
    
    print("\n" + "=" * 70)
    print("REALIZANDO PREDICCIONES")
    print("=" * 70)
    
    for i, coords in enumerate(test_coordinates, 1):
        print(f"\n[{i}] Predicción para coordenadas: {coords}")
        
        # Realizar predicción
        predicted_signal = predictor.predict(coords)
        
        print(f"  ✓ Señal predicha (longitud: {len(predicted_signal)})")
        print(f"  ✓ Amplitud máxima: {np.max(predicted_signal):.6f}")
        print(f"  ✓ Amplitud mínima: {np.min(predicted_signal):.6f}")
        
        # Visualizar
        output_plot = Path(paths.results_dir) / f"prediccion_{i}_coords{coords}.png"
        
        visualize_prediction(
            prediction_coords=coords,
            predicted_signal=predicted_signal,
            reference_data=reference_data,
            output_path=str(output_plot)
        )
    
    # Evaluar en datos reales
    print("\n" + "=" * 70)
    predictor.evaluate_on_real_data(reference_data, n_samples=5)
    print("=" * 70)
    
    print("\nProceso de inferencia completado\n")


if __name__ == "__main__":
    main()
