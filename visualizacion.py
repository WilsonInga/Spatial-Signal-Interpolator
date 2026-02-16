"""
Módulo de visualización para análisis de señales espaciales
Genera gráficas comparativas entre señales reales y predichas
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Tuple, Optional
import matplotlib.gridspec as gridspec


class SignalVisualizer:
    """
    Clase para visualizar señales y comparaciones
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Inicializa el visualizador
        
        Args:
            style: Estilo de matplotlib a usar
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        self.colors = {
            'real': '#2E86AB',
            'predicted': '#A23B72',
            'neighbor1': '#F18F01',
            'neighbor2': '#C73E1D'
        }
    
    def plot_single_signal(
        self,
        signal: np.ndarray,
        coordinates: Tuple[float, float, float],
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ):
        """
        Grafica una señal individual
        
        Args:
            signal: Array con la señal
            coordinates: Coordenadas espaciales (x, y, z)
            title: Título personalizado
            save_path: Ruta para guardar la figura
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        
        time_indices = np.arange(len(signal))
        
        ax.plot(time_indices, signal, linewidth=2, color=self.colors['real'])
        ax.set_xlabel('Índice Temporal', fontsize=12)
        ax.set_ylabel('Amplitud', fontsize=12)
        
        if title is None:
            title = f'Señal en coordenadas {coordinates}'
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Figura guardada: {save_path}")
        
        plt.show()
        plt.close()
    
    def find_nearest_neighbors(
        self,
        target_coords: Tuple[float, float, float],
        data_dict: Dict[Tuple[float, float, float], np.ndarray],
        n_neighbors: int = 2
    ) -> list:
        """
        Encuentra los vecinos más cercanos a unas coordenadas
        
        Args:
            target_coords: Coordenadas objetivo
            data_dict: Diccionario con todas las señales
            n_neighbors: Número de vecinos a encontrar
            
        Returns:
            Lista de tuplas (coordenadas, distancia)
        """
        target = np.array(target_coords)
        distances = []
        
        for coords in data_dict.keys():
            coord_array = np.array(coords)
            distance = np.linalg.norm(coord_array - target)
            distances.append((coords, distance))
        
        # Ordenar por distancia
        distances.sort(key=lambda x: x[1])
        
        return distances[:n_neighbors]
    
    def plot_prediction_comparison(
        self,
        target_coords: Tuple[float, float, float],
        predicted_signal: np.ndarray,
        reference_data: Dict[Tuple[float, float, float], np.ndarray],
        signal_length: int = 250,
        save_path: Optional[str] = None
    ):
        """
        Compara la señal predicha con vecinos reales
        
        Args:
            target_coords: Coordenadas de predicción
            predicted_signal: Señal predicha por el modelo
            reference_data: Diccionario con señales reales
            signal_length: Longitud de señal a graficar
            save_path: Ruta para guardar
        """
        # Encontrar vecinos más cercanos
        neighbors = self.find_nearest_neighbors(target_coords, reference_data, n_neighbors=3)
        
        # Crear figura con subplots
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1])
        
        # Subplot principal: comparación
        ax_main = plt.subplot(gs[0])
        time_indices = np.arange(signal_length)
        
        # Graficar vecinos
        for i, (coords, dist) in enumerate(neighbors[:2]):
            neighbor_signal = reference_data[coords][:signal_length]
            linestyle = '--' if i == 0 else ':'
            color = self.colors['neighbor1'] if i == 0 else self.colors['neighbor2']
            
            ax_main.plot(
                time_indices,
                neighbor_signal,
                linestyle=linestyle,
                linewidth=2,
                label=f'Real más cercano {i+1}: {coords} (d={dist:.3f})',
                color=color,
                alpha=0.7
            )
        
        # Graficar predicción
        ax_main.plot(
            time_indices,
            predicted_signal[:signal_length],
            linewidth=2.5,
            label=f'Predicción RNA: {target_coords}',
            color=self.colors['predicted']
        )
        
        ax_main.set_ylabel('Amplitud', fontsize=12)
        ax_main.set_title(
            'Comparación: Señal Predicha vs Señales Reales Cercanas',
            fontsize=14,
            fontweight='bold'
        )
        ax_main.legend(fontsize=10, loc='upper right')
        ax_main.grid(True, alpha=0.3)
        
        # Subplot 2: Error con vecino más cercano
        ax_error = plt.subplot(gs[1])
        nearest_coords, _ = neighbors[0]
        nearest_signal = reference_data[nearest_coords][:signal_length]
        error = predicted_signal[:signal_length] - nearest_signal
        
        ax_error.plot(time_indices, error, linewidth=1.5, color='red')
        ax_error.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax_error.set_ylabel('Error', fontsize=11)
        ax_error.set_title('Diferencia: Predicción - Vecino más cercano', fontsize=12)
        ax_error.grid(True, alpha=0.3)
        
        # Subplot 3: Error absoluto acumulado
        ax_cumulative = plt.subplot(gs[2])
        cumulative_error = np.cumsum(np.abs(error))
        
        ax_cumulative.plot(
            time_indices,
            cumulative_error,
            linewidth=1.5,
            color='purple'
        )
        ax_cumulative.set_xlabel('Índice Temporal', fontsize=12)
        ax_cumulative.set_ylabel('Error Abs. Acumulado', fontsize=11)
        ax_cumulative.set_title('Error Absoluto Acumulado', fontsize=12)
        ax_cumulative.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Comparación guardada: {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_signal_heatmap(
        self,
        data_dict: Dict[Tuple[float, float, float], np.ndarray],
        z_slice: float = 0.0,
        time_index: int = 0,
        save_path: Optional[str] = None
    ):
        """
        Genera un mapa de calor de amplitudes en un corte Z
        
        Args:
            data_dict: Diccionario con señales
            z_slice: Valor de Z para el corte
            time_index: Índice temporal a visualizar
            save_path: Ruta para guardar
        """
        # Filtrar puntos en el corte Z
        points = []
        amplitudes = []
        
        for coords, signal in data_dict.items():
            if abs(coords[2] - z_slice) < 0.1:  # Tolerancia
                points.append([coords[0], coords[1]])
                amplitudes.append(signal[time_index])
        
        if not points:
            print(f"⚠ No hay puntos en el corte Z={z_slice}")
            return
        
        points = np.array(points)
        amplitudes = np.array(amplitudes)
        
        # Crear figura
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(
            points[:, 0],
            points[:, 1],
            c=amplitudes,
            s=200,
            cmap='viridis',
            edgecolors='black',
            linewidth=1
        )
        
        ax.set_xlabel('Coordenada X', fontsize=12)
        ax.set_ylabel('Coordenada Y', fontsize=12)
        ax.set_title(
            f'Mapa de Amplitudes en Z={z_slice} (t={time_index})',
            fontsize=14,
            fontweight='bold'
        )
        ax.grid(True, alpha=0.3)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Amplitud', fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Mapa de calor guardado: {save_path}")
        
        plt.show()
        plt.close()


def visualize_prediction(
    prediction_coords: Tuple[float, float, float],
    predicted_signal: np.ndarray,
    reference_data: Dict,
    output_path: Optional[str] = None
):
    """
    Función de conveniencia para visualizar predicciones
    
    Args:
        prediction_coords: Coordenadas de predicción
        predicted_signal: Señal predicha
        reference_data: Datos de referencia
        output_path: Ruta de salida opcional
    """
    visualizer = SignalVisualizer()
    
    visualizer.plot_prediction_comparison(
        target_coords=prediction_coords,
        predicted_signal=predicted_signal,
        reference_data=reference_data,
        save_path=output_path
    )
