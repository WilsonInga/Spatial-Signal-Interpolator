"""
Script de entrenamiento del modelo de interpolación espacial
Incluye métricas avanzadas, visualización y early stopping
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict
import time

from settings import paths, model_params, data_params
from data_loader import load_spatial_data
from custom_dataset import prepare_dataset_from_dict
from neural_network import SpatialSignalInterpolator, EarlyStopping


class TrainingMetrics:
    """Clase para rastrear métricas de entrenamiento"""
    
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_mae = []
        self.val_mae = []
        self.epoch_times = []
    
    def add_epoch(
        self,
        train_loss: float,
        val_loss: float,
        train_mae: float,
        val_mae: float,
        epoch_time: float
    ):
        """Agrega métricas de una época"""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_mae.append(train_mae)
        self.val_mae.append(val_mae)
        self.epoch_times.append(epoch_time)
    
    def plot_training_curves(self, save_path: str):
        """Genera gráficas de curvas de aprendizaje"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Gráfica de MSE Loss
        axes[0].plot(self.train_losses, label='Entrenamiento', linewidth=2)
        axes[0].plot(self.val_losses, label='Validación', linewidth=2, linestyle='--')
        axes[0].set_xlabel('Época', fontsize=12)
        axes[0].set_ylabel('Error Cuadrático Medio (MSE)', fontsize=12)
        axes[0].set_title('Evolución del Error durante Entrenamiento', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Gráfica de MAE
        axes[1].plot(self.train_mae, label='Entrenamiento', linewidth=2, color='green')
        axes[1].plot(self.val_mae, label='Validación', linewidth=2, linestyle='--', color='orange')
        axes[1].set_xlabel('Época', fontsize=12)
        axes[1].set_ylabel('Error Absoluto Medio (MAE)', fontsize=12)
        axes[1].set_title('Error Absoluto Medio vs Épocas', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráficas guardadas en: {save_path}")
        plt.close()


class ModelTrainer:
    """Clase para gestionar el entrenamiento del modelo"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        config: object
    ):
        """
        Inicializa el entrenador
        
        Args:
            model: Modelo a entrenar
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación
            device: Dispositivo (CPU/GPU)
            config: Configuración del modelo
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Definir función de pérdida y optimizador
        self.criterion_mse = nn.MSELoss()
        self.criterion_mae = nn.L1Loss()
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler para ajustar learning rate
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta,
            verbose=True
        )
        
        self.metrics = TrainingMetrics()
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Entrena una época completa
        
        Returns:
            Tupla (MSE loss, MAE)
        """
        self.model.train()
        total_mse = 0.0
        total_mae = 0.0
        num_batches = 0
        
        for coords, signals in self.train_loader:
            coords = coords.to(self.device)
            signals = signals.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(coords)
            
            # Calcular pérdidas
            loss_mse = self.criterion_mse(predictions, signals)
            loss_mae = self.criterion_mae(predictions, signals)
            
            # Backward pass
            loss_mse.backward()
            self.optimizer.step()
            
            # Acumular métricas
            total_mse += loss_mse.item() * coords.size(0)
            total_mae += loss_mae.item() * coords.size(0)
            num_batches += coords.size(0)
        
        epoch_mse = total_mse / num_batches
        epoch_mae = total_mae / num_batches
        
        return epoch_mse, epoch_mae
    
    def validate(self) -> Tuple[float, float]:
        """
        Valida el modelo
        
        Returns:
            Tupla (MSE loss, MAE)
        """
        self.model.eval()
        total_mse = 0.0
        total_mae = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for coords, signals in self.val_loader:
                coords = coords.to(self.device)
                signals = signals.to(self.device)
                
                predictions = self.model(coords)
                
                loss_mse = self.criterion_mse(predictions, signals)
                loss_mae = self.criterion_mae(predictions, signals)
                
                total_mse += loss_mse.item() * coords.size(0)
                total_mae += loss_mae.item() * coords.size(0)
                num_samples += coords.size(0)
        
        val_mse = total_mse / num_samples
        val_mae = total_mae / num_samples
        
        return val_mse, val_mae
    
    def train(self) -> Dict:
        """
        Ejecuta el entrenamiento completo
        
        Returns:
            Diccionario con resultados finales
        """
        print("\n" + "=" * 70)
        print("INICIANDO ENTRENAMIENTO")
        print("=" * 70)
        print(f"Épocas totales: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate inicial: {self.config.learning_rate}")
        print(f"Dispositivo: {self.device}")
        print("=" * 70 + "\n")
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()
            
            # Entrenar
            train_mse, train_mae = self.train_epoch()
            
            # Validar
            val_mse, val_mae = self.validate()
            
            epoch_time = time.time() - epoch_start
            
            # Guardar métricas
            self.metrics.add_epoch(
                train_mse, val_mse,
                train_mae, val_mae,
                epoch_time
            )
            
            # Actualizar learning rate
            self.scheduler.step(val_mse)
            
            # Imprimir progreso
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Época [{epoch+1:3d}/{self.config.num_epochs}] | "
                      f"Train MSE: {train_mse:.6f} | Val MSE: {val_mse:.6f} | "
                      f"Train MAE: {train_mae:.6f} | Val MAE: {val_mae:.6f} | "
                      f"Tiempo: {epoch_time:.2f}s")
            
            # Early stopping
            if self.early_stopping(val_mse, self.model):
                print(f"\n⚠ Early stopping activado en época {epoch+1}")
                break
        
        total_time = time.time() - start_time
        
        # Cargar mejor modelo
        self.early_stopping.load_best_model(self.model)
        
        print("\n" + "=" * 70)
        print(f"ENTRENAMIENTO COMPLETADO en {total_time/60:.2f} minutos")
        print("=" * 70)
        
        return {
            'best_val_loss': self.early_stopping.best_loss,
            'total_epochs': len(self.metrics.train_losses),
            'total_time': total_time
        }
    
    def compute_final_metrics(self) -> Dict:
        """Calcula métricas finales en conjunto de validación"""
        self.model.eval()
        
        predictions_list = []
        targets_list = []
        
        with torch.no_grad():
            for coords, signals in self.val_loader:
                coords = coords.to(self.device)
                signals = signals.to(self.device)
                
                preds = self.model(coords)
                
                predictions_list.append(preds.cpu())
                targets_list.append(signals.cpu())
        
        all_preds = torch.cat(predictions_list, dim=0)
        all_targets = torch.cat(targets_list, dim=0)
        
        # Calcular métricas
        mse = float(self.criterion_mse(all_preds, all_targets))
        mae = float(self.criterion_mae(all_preds, all_targets))
        rmse = np.sqrt(mse)
        
        # R² Score
        ss_res = torch.sum((all_targets - all_preds) ** 2)
        ss_tot = torch.sum((all_targets - torch.mean(all_targets)) ** 2)
        r2_score = 1 - (ss_res / ss_tot)
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2_Score': float(r2_score),
            'Num_Parameters': self.model.count_parameters()
        }


def main():
    """Función principal de entrenamiento"""
    
    # Crear directorios necesarios
    paths.create_directories()
    
    # Configurar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Usando dispositivo: {device}")
    
    # Cargar datos
    print("\n📂 Cargando datos de entrenamiento...")
    raw_data = load_spatial_data(paths.training_data_dir)
    
    # Preparar dataset
    full_dataset = prepare_dataset_from_dict(
        data_dict=raw_data,
        signal_length=data_params.signal_length,
        device=device,
        apply_normalization=data_params.normalize_signals
    )
    
    # Dividir en train/val
    train_size = int(model_params.train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"\n📊 División de datos:")
    print(f"  - Entrenamiento: {len(train_dataset)} muestras")
    print(f"  - Validación: {len(val_dataset)} muestras")
    
    # Crear DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=model_params.batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=model_params.batch_size,
        shuffle=False
    )
    
    # Crear modelo
    model = SpatialSignalInterpolator(
        input_dim=model_params.input_dimensions,
        hidden_dims=(
            model_params.hidden_layer_1,
            model_params.hidden_layer_2,
            model_params.hidden_layer_3
        ),
        output_dim=model_params.output_length,
        dropout_prob=model_params.dropout_rate,
        use_batch_norm=model_params.use_batch_norm
    )
    
    print("\n" + model.get_architecture_summary())
    
    # Entrenar modelo
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=model_params
    )
    
    training_results = trainer.train()
    
    # Métricas finales
    print("\n📈 Calculando métricas finales...")
    final_metrics = trainer.compute_final_metrics()
    
    print("\n" + "=" * 70)
    print("MÉTRICAS FINALES DEL MODELO")
    print("=" * 70)
    for metric, value in final_metrics.items():
        if 'Score' in metric:
            print(f"{metric:20s}: {value:.4f}")
        elif 'Parameters' in metric:
            print(f"{metric:20s}: {value:,}")
        else:
            print(f"{metric:20s}: {value:.6f}")
    print("=" * 70)
    
    # Guardar modelo
    checkpoint_path = Path(paths.models_dir) / model_params.checkpoint_name
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'normalization_factor': full_dataset.get_normalization_factor(),
        'model_config': {
            'input_dim': model_params.input_dimensions,
            'hidden_dims': (
                model_params.hidden_layer_1,
                model_params.hidden_layer_2,
                model_params.hidden_layer_3
            ),
            'output_dim': model_params.output_length,
            'dropout_prob': model_params.dropout_rate,
            'use_batch_norm': model_params.use_batch_norm
        },
        'training_results': training_results,
        'final_metrics': final_metrics
    }, checkpoint_path)
    
    print(f"\n💾 Modelo guardado en: {checkpoint_path}")
    
    # Generar gráficas
    plot_path = Path(paths.results_dir) / "curvas_entrenamiento.png"
    trainer.metrics.plot_training_curves(str(plot_path))
    
    print("\n✅ Proceso de entrenamiento completado exitosamente\n")


if __name__ == "__main__":
    main()
