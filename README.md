# Spatial Signal Interpolator 📡🧠

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Status](https://img.shields.io/badge/Mantenimiento-Activo-green)

Framework de Deep Learning diseñado para la reconstrucción y predicción de señales temporales en coordenadas espaciales 3D. Este proyecto resuelve el problema de la dispersión de sensores mediante un **Pipeline de Aumentación Trilineal** y una **Red Neuronal Profunda (DNN)** regresiva.

## 📋 Características del Proyecto

- **Arquitectura MLOps Modular:** Separación clara entre configuración, procesamiento de datos, definición del modelo y loops de entrenamiento.
- **Data Augmentation 3D:** Generación sintética de puntos de entrenamiento utilizando interpolación trilineal (`data_augmentation.py`) para densificar la nube de puntos y mejorar la generalización.
- **Modelo Regresivo Profundo:** Perceptrón Multicapa (MLP) con:
  - Entrada: Coordenadas $(x, y, z)$.
  - Salida: Vector de señal temporal ($L=250$).
  - Regularización: Batch Normalization y Dropout.
- **Early Stopping:** Implementación personalizada para detener el entrenamiento cuando la pérdida de validación se estanca, evitando el overfitting.

## 🛠️ Estructura del Repositorio

```text
├── settings.py            # Configuración global e hiperparámetros
├── requirements.txt       # Dependencias del proyecto
├── data_loader.py         # Carga y validación de archivos .txt
├── data_augmentation.py   # Motor de interpolación trilineal
├── custom_dataset.py      # Dataset de PyTorch con normalización
├── neural_network.py      # Arquitectura del modelo (nn.Module)
├── entrenar_modelo.py     # Script principal de entrenamiento
├── ejecutar_prediccion.py # Script de inferencia y evaluación
├── utilidades.py          # Herramientas de análisis y conversión
└── visualizacion.py       # Gráficas comparativas y mapas de calor
```

## 🚀 Instalación y Uso

### 1. Clonar y preparar entorno

```bash
git clone [https://github.com/WilsonInga/Spatial-Signal-Interpolator.git](https://github.com/WilsonInga/Spatial-Signal-Interpolator.git)
cd spatial-signal-interpolator
pip install -r requirements.txt

```

### 2. Preparación de Datos

Coloca tus archivos de sensores en la carpeta `datos_originales/`.

- **Formato requerido:** `DatReceptor_X_Y_Z.txt` (Ej: `DatReceptor_1.0_0.0_-0.5.txt`).

### 3. Entrenamiento

Ejecuta el pipeline de entrenamiento. El script detectará automáticamente si tienes GPU (CUDA) disponible.

```bash
python entrenar_modelo.py

```

_Esto guardará el mejor modelo en `modelos_guardados/` y generará gráficas de pérdida en `resultados/`._

### 4. Inferencia (Predicción)

Para predecir la señal en una coordenada específica donde no existe un sensor físico:

```bash
python ejecutar_prediccion.py

```

## 📊 Resultados Visuales

El módulo `visualizacion.py` permite generar comparativas directas entre la señal predicha por la IA y los sensores reales más cercanos (Nearest Neighbors), calculando métricas de error MSE y MAE para validar la precisión espacial.

## ✒️ Autor

**Proyecto Grupal- Modelos y Simulacion**
````
