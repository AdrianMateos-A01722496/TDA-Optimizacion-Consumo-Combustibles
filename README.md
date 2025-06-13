# 🔬 Análisis Topológico de Consumo de Combustible - SLB México

Presentación interactiva final del proyecto de análisis topológico para la optimización de consumo de combustible en la flota de vehículos de SLB México utilizando KeplerMapper y Machine Learning.

## 🎯 Características Principales

- **📊 Serie Temporal Interactiva**: Comparación de rendimiento real vs predicho con datos históricos
- **🗺️ Visualización Mapper**: Análisis topológico integrado usando KeplerMapper con clusters DBSCAN
- **🔮 Calculadora Predictiva**: Predicciones en tiempo real integrando características topológicas
- **📱 Diseño Responsivo**: Interfaz moderna con Tailwind CSS y Chart.js
- **🧠 Modelo Avanzado**: RandomForest optimizado con 9 características ingeniadas incluyendo topológicas

## 🚀 Inicio Rápido

### Ejecución del Sistema Completo

El sistema requiere **DOS procesos** ejecutándose simultáneamente:

#### 📋 Pasos para Ejecutar

1. **Instalar dependencias:**

```bash
pip install -r requirements.txt
```

2. **Ejecutar el sistema completo:**

```bash
python3 run_final_presentation.py
```

✅ Este script automáticamente:

- Inicia el servidor de presentación en puerto 8000
- Inicia la API Flask en puerto 5001
- Abre automáticamente el navegador en la presentación

**Alternativa Manual (DOS terminales):**

3. **Terminal 1 - API de Predicción (Puerto 5001):**

```bash
python3 app.py
```

✅ Verás: `Modelo cargado exitosamente!` y `Running on http://127.0.0.1:5001`

4. **Terminal 2 - Servidor de Presentación (Puerto 8000):**

```bash
python3 -m http.server 8000
```

5. **Abrir navegador:**

- 🌐 Ir a: `http://localhost:8000/presentacion_final.html`

### 🛑 Instrucciones para Cerrar

#### Cerrar los Servidores

En cada terminal donde tienes un servidor ejecutándose:

```bash
Ctrl + C
```

#### Verificar que los Puertos están Libres

```bash
# Verificar puerto 8000 (servidor de presentación)
lsof -i :8000

# Verificar puerto 5001 (API)
lsof -i :5001
```

## 📁 Estructura del Proyecto

```
RetoTopo/
├── 📄 presentacion_final.html          # Presentación interactiva principal
├── 🐍 app.py                          # API Flask para predicciones (puerto 5001)
├── 🧠 final_predictor.py              # Lógica de predicción con Mapper integrado
├── 🚀 run_final_presentation.py       # Script de inicio automático del sistema
├── 🔧 advanced_model_optimization.py  # Entrenamiento y optimización del modelo
├── 🗺️ generate_final_mapper.py        # Generación del análisis topológico
├── 📊 convert_csv_to_js.py            # Conversión de datos para frontend
├── 🔄 regenerate_mapper.py            # Regeneración de clusters topológicos
├── 💾 trained_model_complete.pkl      # Modelo RandomForest entrenado
├── ⚙️ best_model_params.json          # Parámetros y métricas del mejor modelo
├── 📋 requirements.txt                # Dependencias Python
├── 📖 README.md                       # Este archivo
├── data/
│   └── 📊 df_21_24_clean.xlsx         # Datos originales limpios de Edenred
├── final_presentation_data/
│   ├── 📈 pred_vs_real_advanced.csv   # Predicciones vs valores reales
│   ├── 📊 prediction_data.js          # Datos preprocesados para frontend
│   ├── 🗺️ edenred_mapper_final.html   # Visualización Mapper interactiva
└── notebooks/
    ├── 📔 RetoFase2.ipynb             # Análisis exploratorio inicial
    └── 📈 AnalisisDispersionYSerieTiempo.ipynb  # Análisis temporal
```

## 🔮 Usar la Calculadora Predictiva

### 1. Completar Variables Categóricas

Selecciona valores para cada campo usando los dropdowns:

- **Id Región**: Identificador de la región geográfica (1-20)
- **Cliente**: Código del cliente (múltiples opciones disponibles)
- **División**: División organizacional
- **BL**: Business Line
- **Mercancía**: Tipo de combustible (Gasolina, Diesel, etc.)
- **Conductor**: Identificador del conductor
- **Vehículo**: Código del vehículo

### 2. Ingresar Variables Numéricas

- **Recorrido**: Distancia en kilómetros (1-10,000)
- **Rendimiento Esperado**: Eficiencia esperada en km/L (1-50)

### 3. Obtener Predicción

El sistema automáticamente:

1. 🧮 Aplica ingeniería de características (8 nuevas features)
2. 🗺️ Calcula el cluster topológico usando KeplerMapper
3. 🎯 Predice el rendimiento REAL usando RandomForest
4. 📊 Clasifica la eficiencia y proporciona análisis comparativo
5. 💡 Incluye distancia al centroide del cluster para contexto topológico

## 🧪 Tecnologías y Metodología

### Backend y Machine Learning

- **Python 3.8+**: Lenguaje principal
- **Flask**: API web RESTful para predicciones
- **scikit-learn**: RandomForest con 375 estimadores
- **KeplerMapper**: Análisis topológico de datos complejos
- **pandas + numpy**: Manipulación y procesamiento de datos

### Análisis Topológico (KeplerMapper)

- **Proyección**: PCA con 2 componentes principales
- **Cobertura**: 5 cubos con 50% de solapamiento
- **Clustering**: DBSCAN (eps=2.5, min_samples=100)
- **Resultado**: 13 clusters topológicos identificados

### Ingeniería de Características

El modelo utiliza **9 características ingeniadas**:

1. **Recorrido × Rendimiento Esperado**: Interacción multiplicativa
2. **Recorrido ÷ Rendimiento Esperado**: Ratio de eficiencia
3. **log(Recorrido)**: Transformación logarítmica
4. **Recorrido²**: Transformación cuadrática
5. **√Recorrido**: Transformación de raíz cuadrada
6. **Recorrido³**: Transformación cúbica
7. **Rendimiento Esperado²**: Cuadrática del rendimiento
8. **Cluster Mapper**: ID del cluster topológico
9. **Distancia al Centroide**: Proximidad dentro del cluster

### Frontend

- **HTML5 + CSS3**: Estructura semántica moderna
- **Tailwind CSS**: Framework de utilidades CSS
- **Chart.js**: Visualizaciones interactivas responsivas
- **JavaScript ES6+**: Lógica asíncrona del frontend

## 📊 Métricas del Modelo Actual

El modelo RandomForest optimizado logra:

- **R² Score**: 0.6583 (65.83% de varianza explicada)
- **MSE**: 3.299
- **RMSE**: 1.816 km/L
- **MAE**: 1.357 km/L
- **MAPE**: 32.60%

### Parámetros del Modelo

- **Estimadores**: 375 árboles
- **Profundidad máxima**: 11 niveles
- **Muestras mínimas por split**: 11
- **Muestras mínimas por hoja**: 2
- **Características por split**: sqrt (característica automática)

### Dataset Procesado

- **Registros totales**: 36,158 observaciones
- **Período**: 2021-2024
- **Clusters topológicos**: 13 grupos identificados
- **Variables categóricas**: 6 codificadas con OneHot
- **Características finales**: 47 variables después de ingeniería

## 🔧 Solución de Problemas

### Error: Puerto 5001 Ocupado

```bash
# Verificar qué proceso usa el puerto
lsof -i :5001

# Matar el proceso si es necesario
kill -9 <PID>
```

### La Calculadora Devuelve 0

1. ✅ Verificar que **ambos servidores** estén ejecutándose
2. 🧠 Comprobar logs en la terminal de la API (puerto 5001)
3. 🔍 Revisar la consola del navegador (F12) para errores de red
4. 📊 Verificar que `trained_model_complete.pkl` existe y no está corrupto

### Las Gráficas No Aparecen

1. 🌐 Verificar acceso vía HTTP (`http://localhost:8000`), no `file://`
2. 📂 Confirmar que `prediction_data.js` esté en `final_presentation_data/`
3. 🔄 Regenerar datos si es necesario: `python3 convert_csv_to_js.py`

### Error en Carga del Modelo

```bash
# Reentrenar el modelo si está corrupto
python3 advanced_model_optimization.py

# Verificar integridad
python3 -c "import pickle; pickle.load(open('trained_model_complete.pkl', 'rb'))"
```

### Clusters Mapper Inconsistentes

```bash
# Regenerar análisis topológico
python3 regenerate_mapper.py

# Verificar consistencia
python3 generate_final_mapper.py
```

## 🔄 Flujo de Trabajo para Actualizar el Modelo

### 1. Reentrenamiento Completo

```bash
python3 advanced_model_optimization.py
```

Esto actualiza:

- `trained_model_complete.pkl`
- `best_model_params.json`
- Clusters Mapper integrados

### 2. Regenerar Datos de Presentación

```bash
python3 convert_csv_to_js.py
```

### 3. Actualizar Visualización Mapper

```bash
python3 generate_final_mapper.py
```

### 4. Verificar Sistema Completo

```bash
python3 run_final_presentation.py
```

## 📈 Interpretación de Resultados

### Clusters Topológicos

- **13 clusters identificados**: Agrupaciones naturales en el espacio categórico
- **Continuidad topológica**: Transiciones suaves entre regiones similares
- **Mejora predictiva**: Los clusters capturan patrones no lineales complejos

### Clasificación de Eficiencia

- **Excelente**: ≥ 10 km/L (Verde)
- **Bueno**: 7-9.9 km/L (Azul)
- **Regular**: 5-6.9 km/L (Amarillo)
- **Bajo**: < 5 km/L (Rojo)

### Importancia de Características

1. **Recorrido y transformaciones**: Variables más predictivas
2. **Cluster Mapper**: Contexto topológico significativo
3. **Interacciones**: Capturan relaciones no lineales
4. **Variables categóricas**: Contexto operacional importante

## 🏆 Logros del Proyecto

- ✅ **Integración exitosa** de análisis topológico con ML tradicional
- ✅ **Sistema end-to-end** desde datos raw hasta predicciones en producción
- ✅ **Interfaz interactiva** para análisis exploratorio y predicción
- ✅ **Modelo robusto** con validación cruzada y métricas competitivas
- ✅ **Documentación completa** y código reproducible

---

*Proyecto desarrollado para SLB México - Optimización de Consumo de Combustible mediante Análisis Topológico de Datos*
