import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
import kmapper as km
import warnings
warnings.filterwarnings('ignore')

class AdvancedFuelPredictionModel:
    def __init__(self):
        self.df_original = None
        self.ohe = None
        self.pca = None
        self.mapper = None
        self.graph = None
        self.best_models = {}
        self.ensemble_model = None
        self.best_params = {}
        self.metrics = {}
        self.feature_names = []
        self.preprocessor = None
        
    def load_and_prepare_data(self):
        """Carga y prepara los datos con feature engineering avanzado"""
        print("📊 Cargando datos...")
        self.df_original = pd.read_excel('data/df_21_24_clean.xlsx')
        
        # ordenar por fecha
        self.df_original = self.df_original.sort_values(['Cliente', 'Conductor', 'Year', 'Month']).reset_index(drop=True)
        
        print(f"✅ Datos cargados: {len(self.df_original)} registros")
        
        # **FEATURE ENGINEERING AVANZADO**
        print("🔧 Aplicando feature engineering avanzado...")
        
        # 1. Interacciones y ratios
        self.df_original['Recorrido_x_Rendimiento'] = self.df_original['Recorrido'] * self.df_original['Rendimiento']
        self.df_original['Rendimiento_per_km'] = self.df_original['Rendimiento'] / (self.df_original['Recorrido'] + 1)  # +1 para evitar división por 0
        self.df_original['Eficiencia_esperada'] = self.df_original['Rendimiento'] / self.df_original['Recorrido'].clip(lower=1)
        
        # 2. Transformaciones no lineales
        self.df_original['Recorrido_log'] = np.log1p(self.df_original['Recorrido'])
        self.df_original['Rendimiento_squared'] = self.df_original['Rendimiento'] ** 2
        
        # llenar NaN con medianas
        numeric_cols = self.df_original.select_dtypes(include=[np.number]).columns
        self.df_original[numeric_cols] = self.df_original[numeric_cols].fillna(self.df_original[numeric_cols].median())
        
        print("✅ Feature engineering completado")
        return self.df_original
    
    def create_advanced_mapper_clusters(self):
        """Crea clusters usando Mapper con configuración avanzada"""
        print("🗺️ Creando clusters topológicos avanzados...")
        
        categorical_cols = [
            'Id Región', 'Cliente', 'Division', 'BL', 
            'Mercancía', 'Conductor', 'Vehículo_Corto'
        ]
        
        # preparar datos categóricos
        self.df_original[categorical_cols] = self.df_original[categorical_cols].fillna('NULO')
        for c in categorical_cols:
            self.df_original[c] = self.df_original[c].astype(str)
        
        # one-hot encoding
        self.ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        X_cat_ohe = self.ohe.fit_transform(self.df_original[categorical_cols])
        print(f"✅ Dimensiones después de OneHotEncoder: {X_cat_ohe.shape}")
        
        # PCA con 2 componentes
        self.pca = PCA(n_components=2, random_state=42)
        lens = self.pca.fit_transform(X_cat_ohe)
        print(f"✅ Dimensiones después de PCA: {lens.shape}")
        
        # configurar Mapper con parámetros optimizados
        self.mapper = km.KeplerMapper(verbose=1)
        coverer = km.Cover(n_cubes=5, perc_overlap=0.5)
        clusterer = DBSCAN(eps=2.5, min_samples=100)
        
        # construir grafo
        print("🔄 Generando grafo...")
        self.graph = self.mapper.map(
            lens,
            X_cat_ohe,
            cover=coverer,
            clusterer=clusterer
        )
        
        # asignar clusters
        n_samples = X_cat_ohe.shape[0]
        mapper_cluster = np.full(shape=n_samples, fill_value=-1, dtype=int)
        
        cluster_id = 0
        for node_id_str in self.graph["nodes"].keys():
            for idx in self.graph["nodes"][node_id_str]:
                if mapper_cluster[idx] == -1:
                    mapper_cluster[idx] = cluster_id
            cluster_id += 1
        
        self.df_original["mapper_cluster"] = mapper_cluster.astype(str)
        
        # **FEATURES TOPOLÓGICAS ADICIONALES**
        # distancia al centroide del cluster
        cluster_centroids = {}
        for cluster_id in np.unique(mapper_cluster):
            if cluster_id >= 0:
                cluster_mask = mapper_cluster == cluster_id
                cluster_centroids[cluster_id] = lens[cluster_mask].mean(axis=0)
        
        distances_to_centroid = []
        for i, cluster_id in enumerate(mapper_cluster):
            if cluster_id >= 0 and cluster_id in cluster_centroids:
                dist = np.linalg.norm(lens[i] - cluster_centroids[cluster_id])
                distances_to_centroid.append(dist)
            else:
                distances_to_centroid.append(0)
        
        self.df_original['cluster_distance'] = distances_to_centroid
        
        # tamaño del cluster
        cluster_sizes = {}
        for cluster_id in np.unique(mapper_cluster):
            cluster_sizes[cluster_id] = np.sum(mapper_cluster == cluster_id)
        
        self.df_original['cluster_size'] = [cluster_sizes[cid] for cid in mapper_cluster]
        
        print(f"✅ Clusters creados: {len(np.unique(mapper_cluster))}")
        return self.df_original
    
    def remove_outliers(self):
        """Remueve outliers usando IQR"""
        print("🧹 Removiendo outliers...")
        
        target = 'Rendimiento Real'
        Q1 = self.df_original[target].quantile(0.25)
        Q3 = self.df_original[target].quantile(0.75)
        IQR = Q3 - Q1
        
        # definir límites
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # filtrar outliers
        initial_count = len(self.df_original)
        self.df_original = self.df_original[
            (self.df_original[target] >= lower_bound) & 
            (self.df_original[target] <= upper_bound)
        ].reset_index(drop=True)
        
        final_count = len(self.df_original)
        removed = initial_count - final_count
        
        print(f"✅ Outliers removidos: {removed} ({removed/initial_count*100:.1f}%)")
        return self.df_original
    
    def prepare_features(self):
        """Prepara las features para el modelo"""
        print("🎯 Preparando features...")
        
        # features numéricas básicas
        numeric_features = [
            'Recorrido', 'Rendimiento', 
            'Recorrido_x_Rendimiento', 'Rendimiento_per_km', 'Eficiencia_esperada',
            'Recorrido_log', 'Rendimiento_squared',
            'cluster_distance', 'cluster_size'
        ]
        
        # features categóricas
        categorical_features = ['mapper_cluster']
        
        self.feature_names = numeric_features + categorical_features
        
        X = self.df_original[self.feature_names].copy()
        y = self.df_original['Rendimiento Real']
        
        print(f"✅ Features preparadas: {len(self.feature_names)} variables")
        return X, y
    
    def optimize_multiple_models(self, X, y):
        """Optimiza múltiples algoritmos"""
        print("🚀 Optimizando múltiples algoritmos...")
        
        # configurar validación temporal
        tscv = TimeSeriesSplit(n_splits=5)
        
        # configurar preprocessor
        numeric_features = [f for f in self.feature_names if f != 'mapper_cluster']
        categorical_features = ['mapper_cluster']
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ],
            remainder='drop'
        )
        
        models_to_test = {}
        
        # 1. Random Forest (fine-tuned)
        print("🌳 Optimizando Random Forest...")
        rf_param_grid = {
            'regressor__n_estimators': [325, 350, 375],
            'regressor__max_depth': [11, 12, 13],
            'regressor__min_samples_split': [9, 10, 11],
            'regressor__min_samples_leaf': [1, 2],
            'regressor__max_features': ['sqrt']
        }
        
        rf_pipeline = Pipeline([
            ('preproc', self.preprocessor),
            ('regressor', RandomForestRegressor(random_state=42))
        ])
        
        rf_grid = GridSearchCV(rf_pipeline, rf_param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
        rf_grid.fit(X, y)
        models_to_test['RandomForest'] = rf_grid.best_estimator_
        self.best_params['RandomForest'] = rf_grid.best_params_
        
        self.best_models = models_to_test
        print(f"✅ Modelos optimizados: {list(models_to_test.keys())}")
        return models_to_test
    
    def create_ensemble_model(self, X, y):
        """Crea un modelo ensemble votante"""
        print("🤝 Creando modelo ensemble...")
        
        if len(self.best_models) >= 2:
            # crear ensemble con los mejores modelos
            estimators = [(name, model) for name, model in self.best_models.items()]
            
            self.ensemble_model = VotingRegressor(
                estimators=estimators,
                n_jobs=-1
            )
            
            self.ensemble_model.fit(X, y)
            print(f"✅ Ensemble creado con {len(estimators)} modelos")
        else:
            # usar el mejor modelo individual
            self.ensemble_model = list(self.best_models.values())[0]
            print("✅ Usando mejor modelo individual")
        
        return self.ensemble_model
    
    def evaluate_all_models(self, X, y):
        """Evalúa todos los modelos"""
        print("📊 Evaluando todos los modelos...")
        
        results = {}
        
        # evaluar modelos individuales
        for name, model in self.best_models.items():
            y_pred = model.predict(X)
            
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            mape = np.mean(np.abs((y - y_pred) / y)) * 100
            
            results[name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'MAPE': mape
            }
        
        # evaluar ensemble
        if self.ensemble_model:
            y_pred_ensemble = self.ensemble_model.predict(X)
            
            mse = mean_squared_error(y, y_pred_ensemble)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, y_pred_ensemble)
            r2 = r2_score(y, y_pred_ensemble)
            mape = np.mean(np.abs((y - y_pred_ensemble) / y)) * 100
            
            results['Ensemble'] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'MAPE': mape
            }
        
        # mostrar resultados
        print("\n📈 RESULTADOS DE TODOS LOS MODELOS:")
        print("-" * 60)
        for model_name, metrics in results.items():
            print(f"{model_name}:")
            for metric, value in metrics.items():
                print(f"   {metric}: {value:.4f}")
            print()
        
        # encontrar el mejor modelo
        best_model_name = max(results.keys(), key=lambda x: results[x]['R²'])
        best_r2 = results[best_model_name]['R²']
        
        print(f"🏆 MEJOR MODELO: {best_model_name} (R² = {best_r2:.4f})")
        
        # mostrar hiperparámetros del mejor modelo
        if best_model_name in self.best_params:
            print(f"\n🎯 HIPERPARÁMETROS DEL MEJOR MODELO ({best_model_name}):")
            print("-" * 50)
            for param, value in self.best_params[best_model_name].items():
                print(f"   {param}: {value}")
            print()
        
        self.metrics = results
        return results, best_model_name
    
    def generate_predictions_csv(self, X, y, best_model_name, output_path='final_presentation_data/pred_vs_real_advanced.csv'):
        """Genera CSV con predicciones del mejor modelo"""
        print("📄 Generando CSV con predicciones avanzadas...")
        
        # usar el mejor modelo para predicciones
        if best_model_name == 'Ensemble':
            best_model = self.ensemble_model
        else:
            best_model = self.best_models[best_model_name]
        
        y_pred = best_model.predict(X)
        
        # crear DataFrame para export
        results_df = pd.DataFrame({
            'Year': self.df_original['Year'],
            'Month': self.df_original['Month'],
            'y_real': y,
            'y_pred': y_pred
        })
        
        # agregar índice
        results_df.reset_index(inplace=True)
        
        # guardar CSV
        results_df.to_csv(output_path, index=False)
        
        print(f"✅ CSV guardado en: {output_path}")
        print(f"📊 Registros exportados: {len(results_df)}")
        
        return results_df

def main():
    """Función principal"""
    print("🚀 INICIANDO OPTIMIZACIÓN AVANZADA DEL MODELO")
    print("=" * 70)
    
    # crear modelo
    model = AdvancedFuelPredictionModel()
    
    # paso 1: cargar y preparar datos con feature engineering
    model.load_and_prepare_data()
    
    # paso 2: crear clusters mapper avanzados
    model.create_advanced_mapper_clusters()
    
    # paso 3: remover outliers
    model.remove_outliers()
    
    # paso 4: preparar features
    X, y = model.prepare_features()
    
    # paso 5: optimizar múltiples modelos
    model.optimize_multiple_models(X, y)
    
    # paso 6: crear ensemble
    model.create_ensemble_model(X, y)
    
    # paso 7: evaluar todos los modelos
    results, best_model_name = model.evaluate_all_models(X, y)
    
    # paso 8: generar CSV con el mejor modelo
    results_df = model.generate_predictions_csv(X, y, best_model_name)
    
    # guardar los mejores parámetros en un archivo
    import json
    import pickle
    
    best_params_file = 'best_model_params.json'
    with open(best_params_file, 'w') as f:
        json.dump({
            'best_model': best_model_name,
            'best_r2': results[best_model_name]['R²'],
            'best_params': model.best_params.get(best_model_name, {}),
            'all_results': results
        }, f, indent=2)
    
    # obtener opciones categóricas disponibles
    categorical_columns = ['Id Región', 'Cliente', 'Division', 'BL', 'Mercancía', 'Conductor', 'Vehículo_Corto']
    categorical_options = {}
    for col in categorical_columns:
        unique_values = sorted(model.df_original[col].dropna().unique().astype(str).tolist())
        categorical_options[col] = unique_values
    
    # guardar TODOS los componentes necesarios para predicciones
    model_components = {
        'best_model': model.best_models[best_model_name] if best_model_name != 'Ensemble' else model.ensemble_model,
        'feature_names': model.feature_names,
        'ohe_mapper': model.ohe,  # OneHotEncoder para mapper
        'pca_mapper': model.pca,  # PCA para mapper  
        'mapper_graph': model.graph,  # Grafo del mapper
        'reference_stats': {
            'rendimiento_real_median': model.df_original['Rendimiento Real'].median(),
            'rendimiento_real_std': model.df_original['Rendimiento Real'].std(),
            'cluster_size_median': model.df_original['cluster_size'].median()
        },
        'categorical_columns': categorical_columns,
        'categorical_options': categorical_options,  # opciones para dropdowns
        'numeric_features': [f for f in model.feature_names if f != 'mapper_cluster'],
        'categorical_features': ['mapper_cluster']
    }
    
    model_file = 'trained_model_complete.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model_components, f)
    
    print(f"💾 Modelo completo guardado en: {model_file}")
    print("\nComponentes guardados:")
    print("- Modelo RandomForest optimizado")
    print("- Feature names")
    print("- OneHotEncoder para mapper")
    print("- PCA para mapper")
    print("- Grafo del mapper")
    print("- Estadísticas de referencia")
    print("- Columnas categóricas y sus opciones")
    
    print("\n" + "=" * 70)
    print("🎉 ¡OPTIMIZACIÓN AVANZADA COMPLETADA!")
    print(f"🏆 Mejor R²: {results[best_model_name]['R²']:.4f}")
    print(f"🎯 Objetivo alcanzado: {'✅ SÍ' if results[best_model_name]['R²'] >= 0.75 else '❌ NO'}")
    print(f"💾 Parámetros guardados en: {best_params_file}")
    print("=" * 70)
    
    return model, results_df, results, best_model_name

if __name__ == "__main__":
    model, results_df, results, best_model_name = main()