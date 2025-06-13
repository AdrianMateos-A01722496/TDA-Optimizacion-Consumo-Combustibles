import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import kmapper as km
import pickle
import warnings
warnings.filterwarnings('ignore')

def regenerate_mapper():
    print("🔄 Regenerando grafo mapper con configuración optimizada...")
    
    # 1. Cargar datos
    print("📂 Cargando datos originales...")
    df_original = pd.read_excel('data/df_21_24_clean.xlsx')
    print(f"✅ Datos cargados: {len(df_original)} registros")
    
    # 2. Preparar variables categóricas
    print("🔧 Preparando variables categóricas...")
    categorical_cols = [
        'Id Región', 'Cliente', 'Division', 'BL',
        'Mercancía', 'Vehículo_Corto', 'Conductor'
    ]
    
    # Asegurar que todas las columnas categóricas sean strings y no tengan NaN
    df_original[categorical_cols] = df_original[categorical_cols].fillna('NULO')
    for c in categorical_cols:
        df_original[c] = df_original[c].astype(str)
    
    # 3. Aplicar OneHotEncoder
    print("🎯 Aplicando OneHotEncoder...")
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_cat_ohe = ohe.fit_transform(df_original[categorical_cols])
    print(f"✅ Dimensiones después de OneHotEncoder: {X_cat_ohe.shape}")
    
    # 4. Aplicar PCA
    print("📐 Aplicando PCA...")
    pca = PCA(n_components=2, random_state=42)
    lens = pca.fit_transform(X_cat_ohe)
    print(f"✅ Dimensiones después de PCA: {lens.shape}")
    
    # 5. Configurar y aplicar Mapper
    print("🗺️ Configurando Mapper...")
    mapper = km.KeplerMapper(verbose=1)
    
    # Configuración optimizada
    coverer = km.Cover(n_cubes=5, perc_overlap=0.5)
    clusterer = DBSCAN(eps=2.5, min_samples=100)
    
    print("🔄 Generando grafo...")
    graph = mapper.map(
        lens,
        X_cat_ohe,
        cover=coverer,
        clusterer=clusterer
    )
    
    # 6. Preparar datos para guardar
    print("💾 Preparando datos para guardar...")
    categorical_options = {
        col: df_original[col].unique().tolist()
        for col in categorical_cols
    }
    
    # 7. Guardar componentes
    print("📦 Guardando componentes...")
    components = {
        'ohe_mapper': ohe,
        'pca_mapper': pca,
        'mapper_graph': graph,
        'categorical_columns': categorical_cols,
        'categorical_options': categorical_options
    }
    
    with open('trained_model_complete.pkl', 'wb') as f:
        pickle.dump(components, f)
    
    print("✅ Grafo mapper regenerado y guardado exitosamente")
    print(f"📊 Número de nodos en el grafo: {len(graph['nodes'])}")
    
    # 8. Generar visualización
    print("🎨 Generando visualización...")
    color_array = df_original['Rendimiento Real'].values
    tooltips_array = df_original.index.astype(str).values
    
    html = mapper.visualize(
        graph,
        path_html="final_presentation_data/edenred_mapper_final.html",
        title="Mapper Topológico - Edenred",
        color_function=color_array,
        color_function_name="Rendimiento Real",
        custom_tooltips=tooltips_array
    )
    
    print("✅ Visualización generada exitosamente")

if __name__ == "__main__":
    regenerate_mapper() 