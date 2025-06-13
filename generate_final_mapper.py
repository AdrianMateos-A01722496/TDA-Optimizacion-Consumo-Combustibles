# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import kmapper as km
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

def generate_mapper_visualization():
    print("📊 Generando visualización del Mapper final...")

    # Cargar los datos originales
    print("📂 Cargando datos originales...")
    df_edenred = pd.read_excel('data/df_21_24_clean.xlsx')

    # Preparar variables categóricas
    print("🔧 Preparando variables categóricas...")
    categorical_cols = [
        'Id Región', 'Cliente', 'Division', 'BL',
        'Mercancía', 'Vehículo_Corto','Conductor'
    ]
    for c in categorical_cols:
      df_edenred[c] = df_edenred[c].astype(str)
    
    df_categorical = pd.get_dummies(df_edenred[categorical_cols])


    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    df_categorical = ohe.fit_transform(df_edenred[categorical_cols])
    # Aplicar PCA con 2 componentes
    print("📐 Aplicando PCA...")
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(df_categorical)

    # Configurar Mapper con nuevos parámetros
    print("🔧 Configurando Mapper...")
    mapper = km.KeplerMapper(verbose=1)

    # Crear la proyección
    print("🎯 Creando proyección...")
    lens = mapper.fit_transform(pca_result, projection=pca_result)

    # Crear el grafo
    print("🔄 Creando grafo...")
    graph = mapper.map(
        lens,
        df_categorical,
        cover=km.Cover(n_cubes=5, perc_overlap=0.5),
        clusterer=DBSCAN(eps=2.5, min_samples=100)
    )

    # Preparar datos para la visualización
    print("🎨 Preparando visualización...")
    color_array = df_edenred['Rendimiento Real'].values

    # Crear tooltips con información relevante (simplificado)
    tooltips_array = df_edenred.index.astype(str).values

    # Generar HTML
    print("💾 Generando HTML...")
    html = mapper.visualize(
        graph,
        path_html="final_presentation_data/edenred_mapper_final.html",
        title="Mapper Topológico - Edenred",
        color_function=color_array,
        color_function_name="Rendimiento Real",
        custom_tooltips=tooltips_array
    )

    print("✅ ¡Visualización generada con éxito!")

if __name__ == "__main__":
    generate_mapper_visualization()