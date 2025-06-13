import pandas as pd
import json

# leer el archivo csv
df = pd.read_csv('umap_labels.csv')

# extraer las coordenadas x e y
x_coords = df.iloc[:, 0].tolist()
y_coords = df.iloc[:, 1].tolist()

# extraer los valores para las diferentes opciones de color
kg_co2 = df['KG CO2/km'].tolist()
mercancia = df['Mercancía'].tolist()
division = df['Division'].tolist()
bl = df['BL'].tolist()
cliente = df['Cliente'].tolist()

# crear el texto para el hover usando las columnas restantes
hover_text = []
for _, row in df.iloc[:, 2:].iterrows():
    # crear un texto formateado con los valores de cada columna
    text = '<br>'.join([f"{col}: {val}" for col, val in row.items()])
    hover_text.append(text)

# crear el diccionario con los datos
umap_data = {
    'x': x_coords,
    'y': y_coords,
    'kg_co2': kg_co2,
    'mercancia': mercancia,
    'division': division,
    'bl': bl,
    'cliente': cliente,
    'text': hover_text
}

# guardar como json
with open('umap_data.json', 'w') as f:
    json.dump(umap_data, f)

print("JSON file created successfully!") 