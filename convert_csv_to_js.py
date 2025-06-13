import pandas as pd
import json

def convert_csv_to_js():
    """Convierte el CSV de predicciones a formato JavaScript"""
    
    # Leer el CSV
    print("📖 Leyendo CSV...")
    df = pd.read_csv('final_presentation_data/pred_vs_real_advanced.csv')
    
    # Convertir a formato JavaScript
    print("🔄 Convirtiendo datos...")
    data = []
    for _, row in df.iterrows():
        data.append({
            'year': int(row['Year']),
            'month': int(row['Month']),
            'yReal': float(row['y_real']),
            'yPred': float(row['y_pred'])
        })
    
    # Guardar como archivo JavaScript
    js_content = f"""// Datos de predicciones generados automáticamente
// Modelo: RandomForest Optimizado (R² = 0.6591)
// Total de registros: {len(data)}

const PREDICTION_DATA = {json.dumps(data, indent=2)};

// Función para obtener los datos (reemplaza el fetch del CSV)
function getPredictionData() {{
    console.log('✅ Usando datos embebidos:', PREDICTION_DATA.length, 'registros');
    return PREDICTION_DATA;
}}
"""
    
    output_file = 'final_presentation_data/prediction_data.js'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(js_content)
    
    print(f'✅ Convertido: {len(data)} registros a JavaScript')
    print(f'📁 Archivo creado: {output_file}')
    print(f'📊 Rango de años: {min(d["year"] for d in data)} - {max(d["year"] for d in data)}')
    
    return data

if __name__ == "__main__":
    convert_csv_to_js()