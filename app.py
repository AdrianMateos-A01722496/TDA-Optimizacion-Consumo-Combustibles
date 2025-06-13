from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import json
import sys
import os

# agregar el directorio actual al path para importar predictor_model
sys.path.append(os.getcwd())

try:
    from final_predictor import get_final_model, predict_efficiency_final, get_categorical_options_final
except ImportError as e:
    print(f"Error importing predictor_model: {e}")
    # crear funciones mock para desarrollo
    def get_categorical_options_final():
        return {
            'Id Región': ['1000', '2000', '3000'],
            'Cliente': ['02125-031', '02125-032', '02125-033'],
            'Division': ['DAI', 'DBZ', 'DCC'],
            'BL': ['IGWC', 'IGWD', 'IGWE'],
            'Mercancía': ['MAGNA', 'G SUPER', 'BP REGULAR'],
            'Conductor': ['CONDUCTOR_01', 'CONDUCTOR_02', 'CONDUCTOR_03'],
            'Vehículo_Corto': ['VEH_001', 'VEH_002', 'VEH_003']
        }
    
    def predict_efficiency_final(categorical_data, recorrido, rendimiento_esperado):
        return {
            'rendimiento_real_predicho': 7.5,
            'cluster_asignado': '9',
            'recorrido': recorrido,
            'rendimiento_esperado': rendimiento_esperado,
            'cluster_distance': 0.123,
            'features_utilizadas': 17,
            'modelo': 'RandomForest Optimizado',
            'r2_score': 0.7098
        }

app = Flask(__name__)
CORS(app)  # permitir CORS para todas las rutas

@app.route('/')
def home():
    return "API de Predicción de Rendimiento de Combustible"

@app.route('/api/options', methods=['GET'])
def get_options():
    """Endpoint para obtener las opciones categóricas disponibles"""
    try:
        options = get_categorical_options_final()
        return jsonify({
            'success': True,
            'data': options
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Validar campos requeridos
        required_fields = ['recorrido', 'rendimiento_esperado', 'categorical_data']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Campo requerido faltante: {field}'
                }), 400
        
        # Validar que los campos numéricos sean números
        try:
            recorrido = float(data['recorrido'])
            rendimiento_esperado = float(data['rendimiento_esperado'])
        except ValueError:
            return jsonify({
                'success': False,
                'error': 'Los campos recorrido y rendimiento_esperado deben ser números'
            }), 400
        
        # Validar que los valores sean positivos
        if recorrido <= 0 or rendimiento_esperado <= 0:
            return jsonify({
                'success': False,
                'error': 'Los valores de recorrido y rendimiento_esperado deben ser positivos'
            }), 400
        
        # Obtener datos categóricos
        categorical_data = data['categorical_data']
        
        # Hacer predicción
        try:
            result = predict_efficiency_final(categorical_data, recorrido, rendimiento_esperado)
            return jsonify({
                'success': True,
                'data': result
            })
        except Exception as e:
            print(f"Error en predicción: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Error en predicción: {str(e)}'
            }), 500
            
    except Exception as e:
        print(f"Error general: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Error general: {str(e)}'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Endpoint para verificar el estado de la API"""
    return jsonify({
        'success': True,
        'status': 'API funcionando correctamente',
        'model_loaded': 'model_instance' in globals()
    })

if __name__ == '__main__':
    print("Iniciando API de Predicción de Combustible...")
    print("Cargando modelo... (esto puede tomar unos minutos)")
    
    try:
        # cargar el modelo final al inicio
        model = get_final_model()
        print("Modelo final cargado exitosamente!")
    except Exception as e:
        print(f"Advertencia: No se pudo cargar el modelo: {e}")
        print("La API funcionará en modo mock")
    
    # ejecutar en puerto 5001 (5000 está ocupado por AirPlay)
    app.run(debug=True, host='0.0.0.0', port=5001) 