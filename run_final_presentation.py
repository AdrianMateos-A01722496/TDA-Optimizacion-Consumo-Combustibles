#!/usr/bin/env python3
"""
Script para ejecutar la presentación final con el modelo optimizado
Análisis Topológico de Consumo de Combustible - SLB México
"""

import os
import sys
import time
import subprocess
import threading
import webbrowser
from pathlib import Path
from app import app

def print_header():
    """Muestra el header de la aplicación"""
    print("=" * 70)
    print("🔬 Análisis Topológico de Consumo de Combustible - SLB México")
    print("=" * 70)

def check_requirements():
    """Verifica que todos los archivos necesarios estén presentes"""
    required_files = [
        'presentacion_final.html',
        'app.py',
        'final_predictor.py',
        'trained_model_complete.pkl',
        'final_presentation_data/pred_vs_real_advanced.csv',
        'data/df_21_24_clean.xlsx'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Archivos faltantes:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    print("✅ Todos los archivos necesarios están presentes")
    return True

def install_dependencies():
    """Instala las dependencias necesarias"""
    print("📦 Verificando dependencias...")
    try:
        # verificar si requirements.txt existe
        if os.path.exists('requirements.txt'):
            print("🔧 Instalando dependencias...")
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt', '--user'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Dependencias instaladas exitosamente")
            else:
                print("⚠️ Advertencia en instalación de dependencias:")
                print(result.stderr)
        else:
            print("⚠️ requirements.txt no encontrado, continuando...")
    except Exception as e:
        print(f"⚠️ Error instalando dependencias: {e}")

def test_model():
    """Prueba que el modelo final funcione correctamente"""
    print("🧪 Probando modelo final...")
    try:
        from final_predictor import FinalFuelPredictionModel
        
        # crear instancia del modelo
        model = FinalFuelPredictionModel()
        
        # hacer una predicción de prueba
        test_data = {
            'Id Región': '1000',
            'Cliente': '02125-031', 
            'Division': 'DAI',
            'BL': 'IGWC',
            'Mercancía': 'MAGNA',
            'Conductor': 'CONDUCTOR_01',
            'Vehículo_Corto': 'VEH_001'
        }
        
        result = model.predict_fuel_efficiency(test_data, 300, 8.5)
        
        if 'rendimiento_real_predicho' in result:
            print(f"✅ Modelo funcionando: Predicción = {result['rendimiento_real_predicho']} km/L")
            return True
        else:
            print("❌ Error en predicción del modelo")
            return False
            
    except Exception as e:
        print(f"❌ Error probando modelo: {e}")
        return False

def start_flask_server():
    """Inicia el servidor Flask en background"""
    print("🚀 Iniciando API Flask...")
    try:
        # cambiar al directorio actual
        env = os.environ.copy()
        env['PYTHONPATH'] = os.getcwd()
        
        # iniciar Flask
        process = subprocess.Popen([
            sys.executable, 'app.py'
        ], env=env, cwd=os.getcwd())
        
        print("✅ API iniciada en puerto 5000")
        return process
        
    except Exception as e:
        print(f"❌ Error iniciando API: {e}")
        return None

def open_presentation():
    """Abre la presentación en el navegador"""
    print("🎨 Iniciando presentación...")
    try:
        # obtener ruta absoluta del archivo HTML
        html_file = os.path.abspath('presentacion_final.html')
        file_url = f"file://{html_file}"
        
        print(f"🌐 Abriendo presentación: {file_url}")
        
        # abrir en navegador
        webbrowser.open(file_url)
        print("✅ Presentación abierta en el navegador")
        
        return file_url
        
    except Exception as e:
        print(f"❌ Error abriendo presentación: {e}")
        return None

def main():
    """Función principal"""
    print_header()
    
    # verificar archivos
    if not check_requirements():
        print("❌ No se puede continuar. Archivos faltantes.")
        return
    
    # instalar dependencias
    install_dependencies()
    
    # probar modelo
    if not test_model():
        print("❌ El modelo no funciona correctamente")
        return
    
    print("🔧 Configurando servidor...")
    
    # iniciar servidor Flask
    flask_process = start_flask_server()
    if not flask_process:
        print("❌ No se pudo iniciar el servidor")
        return
    
    # esperar que el servidor se inicie
    print("⏳ Esperando que la API se inicialice...")
    time.sleep(3)
    
    # abrir presentación
    presentation_url = open_presentation()
    if not presentation_url:
        print("❌ No se pudo abrir la presentación")
        return
    
    print()
    print("=" * 70)
    print("🎉 ¡Presentación lista!")
    print("📊 La serie temporal carga datos del modelo optimizado")
    print("🔮 La calculadora usa RandomForest")
    print("🗺️ El mapa topológico Mapper está integrado")
    print("=" * 70)
    print()
    print("📝 Instrucciones:")
    print("   • La presentación se abrió en tu navegador")
    print("   • Usa la calculadora para hacer predicciones en tiempo real")
    print("   • Si la API no responde, se usarán datos de ejemplo")
    print("   • Presiona Ctrl+C para detener el servidor")
    print()
    print(f"🌐 URL de la API: http://localhost:5001")
    print(f"📄 Archivo de presentación: presentacion_final.html")
    print()
    print("⏸️  Presiona Ctrl+C para detener...")
    
    try:
        # mantener el script corriendo
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Deteniendo servidor...")
        if flask_process:
            flask_process.terminate()
            flask_process.wait()
        print("✅ Servidor detenido")

if __name__ == "__main__":
    main()

if __name__ == '__main__':
    print("🚀 Iniciando servidor de presentación...")
    print("📊 Abre http://localhost:5001 en tu navegador")
    app.run(host='0.0.0.0', port=5001, debug=True) 