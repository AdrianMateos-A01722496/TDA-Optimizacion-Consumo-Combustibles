import pandas as pd
import json

# datos de consumo por zona
consumo_zona_data = {
    'labels': ['REFORMA', 'COATZACOALCOS', 'PARAISO', 'CHIAPAS', 'TAMPICO', 
               'CIUDAD DEL CARMEN', 'VILLAHERMOSA', 'HECTOR RIVERA', 'VERACRUZ'],
    'data': [51105.50, 39024.41, 34229.20, 25283.35, 21958.44, 
             14797.88, 10000.00, 10000.00, 8731.75]
}

# datos de métricas por tipo de combustible
fuel_metrics_data = {
    'kgCo2': {
        'labels': ['MOBIL EXTRA', 'GULF REGULAR', 'SHELL SUPER', 'REPSOL DIESEL', 
                  'REPSOL EFITEC 87', 'BP DIESEL', 'MAGNA', 'BP REGULAR', 'G SUPER', 
                  'G DIESEL', 'DIESEL'],
        'data': [0.341705, 0.367190, 0.396292, 0.691323, 1.027113, 
                1.177252, 2.035523, 2.130832, 4.176725, 8.200128, 11.144934]
    },
    'precioUnitario': {
        'labels': ['MOBIL EXTRA', 'MAGNA', 'GULF REGULAR', 'G SUPER', 'REPSOL EFITEC 87', 
                  'BP REGULAR', 'DIESEL', 'SHELL SUPER', 'BP DIESEL', 'G DIESEL', 
                  'REPSOL DIESEL'],
        'data': [20.147472, 21.128342, 21.766650, 22.066280, 22.177849, 22.257353, 
                23.058347, 23.228923, 24.017860, 24.047871, 24.228060]
    },
    'rendimiento': {
        'labels': ['MOBIL EXTRA', 'SHELL SUPER', 'GULF REGULAR', 'BP REGULAR', 'MAGNA', 
                  'BP DIESEL', 'G SUPER', 'REPSOL DIESEL', 'REPSOL EFITEC 87', 'DIESEL', 
                  'G DIESEL'],
        'data': [7.467500, 7.356667, 7.050000, 6.265247, 5.735860, 5.510000, 5.249593, 
                5.213333, 5.162075, 5.062617, 3.873333]
    },
    'cantidadMercancia': {
        'labels': ['G DIESEL', 'DIESEL', 'BP DIESEL', 'REPSOL DIESEL', 'BP REGULAR', 
                  'REPSOL EFITEC 87', 'G SUPER', 'MAGNA', 'MOBIL EXTRA', 'GULF REGULAR', 
                  'SHELL SUPER'],
        'data': [83.131485, 72.402175, 72.394301, 69.533627, 68.688999, 67.493997, 
                67.325964, 60.542106, 55.845000, 55.207483, 48.355500]
    }
}

# guardar los archivos json
with open('consumo_zona_data.json', 'w') as f:
    json.dump(consumo_zona_data, f)

with open('fuel_metrics_data.json', 'w') as f:
    json.dump(fuel_metrics_data, f)

print("Archivos JSON generados exitosamente:")
print("- consumo_zona_data.json")
print("- fuel_metrics_data.json")