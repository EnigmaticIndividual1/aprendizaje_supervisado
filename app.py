from pathlib import Path

from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent

# Cargar el modelo guardado
with open(BASE_DIR / 'pipeline.pkl', 'rb') as archivo_modelo:
    modelo = pickle.load(archivo_modelo)

@app.route('/predecir', methods=['POST'])
def predecir():
    # Obtener los datos de la solicitud
    data = request.get_json()

    # Crear un DataFrame de pandas a partir del JSON
    input_data = pd.DataFrame([data])

    # Hacer la predicción usando el modelo que tiene el pipeline que hará la transformación
    prediccion = modelo.predict(input_data)
    
    # Devolver la predicción como JSON
    output = {'Survived': int(prediccion[0])}
    
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)
