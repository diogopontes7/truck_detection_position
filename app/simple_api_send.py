from flask import Flask, request, jsonify
import requests

from inference import check_truck_position, inference


app = Flask(__name__)

@app.route('/')
def home():
    return "Home"


@app.route('/get-position')
def get_position():
    detections = inference()
    truck_status = check_truck_position()
    
    data = {
        "detections": detections,
        "truck_status": truck_status
    }
    
    # Isto simula o envio para a API externa que queremos testar, neste caso, a nossa própria API, dps tinhamos que mudar para a API que nos foi fornecida
    # Para o llm, e dps o llm vai enviar outro tipo de resposta, com a sugestão para a melhoria do posicionamento do camião
    response = requests.post('http://localhost:5000/verificar-alinhamento', json=data)
    result = response.json()
    
    return jsonify(result),response.status_code
    
    
# Recebemos os dados das funções definidas, posteriormente o llm onde enviamos os dados vai receber esses dados e vai enviar certas sugestões para resolver esse problema
@app.route('/verificar-alinhamento', methods=['POST'])
def verificar_alinhamento():
    data = request.get_json() #Vai receber os valores necessários para verificar o alinhamento
    
    message = "Recebemos os dados da funções inference e check_truck_position"
    
    return jsonify({"received": data, "message": message}), 201

if __name__ == "__main__":
    app.run(debug=True)