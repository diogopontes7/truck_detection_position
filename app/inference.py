# import cv2
# import inference
import ast
import os
from turtle import color
from inference_sdk import InferenceHTTPClient
import base64
import cv2
from PIL import Image, ImageDraw, ImageFont
from matplotlib import patches
import matplotlib.pyplot as plt
from io import BytesIO
from dotenv import load_dotenv
import numpy as np
import requests
import json
from datetime import datetime

# Ainda estamos em fase de testagem, por agora deixa estar assim mas dps passar para um app flask para isto puder ser utilizado
# Meter isto numa função, principalmente a parte de ir buscar o modelo
max_right_x = 0  # Inicializa o max_right_x como 0
min_left_x = float("inf")  # Inicializa o min_left_x como infinito


def create_model(img_path):
    """
    Função para criar o modelo de inferência.
    """
    load_dotenv()
    ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
    
    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com", 
        api_key=ROBOFLOW_API_KEY
    )
    
    result = client.run_workflow(
        workspace_name="summer-n8dui",
        workflow_id="detect-count-and-visualize-7",
        images={"image": img_path},
        use_cache=True,
    )
    
    return result
# Assim vai buscar o path absoluto
script_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(script_dir, "examples", "0013247.jpg")
with open(img_path, "rb") as f:
    image_bytes = f.read()

image = Image.open(img_path)
draw = ImageDraw.Draw(image)

x1_maia = 351        # ponto esquerdo (bottom-left of polygon)
x2_maia = 2075.62    # ponto direito (top-right of polygon)
y1_maia = 569        # ponto base esquerdo (top-left of polygon)
y2_maia = 1080       # ponto base direito (bottom-right of polygon)

# pontos Maia
pontos_maia = [
    (1730, 569), # ponto em cima esquerdo
    (351, 951), # ponto em baixo esquerdo
    (1722, 1080), # ponto em baixo direito
    (2075.62, 601.69), # ponto em cima direito
]

pontos_maia_teste = [
    (1730, y1_maia), # ponto esquerdo
    (x1_maia, 951), # ponto direito
    (1722, y2_maia), # ponto base esquerdo
    (x2_maia, 601.69), # ponto base direito
]


def inference():

    try:
        result = create_model(img_path)
        detections = result[0]["predictions"]["predictions"]  # Vai buscar as previsões do resultado
        print(detections)  # verificar se vai buscar o certo
        #font = ImageFont.truetype("arial.ttf", 40)

        
        #front_detection = None  # Inicializa front_detection como None

        for bounding_box in detections:

            x = bounding_box["x"] # CEntro x, tem que estar alinhado a este para verificar se está dentro do limite
            y = bounding_box["y"] # Centro y
            width = bounding_box["width"] # largura de cada caixa encontrada
            height = bounding_box["height"]
            class_name = bounding_box["class"] # isto é importante para perceber cada classe que encontramos
            confidence = bounding_box["confidence"]

            x1 = x - width / 2  # esquerda
            x2 = x + width / 2  # direita
            y1 = y - height / 2  # topo
            y2 = y + height / 2  # base
    # Nao posso fazer isto para todas as classes que nao fazia sentido
            # if x1 > x1_maia:
            #     print(f"Bounding box {class_name} está fora do limite esquerdo ({x1_maia})")
            # Desenha a caixa da label com o seu nome e confiança
            label = f"{class_name}: {confidence:.2f}"
            draw.text((x1, y1 - 10), label, fill="red", font=None)  

            # Draw rectangle
            draw.rectangle((x1, y1, x2, y2), outline="red", width=2)          
    except (IndexError, KeyError) as e:
        print(f"Erro ao aceder as previsões {e}")
    draw.polygon(pontos_maia, outline="blue", width=6)
            
    return detections # Isto devolve uma lista
# A lógica de inferencia é mais ou menos até aqui, dps disto é a definição do poligono e da linha que vai ser desenhada
# Aqui o que se vai fazer basicamente é ver as labels todas da frente do camião, vamos ver as coorde

def check_vertical_alignment_front():
            detections = inference()  # Chama a função de inferência para obter as detecções
            
            front = [l for l in detections if l["class"] == "front"]
            visible_corner = [l for l in detections if l["class"] == "visible_corner"]
            invisible_corner = [l for l in detections if l["class"] == "invisible_corner"]
            
            x_coords = [l["x"] for l in front + visible_corner + invisible_corner] #coordenadas x de cada label
            
            print(f"Coordenadas x das labels: {x_coords}")

            #Temos de calcular a média e verificar se tem u grande desvio
            mean_x = float(np.mean(x_coords))
            deviations = [np.abs(x-mean_x) for x in x_coords] # Para cada x., vamos ver o desvio absoluto da media menos o seu valor
            max_deviation = float(np.max(deviations))  # Maior desvio absoluto
            
            #tolerancia = 0.05 * int(detections[0]["predictions"]["width"])  # Tolerância em pixels para considerar que as labels estão alinhadas
            tolerancia = 0.05 * 1000
            alinhado = bool(max_deviation <= tolerancia)
            print(f"Desvio máximo: {max_deviation}, Tolerância: {tolerancia}, Alinhado: {alinhado}")
            
            return alinhado,x_coords # Isto devolve um bool alinhado e valores das coordenadas x



            
            if class_name == "front":
                front_detection = {
                    "x1": x1,
                    "x2": x2,
                    "y2": y2,
                }  # define a linha de baixo da label do front, o x1 é o ponto esquerdo e o x2 é o ponto direito, y2 é a base da label do front
                print(f"x1: {x1}, x2: {x2}, y2: {y2}")
                if x1 < x1_maia:
                    print(f"Bounding box {class_name} está fora do limite esquerdo ({x1_maia})")
                if x2 > x2_maia:
                    print(f"Bounding box {class_name} está fora do limite direito ({x2_maia})")
                if y2 > y2_maia:
                    print(f"Bounding box {class_name} está fora do limite superior ({y2_maia})")

            raio = 5 # raio do círculo vermelho que vamos desenhar no centro de cada caixa label
            bbox = (x - raio, y - raio, x + raio, y + raio)
            draw.ellipse(bbox, fill="red")
            # Find the furthermost right point of visible_corner
            if class_name == "visible_corner":
                max_right_x = bbox[2]  # x2 é o ponto direito da caixa, que é o maior ponto direito de todas as caixas encontradas

            # if class_name == "visible_corner":
            #     min_right_x = min(min_right_x, x2)

            # Find the furthermost left point of invisible_corner
            if class_name == "invisible_corner":
                min_left_x = bbox[0]  # x1 é o ponto esquerdo da caixa, que é o menor ponto esquerdo de todas as caixas encontradas

            # Draw label with class name and confidence
            label = f"{class_name}: {confidence:.2f}"
            draw.text((x1, y1 - 10), label, fill="red", font=None)  # Use None for default font

            if front_detection and (max_right_x > 0 or min_left_x < float("inf")):
                line_y = front_detection["y2"] + 10 # 10 pixels abaixo da base da label do front

                # Determina a linha, de acordo se encontra a label ou não
                line_start_x = (
                    min_left_x if min_left_x < float("inf") else front_detection["x1"]
                )  # Se nao for inf, signoifoca que foi encontrado a label e existe um min_left do invisible_corner
                line_end_x = (
                    max_right_x if max_right_x > 0 else front_detection["x2"]
                )  # Mesma logica para o visible_corner, se for maior que 0, significa que foi encontrado a label e existe um max_right do visible_corner
                # line_end_x = min_right_x if min_right_x < float('inf') else front_detection['x2']

                # se nao encontramos, usamos o ponto direito e esquerdo da label do front


            draw.line((line_start_x, line_y, line_end_x, line_y), fill="green", width=2)

            # PEnsar em usar isto como a referencia da label ou seja, a linha chega só até a esse ponto central em cada label
            ## Serve para desenhar um círculo vermelho no centro do bounding box
            # raio = 5
            # bbox = (x - raio, y - raio, x + raio, y + raio)
            # draw.ellipse(bbox, fill="red")


# Display the image
#image.show()

# Nao sei se isto faz muito sentido, prefiro mais a lógica das linhas
def check_behind_side():
    
    # Verificar se a frente está atrás da side
    side = [l for l in inference() if l["class"] == "side"]
    front = [l for l in inference() if l["class"] == "front"]
    visible = [l for l in inference() if l["class"] == "visible_corner"]
    invisible = [l for l in inference() if l["class"] == "invisible_corner"]
    
    # Pegar no lado direito da side
    
    side_right_edges = [l['x'] + l['width']/2 for l in side]  # Right edge = center_x + half_width
    max_side_right = max(side_right_edges)

    front_components = front + visible + invisible
    front_left_edges = [l['x'] - l['width']/2 for l in front_components]  # Left edge = center_x - half_width
    min_front_left = min(front_left_edges)
    # Distancia entre o lado direito da side e o lado esquerdo da front, para ver se a front está atrás da side
    distance = min_front_left - max_side_right + 5000 
    distance = float(distance)  # Convert to float for consistency
    tolerancia = 0.005 * 1000  # Tolerância em pixels para considerar que a front está atrás da side
    is_behind_side = bool(distance >= tolerancia)  
    
    return is_behind_side, distance 

#Verificar a posição do camiao de acordo com estas funçoes que definimos
def check_truck_position():
    
    alinhado_vertical, x_coords = check_vertical_alignment_front()
    is_behind_side, distance = check_behind_side()
    print(f"Alinhado verticalmente: {alinhado_vertical}")
    print(f"Está atrás da side: {is_behind_side}, Distância: {distance}")
    
    return {
        "alinhado_vertical": bool(alinhado_vertical),
        "is_behind_side": bool(is_behind_side),
        "distance": float(distance),
        "x_coords": x_coords
    }
# Tenho que mudar esta função para ser mais simples de perceber e mais fácil de visualizar
def visualizar_alinhamento():
    """
    Simple visualization function to show truck alignment results
    """
    # Load and display the image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    
    # Get detections and alignment results
    detections = inference()
    truck_status = check_truck_position()
    
    # Define colors for different classes
    class_colors = {
        'front': 'red',
        'side': 'blue', 
        'visible_corner': 'green',
        'invisible_corner': 'orange'
    }
    
    # Draw bounding boxes for each detection
    for detection in detections:
        x = detection["x"]
        y = detection["y"]
        width = detection["width"]
        height = detection["height"]
        class_name = detection["class"]
        confidence = detection["confidence"]

        # Calculate box corners
        x1 = x - width / 2
        x2 = x + width / 2
        y1 = y - height / 2
        y2 = y + height / 2
        
        # Get color for this class
        color = class_colors.get(class_name, 'yellow')
        
        # Draw rectangle
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        # Add label
        ax.text(x1, y1-5, f'{class_name}: {confidence:.2f}', 
               color=color, fontsize=10, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
    
    # Draw vertical alignment line if aligned
    if truck_status['alinhado_vertical'] and truck_status['x_coords']:
        mean_x = np.mean(truck_status['x_coords'])
        ax.axvline(x=mean_x, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax.text(mean_x + 10, 50, 'Vertical Alignment Line', color='red', fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    #Isto para visualizaçao, nao interessa tanto mas dava jeito mostrar
    # Add status text
    vertical_status = "✓ ALIGNED" if truck_status['alinhado_vertical'] else "✗ NOT ALIGNED"
    behind_status = "✓ BEHIND SIDE" if truck_status['is_behind_side'] else "✗ NOT BEHIND SIDE"
    
    status_text = f"VERTICAL: {vertical_status}\n"
    status_text += f"POSITION: {behind_status}\n"
    status_text += f"Distance: {truck_status['distance']:.2f}px"
    
    # Overall status color
    overall_status = truck_status['alinhado_vertical'] and truck_status['is_behind_side']
    status_color = 'green' if overall_status else 'red'
    main_status = "TRUCK PROPERLY ALIGNED" if overall_status else "TRUCK NEEDS ADJUSTMENT"
    
    # Add status box
    ax.text(10, 10, f"{main_status}\n\n{status_text}", 
           color=status_color, fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
    
    ax.set_title(f"Truck Alignment Analysis: {os.path.basename(img_path)}", fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# def on_hover(event):
#     if event.inaxes:
#         x, y = int(event.xdata), int(event.ydata)
#         plt.title(f'Coordinates: ({x}, {y})')
#         plt.draw()

# # Convert PIL image to numpy array for matplotlib
# img_array = np.array(image)

# # Create matplotlib figure and display image
# fig, ax = plt.subplots(figsize=(12, 8))
# ax.imshow(img_array)
# ax.set_title('Hover over image to see coordinates')

# # Connect the hover event
# fig.canvas.mpl_connect('motion_notify_event', on_hover)

# # Show the plot
# plt.show()

def main():
    """
    Main function to test the inference and alignment checking functionality.
    """
    print("Starting truck detection and alignment checking...")
    
    # Test the inference function
    print("\n1. Running inference...")
    detections = inference()
    
    # Test the truck position analysis with API notification
    print("\n2. Analyzing truck position...")
    
    # Option 1: Just check without API notification
    # truck_status = check_truck_position()
    
    # Option 2: Check and send to API (uncomment and set your webhook URL)
    # webhook_url = "https://your-api-endpoint.com/webhook"
    # truck_status = check_and_notify(webhook_url=webhook_url, send_notification=True)
    
    # Option 3: Check and send to API using environment variable
    #truck_status = check_and_notify(send_notification=False)  # Set to True to enable API calls
    
    # Show results
    print(f"\nTotal detections found: {len(detections) if detections else 0}")
    print("\nTruck Status Analysis:")
    #print(f"  - Vertically aligned: {truck_status['alinhado_vertical']}")
    #print(f"  - Behind side panel: {truck_status['is_behind_side']}")
    #print(f"  - Distance to side: {truck_status['distance']:.2f}px")
    #print(f"  - Overall status: {truck_status.get('overall_status', 'Unknown')}")
    
    # Print summary of detected classes
    if detections:
        classes_found = {}
        for detection in detections:
            class_name = detection["class"]
            if class_name in classes_found:
                classes_found[class_name] += 1
            else:
                classes_found[class_name] = 1
        
        print("\nDetected classes:")
        for class_name, count in classes_found.items():
            print(f"  - {class_name}: {count} instance(s)")
    
    # Show visualization
    print("\n3. Displaying visualization...")
    visualizar_alinhamento()
    
    print("\nProcessing complete!")
    

if __name__ == "__main__":
    main()

#output_path = "output_image.jpg"
#image.save(output_path)
#vprint(f"Image saved to {output_path}")    plt.tight_layout()
#plt.show()

