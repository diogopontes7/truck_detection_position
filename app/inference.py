# import cv2
# import inference
import ast
import os
from inference_sdk import InferenceHTTPClient
import base64
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from io import BytesIO
from dotenv import load_dotenv
import numpy as np

# Ainda estamos em fase de testagem, por agora deixa estar assim mas dps passar para um app flask para isto puder ser utilizado
# Meter isto numa função, principalmente a parte de ir buscar o modelo
max_right_x = 0  # Inicializa o max_right_x como 0
min_left_x = float("inf")  # Inicializa o min_left_x como infinito

# FUNÇÃO QUE CHAMA O MODELO DO ROBOFLOW

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

## Escolher a imagem que queremos analisar 

img_path = os.path.join("examples", "0013247.jpg")
with open(img_path, "rb") as f:
    image_bytes = f.read()

image = Image.open(img_path)
draw = ImageDraw.Draw(image)

#image_pil = Image.open(img_path)
#image_np = np.array(image_pil)
#normalized_image = image_np.astype(np.float32) / 255.0


###########################################

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

###############################################

# Função para criar estas listas | Detections -> Vem da Inference()
def auxiliary_function(detections):
    front = [l for l in detections if l["class"] == "front"]
    visible_corner = [l for l in detections if l["class"] == "visible_corner"]
    invisible_corner = [l for l in detections if l["class"] == "invisible_corner"]
    side = [l for l in detections if l["class"] == "side"]
     
    return front, visible_corner, invisible_corner, side


def inference():

    try:
        result = create_model(img_path)
        detections = result[0]["predictions"]["predictions"]                                    # Detections é uma lista de dicionários, cada um representando uma detecção
        print(detections)  
        
        
        #font = ImageFont.truetype("arial.ttf", 40)
        #front_detection = None  # Inicializa front_detection como None

        for bounding_box in detections:

            x = bounding_box["x"]                                                               # CEntro x, tem que estar alinhado a este para verificar se está dentro do limite
            y = bounding_box["y"]                                                               # Centro y
            width = bounding_box["width"]                                                       # largura de cada caixa encontrada
            height = bounding_box["height"]                                                     # altura de cada caixa encontrada                              
            class_name = bounding_box["class"]                                                  # isto é importante para perceber cada classe que encontramos
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
        print(f"Erro ao acessar as previsões {e}")
    draw.polygon(pontos_maia, outline="blue", width=6)
            
    return detections
# A lógica de inferencia é mais ou menos até aqui, dps disto é a definição do poligono e da linha que vai ser desenhada
# Aqui o que se vai fazer basicamente é ver as labels todas da frente do camião, vamos ver as coorde

def check_vertical_alignment_front(front, visible_corner, invisible_corner,detections):
            
            
            #front = [l for l in detections if l["class"] == "front"]
            #visible_corner = [l for l in detections if l["class"] == "visible_corner"]
            #invisible_corner = [l for l in detections if l["class"] == "invisible_corner"]
            
            x_coords = [l["x"] for l in front + visible_corner + invisible_corner] #coordenadas x de cada label
            
            print(f"Coordenadas x das labels: {x_coords}")

            #Temos de calcular a média e verificar se tem u grande desvio
            mean_x = np.mean(x_coords)
            deviations = [np.abs(x-mean_x) for x in x_coords] # Para cada x., vamos ver o desvio absoluto da media menos o seu valor
            max_deviation = np.max(deviations)  # Maior desvio absoluto
            
            tolerancia = 0.05 * detections[0]['width']  # Tolerância em pixels para considerar que as labels estão alinhadas
            
            alinhado = max_deviation <= tolerancia
            print(f"Alinhamento: {'Sim' if alinhado else 'Não'} (Máximo desvio: {max_deviation:.2f} pixels, Tolerância: {tolerancia:.2f} pixels)")
            
            return alinhado,x_coords, max_deviation


def check_behind_side(front, visible_corner, invisible_corner, side, detections):

        #detections = inference()                                                                
        #front = [l for l in detections if l["class"] == "front"]
        #visible_corner = [l for l in detections if l["class"] == "visible_corner"]
        #invisible_corner = [l for l in detections if l["class"] == "invisible_corner"]
        #side = [l for l in detections if l["class"] == "side"]

        if not side:
            return False, 0.0, {'error': 'No side labels found'}
        
        side_right_edges = [ l["x"] + l["width"] / 2  for l in side]                            # | Obter a Direita do SIDE
        max_side_right = max(side_right_edges)
        
        front_components = front + visible_corner + invisible_corner
        if not front_components:
            return False, 0.0, {'error': 'No front components found'}
            
        front_left_edges = [l['x'] - l['width'] / 2 for l in front_components]                  # | Obter a Esquerda do FRONT
        min_front_left = min(front_left_edges)
        
        distance = min_front_left - max_side_right                                              # DISTANCE -> Vai ser sempre um valor negativo, o min_front_left é sempre menor que o max_side_right
        is_behind = distance >= 0.02                                                            # | Não entende este valor 0.02
        print(f"is_behind: {is_behind}, distance: {distance:.2f} pixels")                                                                 
        
        return is_behind, distance


def check_truck_position(detections,is_vertically_aligned, is_behind_side):
        """Comprehensive check of truck position with detailed analysis"""
        required_classes = ['front', 'side']
        present_classes = set(l["class"] for l in detections)
        missing_classes = set(required_classes) - present_classes
        
        if missing_classes:
            print(f"Missing required classes: {missing_classes}")
        
        #is_vertically_aligned, x_coords, max_dev = check_vertical_alignment_front(detections)
        #is_behind_side, distance  = check_behind_side(detections)
        is_valid = is_vertically_aligned and is_behind_side
        
        return is_valid
    

def on_hover(event):
    if event.inaxes:
        x, y = int(event.xdata), int(event.ydata)
        plt.title(f'Coordinates: ({x}, {y})')
        plt.draw()

# Convert PIL image to numpy array for matplotlib
img_array = np.array(image)

# Create matplotlib figure and display image
fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(img_array)
ax.set_title('Hover over image to see coordinates')

# Connect the hover event
fig.canvas.mpl_connect('motion_notify_event', on_hover)

# Show the plot
plt.show()

def main():
    print("Starting truck detection and alignment checking...")
    
    # Test the inference function
    print("\n1. Running inference...")
    detections = inference()
    front, visible, invisible, side = auxiliary_function(detections)
    
    # Test the vertical alignment check
    print("\n2. Checking vertical alignment...")
    is_vertically_aligned, x_coords, max_dev = check_vertical_alignment_front(front, visible, invisible, detections)

    
    print("\n3. Checking behind alignment...")
    is_behind_side, distance = check_behind_side(front, visible, invisible, side, detections)
    
    print(f"\nTotal detections found: {len(detections) if detections else 0}")
    print("Displaying the image with detections...")
    # Display the image with detections
    plt.imshow(image)
    plt.axis('off')  # Hide axes
    plt.show()
    
    #print("\n Alignment Detections:")
    #print(alignment_detections)
    #print("\n Alignment Detections Side:")
    #print(alignment_detections_side)

    print("\n4. Checking truck position...")
    is_valid = check_truck_position(detections,is_vertically_aligned,is_behind_side)
    print(f"Truck position is valid: {is_valid}")
    
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
    
    print("\nProcessing complete. Check the displayed image for visual results.")

if __name__ == "__main__":
    main()

#output_path = "output_image.jpg"
#image.save(output_path)
#vprint(f"Image saved to {output_path}")



## TODO:
# - [ ] Aquilo das listas no inicio das funcoes podia ser uma funcao so para isso
# - 