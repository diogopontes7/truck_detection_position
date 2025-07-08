
#import cv2
#import inference
import ast
import os
from inference_sdk import InferenceHTTPClient
import base64
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()  # Carrega as variáveis de ambiente do arquivo .env

# Meter clientInference no início do código, criar uma instância do cliente

api_key = os.getenv("ROBOFLOW_API_KEY")

max_right_x = 0  # Inicializa o valor máximo para o canto visível
min_left_x = float('inf')
min_right_x = float('inf')  # Inicializa o valor mínimo para o canto visível (closest right point)

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key= api_key
)
# Passa para bytes, deve ser melhor que passar o caminho do arquivo
# with open("C:/Users/Diogo/OneDrive - Universidade do Minho/Uni/Summer_Intern_Project/app/labeled_images_diogo/train/images/0001062_jpg.rf.88c8a78010ef47014cf5078d6b55ac16.jpg", "rb") as f:
#     image_bytes = f.read()

result = client.run_workflow(
    workspace_name="summer-n8dui",
    workflow_id="detect-count-and-visualize-7",
    images={
        "image": "C:/Users/Diogo/OneDrive - Universidade do Minho/Uni/Summer_Intern_Project/app/labeled_images_diogo/train/images/0001062_jpg.rf.88c8a78010ef47014cf5078d6b55ac16.jpg" 
    },
    use_cache=True # cache workflow definition for 15 minutes
)
image_path = "C:/Users/Diogo/OneDrive - Universidade do Minho/Uni/Summer_Intern_Project/app/labeled_images_diogo/train/images/0001062_jpg.rf.88c8a78010ef47014cf5078d6b55ac16.jpg"
image = Image.open(image_path)
draw = ImageDraw.Draw(image)

#print(result)  # Print the result dictionary
print(type(result))

# https://docs.roboflow.com/deploy/serverless/object-detection

try:
    detections = result[0]['predictions']['predictions']  # Vai buscar as previsões do resultado
    print(detections)  #verificar se vai buscar o certo
    font = ImageFont.truetype("arial.ttf", 40)
    
    for bounding_box in detections:
        
        x = bounding_box['x']
        y = bounding_box['y']
        width = bounding_box['width']
        height = bounding_box['height']
        class_name = bounding_box['class']
        confidence = bounding_box['confidence']


        x1 = x - width / 2 # esquerda
        x2 = x + width / 2 # direita
        y1 = y - height / 2 # topo
        y2 = y + height / 2 # base

        # Draw rectangle
        draw.rectangle((x1, y1, x2, y2), outline="red", width=2)

        if class_name == "front":
            front_detection = {
                'x1': x1,
                'x2': x2,
                'y2': y2,
            }# define a linha de baixo da label do front, o x1 é o ponto esquerdo e o x2 é o ponto direito, y2 é a base da label do front

        # Find the furthermost right point of visible_corner
        if class_name == "visible_corner":
             max_right_x = max(max_right_x, x2) - width
            
        # if class_name == "visible_corner":
        #     min_right_x = min(min_right_x, x2)
            
        # Find the furthermost left point of invisible_corner
        if class_name == "invisible_corner":
            min_left_x = min(min_left_x, x1)

        # Draw label with class name and confidence
        label = f"{class_name}: {confidence:.2f}"
        draw.text((x1, y1 - 10), label, fill="red", font=font)
        
        if front_detection and (max_right_x >0 or min_left_x < float('inf')):
            line_y = front_detection['y2'] + 10
        
        # Determina a linha, de acordo se encontra a label ou não
            line_start_x = min_left_x if min_left_x < float('inf') else front_detection['x1'] # Se nao for inf, signoifoca que foi encontrado a label e existe um min_left do invisible_corner
            line_end_x = max_right_x if max_right_x > 0 else front_detection['x2'] # Mesma logica para o visible_corner, se for maior que 0, significa que foi encontrado a label e existe um max_right do visible_corner
            #line_end_x = min_right_x if min_right_x < float('inf') else front_detection['x2'] 

        # se nao encontramos, usamos o ponto direito e esquerdo da label do front
        
            draw.line((line_start_x, line_y, line_end_x, line_y), fill="green", width=2)

except (IndexError, KeyError) as e:
    print(f"Erro ao acessar as previsões {e}")
# Display the image
image.show()

