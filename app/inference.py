
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

# Meter clientInference no início do código, criar uma instância do cliente


# Passa para bytes, deve ser melhor que passar o caminho do arquivo
# with open("C:/Users/Diogo/OneDrive - Universidade do Minho/Uni/Summer_Intern_Project/app/labeled_images_diogo/train/images/0001062_jpg.rf.88c8a78010ef47014cf5078d6b55ac16.jpg", "rb") as f:
#     image_bytes = f.read()

result = client.run_workflow(
    workspace_name="summer-n8dui",
    workflow_id="detect-count-and-visualize-3",
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
            line_y = y2 + 10

        draw.line((x1, line_y, x2, line_y), fill="red", width=2)
        # Draw label with class name and confidence
        label = f"{class_name}: {confidence:.2f}"
        draw.text((x1, y1 - 10), label, fill="red", font=font)

except (IndexError, KeyError) as e:
    print(f"Erro ao acessar as previsões {e}")
# Display the image
image.show()

