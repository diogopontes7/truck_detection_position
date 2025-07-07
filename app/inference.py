
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

load_dotenv() 
ROBOFLOW_API_KEY = os.getenv("api_key")

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)

img_path = os.path.join("examples", "0018146.jpg")
with open(img_path, "rb") as f:
     image_bytes = f.read()

result = client.run_workflow(
    workspace_name="summer-n8dui",
    workflow_id="detect-count-and-visualize-3",
    images={
        "image": img_path
    },
    use_cache=True 
)
#image_path = "C:/Users/Diogo/OneDrive - Universidade do Minho/Uni/Summer_Intern_Project/app/labeled_images_diogo/train/images/0001062_jpg.rf.88c8a78010ef47014cf5078d6b55ac16.jpg"
image = Image.open(img_path)
draw = ImageDraw.Draw(image)

#print(result)  # Print the result dictionary
print(type(result))

# https://docs.roboflow.com/deploy/serverless/object-detection

center_point_radius = 8 # Raio do círculo para desenhar o centro
center_point_color = "yellow" # Cor do ponto central

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

        draw.ellipse(
            (x - center_point_radius,
             y - center_point_radius,
             x + center_point_radius,
             y + center_point_radius),
            fill=center_point_color, # Preenche o círculo com a cor definida
            outline=center_point_color # A linha do círculo também é da mesma cor
        )

except (IndexError, KeyError) as e:
    print(f"Erro ao acessar as previsões {e}")
# Display the image
image.show()

