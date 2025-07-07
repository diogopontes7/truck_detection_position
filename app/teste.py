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

# Carrega as variáveis de ambiente do ficheiro .env
load_dotenv() 

# Obtém a API Key da variável de ambiente 'api_key'
ROBOFLOW_API_KEY = os.getenv("api_key")

# Verifica se a API Key foi carregada
if not ROBOFLOW_API_KEY:
    print("Erro: A variável de ambiente 'api_key' não foi definida no ficheiro .env ou no ambiente.")
    print("Por favor, cria um ficheiro .env na raiz do teu projeto com 'api_key=\"tua_chave_aqui\"'.")
    exit(1) # Sai se a API Key não estiver definida

# Inicializa o cliente InferenceHTTPClient
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)

# Caminho da imagem (assegura-te que a pasta 'examples' e a imagem existem)
img_path = os.path.join("examples", "0018146.jpg")

# 1. Carrega a imagem original para um objeto PIL Image
try:
    # Abre a imagem e converte para RGB para garantir compatibilidade com desenho
    original_image = Image.open(img_path).convert("RGB")
except FileNotFoundError:
    print(f"Erro: Imagem não encontrada no caminho: {img_path}")
    exit(1) # Sai se a imagem não for encontrada

# --- ATENÇÃO: Consistência na passagem da imagem para a API ---
# A forma mais robusta é passar o objeto PIL.Image diretamente.
# A linha 'with open(img_path, "rb") as f: image_bytes = f.read()' não é necessária se passares o objeto PIL.Image.
# Também não precisas de 'images={"image": img_path}' se já tens o objeto PIL.Image.

# 2. Executa o workflow de inferência
try:
    result = client.run_workflow(
        workspace_name="summer-n8dui",
        workflow_id="detect-count-and-visualize-3",
        images={
            # Passa o objeto PIL.Image diretamente. A inference_sdk gere a conversão.
            "image": original_image 
        },
        use_cache=True 
    )
except Exception as e:
    print(f"Erro ao executar o workflow de inferência: {e}")
    print(f"Detalhes do erro da API: {getattr(e, 'api_message', 'N/A')}")
    exit(1) # Sai se a inferência falhar

# Prepara para desenhar na imagem original
draw = ImageDraw.Draw(original_image)

# Imprime o tipo do resultado (para depurar, pode ser removido depois)
print(f"Tipo do resultado da API: {type(result)}")

# Configurações de desenho
line_width_box = 3 # Largura das linhas das bounding boxes
line_width_plane = 5 # Largura das linhas dos planos
font_size = 30 # Tamanho da fonte para as labels

try:
    # Tenta carregar uma fonte TTF. Ajusta o caminho se 'arial.ttf' não for encontrado.
    font = ImageFont.truetype("arial.ttf", font_size)
except IOError:
    font = ImageFont.load_default()
    print("Aviso: Fonte 'arial.ttf' não encontrada. A usar fonte padrão.")

# Mapeamento de cores para cada classe
class_colors = {
    'front': "red",
    'visible_corner': "green",
    'side': "blue",
    'invisible_corner': "purple" # Cor para o lado "invisível"
}

# Dicionário para armazenar as coordenadas das bounding boxes para uso posterior nas ligações dos planos
bounding_boxes_by_class = {
    'front': None,
    'visible_corner': None,
    'side': None,
    'invisible_corner': None
}

# 3. Processa e desenha as bounding boxes e etiquetas
try:
    # Acede às previsões. A estrutura 'result[0]' implica que é uma lista de resultados.
    detections = result[0]['predictions']['predictions']
    print(f"Deteções obtidas: {detections}") # Para verificar as deteções

    for bounding_box in detections:
        x_center = bounding_box['x']
        y_center = bounding_box['y']
        width = bounding_box['width']
        height = bounding_box['height']
        class_name = bounding_box['class']
        confidence = bounding_box['confidence']

        # Calcula as coordenadas dos cantos do retângulo (x_min, y_min, x_max, y_max)
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2

        # Armazena as coordenadas para ligar os planos depois
        bounding_boxes_by_class[class_name] = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'x_center': x_center, 'y_center': y_center}

        # Define a cor do retângulo
        color = class_colors.get(class_name, "white")

        # Desenha o retângulo da caixa delimitadora
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width_box)

        # Desenha a etiqueta com nome da classe e confiança
        label = f"{class_name}: {confidence:.2f}"
        # Posição do texto um pouco acima do retângulo, ajustando para não sair da imagem
        text_y_position = max(0, y1 - (font_size + 5)) 
        draw.text((x1, text_y_position), label, fill=color, font=font)

except (IndexError, KeyError) as e:
    print(f"Erro ao acessar ou processar as previsões: {e}")
    print("Verifica a estrutura do 'result' retornado pela API ou se não há deteções.")
    # Imprime os primeiros 500 caracteres do resultado para ajudar na depuração
    print(f"Resultado completo da API (primeiros 500 chars): {str(result)[:500]}...")

### 4. Desenhar Linhas de Definição dos Planos do Camião
# Cores para as linhas de ligação dos planos
plane_connection_color = "cyan" # Uma cor contrastante
plane_detail_color = "magenta"  # Para detalhes internos dos planos

# Obtém as bounding boxes por classe para facilitar o acesso
front_box = bounding_boxes_by_class.get('front')
visible_corner_box = bounding_boxes_by_class.get('visible_corner')
side_box = bounding_boxes_by_class.get('side')
invisible_corner_box = bounding_boxes_by_class.get('invisible_corner')

# --- Desenhar a Frente do Camião ---
if front_box:
    # A frente é a bounding box em si, mas podemos reforçar os seus contornos
    # ou adicionar uma linha de base se for o "chão" ou a parte inferior do plano.
    x1, y1, x2, y2 = front_box['x1'], front_box['y1'], front_box['x2'], front_box['y2']
    # Reforça a base da frente como um "plano"
    draw.line([(x1, y2), (x2, y2)], fill=class_colors.get('front'), width=line_width_plane)
    # Adicionar uma linha vertical no centro da frente (opcional, para dar profundidade)
    # draw.line([(front_box['x_center'], y1), (front_box['x_center'], y2)], fill=plane_detail_color, width=line_width_box)


# --- Ligar Frente ao Canto Visível (e Canto Visível ao Lado) ---
if front_box and visible_corner_box:
    # Ligar canto superior direito da frente ao canto superior esquerdo do canto visível
    draw.line([(front_box['x2'], front_box['y1']), (visible_corner_box['x1'], visible_corner_box['y1'])], 
              fill=plane_connection_color, width=line_width_plane)
    # Ligar canto inferior direito da frente ao canto inferior esquerdo do canto visível
    draw.line([(front_box['x2'], front_box['y2']), (visible_corner_box['x1'], visible_corner_box['y2'])], 
              fill=plane_connection_color, width=line_width_plane)
    
    # Se o 'visible_corner' é um plano em si, podemos reforçar as suas diagonais ou contornos internos
    # draw.line([(visible_corner_box['x1'], visible_corner_box['y1']), (visible_corner_box['x2'], visible_corner_box['y2'])], 
    #           fill=plane_detail_color, width=line_width_box)
    # draw.line([(visible_corner_box['x1'], visible_corner_box['y2']), (visible_corner_box['x2'], visible_corner_box['y1'])], 
    #           fill=plane_detail_color, width=line_width_box)

    # Ligar canto superior direito do canto visível ao canto superior esquerdo do lado
    if side_box:
        draw.line([(visible_corner_box['x2'], visible_corner_box['y1']), (side_box['x1'], side_box['y1'])], 
                  fill=plane_connection_color, width=line_width_plane)
        # Ligar canto inferior direito do canto visível ao canto inferior esquerdo do lado
        draw.line([(visible_corner_box['x2'], visible_corner_box['y2']), (side_box['x1'], side_box['y2'])], 
                  fill=plane_connection_color, width=line_width_plane)

# --- Ligar Frente ao Canto Invisível ---
if front_box and invisible_corner_box:
    # Ligar canto superior esquerdo da frente ao canto superior direito do canto invisível
    draw.line([(front_box['x1'], front_box['y1']), (invisible_corner_box['x2'], invisible_corner_box['y1'])],
              fill=plane_connection_color, width=line_width_plane)
    # Ligar canto inferior esquerdo da frente ao canto inferior direito do canto invisível
    draw.line([(front_box['x1'], front_box['y2']), (invisible_corner_box['x2'], invisible_corner_box['y2'])],
              fill=plane_connection_color, width=line_width_plane)

# --- Desenhar o Lado do Camião ---
if side_box:
    # O lado é a bounding box em si. Poderíamos adicionar uma linha de "chão" ou "topo"
    # draw.line([(side_box['x1'], side_box['y2']), (side_box['x2'], side_box['y2'])], fill=class_colors.get('side'), width=line_width_plane)
    pass # As ligações já ajudam a definir o plano lateral



# Salva a imagem processada numa pasta 'output_planos'
output_dir = "output_planos"
os.makedirs(output_dir, exist_ok=True) 
output_image_name = os.path.basename(img_path).replace(".", "_planos_final.")
output_image_path = os.path.join(output_dir, output_image_name)

original_image.save(output_image_path)
print(f"\nImagem com os planos desenhados guardada em: {output_image_path}")

# Abre a imagem para visualização (no visualizador padrão do sistema)
original_image.show()