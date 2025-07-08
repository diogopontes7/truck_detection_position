import cv2
import os

# Lista para armazenar os pontos clicados
clicked_points = []

# Função de callback para eventos do rato
def get_points_on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN: # Se o botão esquerdo do rato for clicado
        clicked_points.append((x, y))
        print(f"Ponto selecionado: ({x}, {y})")
        # Opcional: desenhar um círculo no ponto clicado para feedback visual
        cv2.circle(image_display, (x, y), 5, (0, 255, 0), -1) # Círculo verde preenchido
        cv2.imshow("Clique nos pontos (Pressione 'q' para sair)", image_display)

def select_points_from_image(image_path):
    global image_display # Usar uma variável global para a imagem exibida

    # Carregar a imagem
    image_display = cv2.imread(image_path)
    if image_display is None:
        print(f"Erro: Não foi possível carregar a imagem em '{image_path}'")
        return []

    print("Clique nos pontos desejados na imagem. Pressione 'q' para terminar.")

    # Criar uma janela e definir a função de callback do rato
    cv2.namedWindow("Clique nos pontos (Pressione 'q' para sair)")
    cv2.setMouseCallback("Clique nos pontos (Pressione 'q' para sair)", get_points_on_click)

    # Mostrar a imagem
    cv2.imshow("Clique nos pontos (Pressione 'q' para sair)", image_display)

    # Esperar por uma tecla (pressione 'q' para sair)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): # Se a tecla 'q' for pressionada
            break

    cv2.destroyAllWindows()
    return clicked_points

# --- Exemplo de Uso ---
if __name__ == "__main__":
    img_path = os.path.join("examples", "0018146.jpg")
    selected_coords = select_points_from_image(img_path)

    print("\nPontos finais selecionados:")
    for point in selected_coords:
        print(point)