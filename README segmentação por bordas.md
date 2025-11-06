# Em terminal/script normal, use: pip install opencv-python matplotlib
!pip install opencv-python

# Importa o OpenCV (processamento de imagem) e o Matplotlib (exibição)
import cv2
import matplotlib.pyplot as plt

def canny_edge_detection(imagem_path, limite_baixo, limite_alto):
    # Carrega a imagem do disco (formato BGR).
    # -> Troque 'imagem_path' pelo caminho do seu arquivo (ex.: 'foto.jpg').
    imagem = cv2.imread(imagem_path)
    
    # Verifica se a imagem foi carregada corretamente.
    if imagem is None:
        print(f"Erro ao carregar a imagem em: {imagem_path}")
        return

    # Converte a imagem para escala de cinza (passo típico antes do Canny).
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # (Opcional) Reduz ruído para melhorar a detecção de bordas.
    # Descomente a linha abaixo para testar um leve desfoque:
    # imagem_cinza = cv2.GaussianBlur(imagem_cinza, (5, 5), 0)
    
    # Aplica o detector de bordas de Canny.
    # 'limite_baixo' e 'limite_alto' são os limiares de histerese:
    #   - Gradientes abaixo do limite_baixo são descartados.
    #   - Acima do limite_alto são aceitos como borda.
    #   - Entre eles, só entram se conectados a bordas fortes.
    bordas_detectadas = cv2.Canny(imagem_cinza, limite_baixo, limite_alto)
    
    # Exibe lado a lado: original (convertida para RGB) e bordas detectadas (grayscale).
    plt.figure(figsize=(10, 5))

    # Painel 1: imagem original (BGR -> RGB para exibir corretamente no matplotlib).
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
    plt.title('Imagem Original')
    plt.axis('off')
    
    # Painel 2: imagem de bordas (mapa binário).
    plt.subplot(1, 2, 2)
    plt.imshow(bordas_detectadas, cmap='gray')
    plt.title(f'Bordas de Canny (t1={limite_baixo}, t2={limite_alto})')
    plt.axis('off')
    
    # Renderiza a figura na tela.
    plt.show()

# --- Exemplo de uso ---
# Troque o caminho pela sua imagem e ajuste os limiares conforme o contraste/ruído:
canny_edge_detection('Imagem Industrial.jpg', 100, 200)
