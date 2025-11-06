# Em terminal/script normal, use: pip install opencv-python
!pip install opencv-python

# Importa o OpenCV (processamento de imagem) e o Matplotlib (exibição)
import cv2
import matplotlib.pyplot as plt

# Carrega a imagem já em ESCALA DE CINZA.
# -> Troque o caminho '/content/Imagem Industrial.jpg' para o seu arquivo.
#    Se der caminho errado ou arquivo não existir, 'imagem' será None.
imagem = cv2.imread('/content/Imagem Industrial.jpg', cv2.IMREAD_GRAYSCALE)

# Verifica se a imagem foi carregada corretamente
if imagem is None:
    print("Erro ao carregar a imagem!")
else:
    # Define os parâmetros da limiarização:
    # - valor_limiar: o corte (0–255). Pixels >= limiar viram brancos.
    # - valor_maximo: valor atribuído aos pixels acima do limiar (normalmente 255).
    valor_limiar = 127
    valor_maximo = 255

    # Aplica a Limiarização Binária:
    # cv2.threshold retorna (limiar_usado, imagem_resultado)
    # usamos '_' para ignorar o primeiro retorno (não precisamos dele aqui).
    _, imagem_limiarizada = cv2.threshold(
        imagem, valor_limiar, valor_maximo, cv2.THRESH_BINARY
    )

    # Cria uma figura com 2 painéis (Original x Limiarizada)
    plt.figure(figsize=(10, 5))

    # Painel 1: imagem original em tons de cinza
    plt.subplot(1, 2, 1)
    plt.imshow(imagem, cmap='gray')          # exibe em escala de cinza
    plt.title('Imagem Original (Escala de Cinza)')
    plt.axis('off')                          # esconde eixos para ficar limpo

    # Painel 2: imagem após a limiarização binária
    plt.subplot(1, 2, 2)
    plt.imshow(imagem_limiarizada, cmap='gray')
    plt.title(f'Imagem Limiarizada (Limiar={valor_limiar})')
    plt.axis('off')

    # Renderiza a figura na tela
    plt.show()

