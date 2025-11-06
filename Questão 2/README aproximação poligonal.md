# Este script carrega uma imagem, binariza, encontra o **maior contorno** e gera uma **aproximação poligonal** (Ramer–Douglas–Peucker) para representar a forma de maneira mais clara. Abaixo está o **mesmo código** com **comentários curtos** explicando o que cada parte faz.

> Dependências (linha de comando): `pip install opencv-python matplotlib numpy`  
> A linha `!pip install ...` funciona em ambientes tipo notebook; em script/terminal use o comando acima.

```python
# aproxima_poligonal.py
# Requer: opencv-python, matplotlib, numpy

# (Instalação em notebook; em terminal use: pip install opencv-python matplotlib numpy)
!pip install opencv-python matplotlib numpy

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Caminho da imagem de entrada (troque pelo seu arquivo, ex.: "Imagem Q2.png")
IMAGE_PATH = "/content/Imagem Q2.png"

# Fator de simplificação do polígono (percentual do perímetro)
# 0.01–0.03 costumam ser bons pontos de partida (menor = mais detalhe, maior = mais simples)
EPSILON_RATIO = 0.02

def main():
    # 1) Carrega a imagem em BGR (colorida)
    img = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Não consegui abrir: {Path(IMAGE_PATH).resolve()}")

    # 2) Pré-processa para facilitar segmentação: cinza -> blur -> limiarização (Otsu)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)       # converte para tons de cinza
    gray = cv2.GaussianBlur(gray, (5, 5), 0)           # suaviza ruído fino
    _, bin_ = cv2.threshold(                           # limiar automático (Otsu)
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 3) Morfologia (fechamento) para fechar pequenos buracos nas regiões brancas
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bin_clean = cv2.morphologyEx(bin_, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 4) Encontra apenas os contornos externos e escolhe o maior (objeto principal)
    cnts, _ = cv2.findContours(bin_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError("Nenhum contorno encontrado. Ajuste limiar/morfologia.")
    largest = max(cnts, key=cv2.contourArea)           # contorno com maior área

    # 5) Aproxima o contorno por um polígono (Ramer–Douglas–Peucker)
    peri = cv2.arcLength(largest, True)                # perímetro do contorno
    epsilon = EPSILON_RATIO * peri                     # tolerância em pixels
    approx = cv2.approxPolyDP(largest, epsilon, True)  # polígono simplificado

    # 6) Desenha resultados em cópias da imagem original
    overlay_contour = img.copy()
    overlay_poly = img.copy()
    cv2.drawContours(overlay_contour, [largest], -1, (0, 255, 255), 2)  # contorno original (amarelo)
    cv2.drawContours(overlay_poly, [approx], -1, (0, 0, 255), 3)        # polígono aproximado (vermelho)

    # 7) Exibe: original, binária, contorno e polígono
    img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)           # BGR->RGB para matplotlib
    cont_rgb = cv2.cvtColor(overlay_contour, cv2.COLOR_BGR2RGB)
    poly_rgb = cv2.cvtColor(overlay_poly, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1); plt.imshow(img_rgb);  plt.title("Original");                    plt.axis("off")
    plt.subplot(2, 2, 2); plt.imshow(bin_clean, cmap="gray"); plt.title("Binária");      plt.axis("off")
    plt.subplot(2, 2, 3); plt.imshow(cont_rgb);  plt.title("Contorno maior");             plt.axis("off")
    plt.subplot(2, 2, 4); plt.imshow(poly_rgb);  plt.title(f"Aproximação poligonal (ε={EPSILON_RATIO:.2%}·perímetro)"); plt.axis("off")
    plt.tight_layout()
    plt.show()

    # 8) Salva imagens com o contorno e o polígono (no mesmo diretório da entrada)
    out1 = Path(IMAGE_PATH).with_name("contorno_maior.png")
    out2 = Path(IMAGE_PATH).with_name("aprox_poligonal.png")
    cv2.imwrite(str(out1), overlay_contour)
    cv2.imwrite(str(out2), overlay_poly)
    print(f"[OK] Salvo: {out1}")
    print(f"[OK] Salvo: {out2}")

if __name__ == "__main__":
    main()