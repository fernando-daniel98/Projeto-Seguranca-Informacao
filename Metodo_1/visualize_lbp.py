"""
Visualização das features LBP extraídas pelo extract_lbp.py
Mostra:
- Imagens originais vs. imagens LBP
- Histogramas LBP
- Comparação entre imagens reais e falsas
"""

from pathlib import Path
import os
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern


def extract_lbp_image(image_path, radius=2, n_points=None, method="uniform", resize_to=(224, 224)):
    """
    Extrai a imagem LBP (não apenas o histograma).
    """
    if n_points is None:
        n_points = 8 * radius

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Não foi possível ler a imagem: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if resize_to is not None:
        gray = cv2.resize(gray, resize_to)

    lbp = local_binary_pattern(gray, n_points, radius, method)
    
    # Também retorna o histograma
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=n_bins,
        range=(0, n_bins),
        density=True
    )
    
    return img, gray, lbp, hist


def visualize_single_image(image_path, radius=2, n_points=None, method="uniform"):
    """
    Visualiza uma única imagem: original, grayscale, LBP e histograma.
    """
    img, gray, lbp, hist = extract_lbp_image(image_path, radius, n_points, method)
    
    # Converter BGR para RGB para matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Imagem original
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Imagem Original')
    axes[0, 0].axis('off')
    
    # Imagem em tons de cinza
    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title('Imagem em Grayscale')
    axes[0, 1].axis('off')
    
    # Imagem LBP
    axes[1, 0].imshow(lbp, cmap='gray')
    axes[1, 0].set_title(f'LBP Pattern (R={radius}, P={n_points or 8*radius})')
    axes[1, 0].axis('off')
    
    # Histograma LBP
    axes[1, 1].bar(range(len(hist)), hist, color='steelblue')
    axes[1, 1].set_title('Histograma LBP Normalizado')
    axes[1, 1].set_xlabel('Bins LBP')
    axes[1, 1].set_ylabel('Frequência Normalizada')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle(f'Análise LBP: {Path(image_path).name}', y=1.02, fontsize=14, fontweight='bold')
    return fig


def compare_real_vs_fake(dataset_dir, n_samples=3, radius=2, n_points=None, method="uniform"):
    """
    Compara imagens reais vs falsas lado a lado.
    """
    dataset_path = Path(dataset_dir)
    train_dir = dataset_path / "train"
    
    # Coletar amostras
    real_dir = train_dir / "real"
    fake_dir = train_dir / "fake"
    
    img_exts = (".jpg", ".jpeg", ".png", ".bmp")
    
    # Buscar imagens reais
    real_images = []
    for root, _, files in os.walk(real_dir):
        for fname in files:
            if fname.lower().endswith(img_exts):
                real_images.append(Path(root) / fname)
    
    # Buscar imagens falsas
    fake_images = []
    for root, _, files in os.walk(fake_dir):
        for fname in files:
            if fname.lower().endswith(img_exts):
                fake_images.append(Path(root) / fname)
    
    # Selecionar amostras aleatórias
    real_samples = random.sample(real_images, min(n_samples, len(real_images)))
    fake_samples = random.sample(fake_images, min(n_samples, len(fake_images)))
    
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4*n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        # Processar imagem real
        if i < len(real_samples):
            img_r, gray_r, lbp_r, hist_r = extract_lbp_image(real_samples[i], radius, n_points, method)
            img_r_rgb = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
            
            axes[i, 0].imshow(img_r_rgb)
            axes[i, 0].set_title(f'REAL #{i+1}')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(lbp_r, cmap='gray')
            axes[i, 1].set_title('LBP Pattern (Real)')
            axes[i, 1].axis('off')
        
        # Processar imagem falsa
        if i < len(fake_samples):
            img_f, gray_f, lbp_f, hist_f = extract_lbp_image(fake_samples[i], radius, n_points, method)
            img_f_rgb = cv2.cvtColor(img_f, cv2.COLOR_BGR2RGB)
            
            axes[i, 2].imshow(img_f_rgb)
            axes[i, 2].set_title(f'FAKE #{i+1}')
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(lbp_f, cmap='gray')
            axes[i, 3].set_title('LBP Pattern (Fake)')
            axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Comparação: Imagens REAIS vs FALSAS (LBP)', y=1.00, fontsize=16, fontweight='bold')
    return fig


def compare_histograms(dataset_dir, n_samples=10, radius=2, n_points=None, method="uniform"):
    """
    Compara os histogramas médios entre imagens reais e falsas.
    """
    dataset_path = Path(dataset_dir)
    train_dir = dataset_path / "train"
    
    real_dir = train_dir / "real"
    fake_dir = train_dir / "fake"
    
    img_exts = (".jpg", ".jpeg", ".png", ".bmp")
    
    # Coletar histogramas
    real_hists = []
    fake_hists = []
    
    # Imagens reais
    real_images = []
    for root, _, files in os.walk(real_dir):
        for fname in files:
            if fname.lower().endswith(img_exts):
                real_images.append(Path(root) / fname)
    
    for img_path in random.sample(real_images, min(n_samples, len(real_images))):
        try:
            _, _, _, hist = extract_lbp_image(img_path, radius, n_points, method)
            real_hists.append(hist)
        except Exception as e:
            print(f"Erro ao processar {img_path}: {e}")
    
    # Imagens falsas
    fake_images = []
    for root, _, files in os.walk(fake_dir):
        for fname in files:
            if fname.lower().endswith(img_exts):
                fake_images.append(Path(root) / fname)
    
    for img_path in random.sample(fake_images, min(n_samples, len(fake_images))):
        try:
            _, _, _, hist = extract_lbp_image(img_path, radius, n_points, method)
            fake_hists.append(hist)
        except Exception as e:
            print(f"Erro ao processar {img_path}: {e}")
    
    # Calcular médias
    real_hist_mean = np.mean(real_hists, axis=0)
    fake_hist_mean = np.mean(fake_hists, axis=0)
    
    # Plotar
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    x = np.arange(len(real_hist_mean))
    width = 0.35
    
    # Gráfico de comparação lado a lado
    axes[0].bar(x - width/2, real_hist_mean, width, label='Real', color='green', alpha=0.7)
    axes[0].bar(x + width/2, fake_hist_mean, width, label='Fake', color='red', alpha=0.7)
    axes[0].set_xlabel('Bins LBP')
    axes[0].set_ylabel('Frequência Média Normalizada')
    axes[0].set_title(f'Comparação de Histogramas LBP Médios\n({n_samples} amostras por classe)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Gráfico de diferença
    diff = real_hist_mean - fake_hist_mean
    colors = ['green' if d > 0 else 'red' for d in diff]
    axes[1].bar(x, diff, color=colors, alpha=0.7)
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].set_xlabel('Bins LBP')
    axes[1].set_ylabel('Diferença (Real - Fake)')
    axes[1].set_title('Diferença entre Histogramas\n(Verde: Real > Fake, Vermelho: Fake > Real)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    """
    Função principal que gera todas as visualizações.
    """
    base_dir = Path(__file__).resolve().parent
    dataset_dir = base_dir / "dataset"
    output_dir = base_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not dataset_dir.exists():
        print(f"ERRO: Diretório dataset não encontrado em {dataset_dir}")
        return
    
    print("=== Visualização de Features LBP ===\n")
    print(f"Salvando visualizações em: {output_dir}\n")
    
    # Parâmetros LBP (mesmos do extract_lbp.py)
    radius = 2
    n_points = 8 * radius
    method = "uniform"
    
    # 1. Visualizar uma imagem individual de exemplo
    print("1. Buscando uma imagem de exemplo...")
    train_dir = dataset_dir / "train" / "real"
    img_exts = (".jpg", ".jpeg", ".png", ".bmp")
    
    example_image = None
    for root, _, files in os.walk(train_dir):
        for fname in files:
            if fname.lower().endswith(img_exts):
                example_image = Path(root) / fname
                break
        if example_image:
            break
    
    if example_image:
        print(f"   Visualizando: {example_image.name}")
        fig1 = visualize_single_image(example_image, radius, n_points, method)
        output_path1 = output_dir / "01_single_image_lbp.png"
        fig1.savefig(output_path1, dpi=150, bbox_inches='tight')
        print(f"   Salvo: {output_path1.name}")
        plt.close(fig1)
    else:
        print("   Nenhuma imagem de exemplo encontrada.")
    
    # 2. Comparar imagens reais vs falsas
    print("\n2. Comparando imagens REAIS vs FALSAS...")
    fig2 = compare_real_vs_fake(dataset_dir, n_samples=3, radius=radius, n_points=n_points, method=method)
    output_path2 = output_dir / "02_real_vs_fake_comparison.png"
    fig2.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"   Salvo: {output_path2.name}")
    plt.close(fig2)
    
    # 3. Comparar histogramas médios
    print("\n3. Comparando histogramas médios...")
    fig3 = compare_histograms(dataset_dir, n_samples=20, radius=radius, n_points=n_points, method=method)
    output_path3 = output_dir / "03_histogram_comparison.png"
    fig3.savefig(output_path3, dpi=150, bbox_inches='tight')
    print(f"   Salvo: {output_path3.name}")
    plt.close(fig3)
    
    print("\n=== Visualização concluída ===")
    print(f"Todas as imagens foram salvas em: {output_dir}")


if __name__ == "__main__":
    main()
