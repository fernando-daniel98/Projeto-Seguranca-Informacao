#!/bin/bash

# ===========================================
# Script para treinar e avaliar o Método 2
# VGG16 Transfer Learning para Anti-Spoofing
# ===========================================

set -e  # Para em caso de erro

# Diretório do script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "============================================="
echo "  MÉTODO 2 - VGG16 Transfer Learning"
echo "============================================="
echo ""
echo "Diretório do projeto: $PROJECT_ROOT"
echo "Diretório do Método 2: $SCRIPT_DIR"
echo ""

# Criar diretórios necessários
mkdir -p "$SCRIPT_DIR/models"
mkdir -p "$SCRIPT_DIR/results"

# Verificar se o dataset2 (CASIA-FASD) existe
if [ ! -d "$PROJECT_ROOT/dataset2/train" ]; then
    echo "❌ ERRO: Dataset2 (CASIA-FASD) não encontrado em $PROJECT_ROOT/dataset2/"
    echo "   Rode primeiro: python database2.py"
    exit 1
fi

echo "✅ Dataset2 (CASIA-FASD) encontrado"
echo ""

# Passo 1: Treinar o modelo
echo "============================================="
echo "  PASSO 1: Treinando o modelo VGG16..."
echo "============================================="
cd "$PROJECT_ROOT"
python -m Metodo_2.main

echo ""
echo "✅ Treinamento concluído!"
echo ""

# Passo 2: Gerar métricas
echo "============================================="
echo "  PASSO 2: Gerando métricas e visualizações..."
echo "============================================="
python -m Metodo_2.metrics

echo ""
echo "============================================="
echo "  ✅ EXECUÇÃO COMPLETA!"
echo "============================================="
echo ""
echo "Arquivos gerados:"
echo "  - Modelo:    $SCRIPT_DIR/models/metodo2_vgg16_best.keras"
echo "  - Resultados: $SCRIPT_DIR/results/results_metodo2.txt"
echo "  - Gráficos:"
echo "      - $SCRIPT_DIR/results/confusion_matrix_metodo2.png"
echo "      - $SCRIPT_DIR/results/roc_curve_metodo2.png"
echo "      - $SCRIPT_DIR/results/score_distribution_metodo2.png"
echo ""