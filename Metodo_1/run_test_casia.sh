#!/bin/bash

# ================================================================
# Script para testar Método 1 (LBP+SVM) no dataset CASIA-FASD
# ================================================================

set -e  # Para em caso de erro

# Diretório do script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=========================================================="
echo "  MÉTODO 1 - Teste no CASIA-FASD (dataset2)"
echo "=========================================================="
echo ""
echo "Diretório do projeto: $PROJECT_ROOT"
echo "Diretório do Método 1: $SCRIPT_DIR"
echo ""

# Verificar se o modelo existe
MODEL_PATH="$SCRIPT_DIR/models/metodo1_lbp_svm.pkl"
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERRO: Modelo não encontrado em $MODEL_PATH"
    echo "Execute svm.py primeiro para treinar o modelo."
    exit 1
fi

# Verificar se o dataset2 existe
DATASET2_PATH="$PROJECT_ROOT/dataset2"
if [ ! -d "$DATASET2_PATH" ]; then
    echo "ERRO: Dataset CASIA-FASD não encontrado em $DATASET2_PATH"
    echo "Execute database2.py primeiro para baixar e organizar o dataset."
    exit 1
fi

echo "Modelo encontrado: $MODEL_PATH"
echo "Dataset encontrado: $DATASET2_PATH"
echo ""

# Executar teste
cd "$SCRIPT_DIR"
python3 test_casia.py

echo ""
echo "=========================================================="
echo "  ✅ Teste concluído!"
echo "=========================================================="
echo ""
echo "Arquivos gerados:"
echo "  - results/results_metodo1_CASIA.txt"
echo "  - results/confusion_matrix_metodo1_CASIA.png"
echo "  - results/roc_curve_metodo1_CASIA.png"
echo ""
