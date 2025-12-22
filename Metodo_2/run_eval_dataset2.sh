#!/bin/bash

# ===========================================
# Avalia o modelo treinado em CASIA no dataset NUAA
# Cross-Dataset Evaluation: CASIA → NUAA
# ===========================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "============================================="
echo "  Cross-Dataset: CASIA → NUAA"
echo "============================================="

# Verificar se o modelo existe
if [ ! -f "$SCRIPT_DIR/models/metodo2_vgg16_best.keras" ] && \
   [ ! -f "$SCRIPT_DIR/models/metodo2_vgg16_final.keras" ]; then
    echo "❌ ERRO: Modelo não encontrado!"
    echo "   Rode primeiro: ./run.sh"
    exit 1
fi

# Verificar se o dataset (NUAA) existe
if [ ! -d "$PROJECT_ROOT/dataset/test" ]; then
    echo "❌ ERRO: dataset (NUAA) não encontrado!"
    echo "   Rode primeiro: python database.py"
    exit 1
fi

cd "$PROJECT_ROOT"
python -m Metodo_2.eval_dataset2

echo ""
echo "✅ Avaliação cross-dataset concluída!"
echo ""
echo "Arquivos gerados em Metodo_2/results/:"
echo "  - results_cross_nuaa.txt"
echo "  - confusion_matrix_nuaa.png"
echo "  - roc_curve_nuaa.png"