#!/bin/bash

# ===========================================
# Avalia o modelo treinado no dataset2 (CASIA-FASD)
# ===========================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "============================================="
echo "  Avaliando modelo no dataset2 (CASIA-FASD)"
echo "============================================="

# Verificar se o modelo existe
if [ ! -f "$SCRIPT_DIR/models/metodo2_vgg16_best.keras" ] && \
   [ ! -f "$SCRIPT_DIR/models/metodo2_vgg16_final.keras" ]; then
    echo "❌ ERRO: Modelo não encontrado!"
    echo "   Rode primeiro: ./run.sh"
    exit 1
fi

# Verificar se o dataset2 existe
if [ ! -d "$PROJECT_ROOT/dataset2/test" ]; then
    echo "❌ ERRO: dataset2 não encontrado!"
    echo "   Rode primeiro: python database2.py"
    exit 1
fi

cd "$PROJECT_ROOT"
python -m Metodo_2.eval_dataset2

echo ""
echo "✅ Avaliação concluída!"
echo ""
echo "Arquivos gerados em Metodo_2/results/:"
echo "  - results_cross_dataset2.txt"
echo "  - confusion_matrix_dataset2.png"
echo "  - roc_curve_dataset2.png"