#!/bin/bash

# ===========================================
# Script para gerar apenas as métricas
# (usa modelo já treinado)
# ===========================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "============================================="
echo "  Gerando métricas do Método 2..."
echo "============================================="

# Verificar se o modelo existe
if [ ! -f "$SCRIPT_DIR/models/metodo2_vgg16_best.keras" ] && \
   [ ! -f "$SCRIPT_DIR/models/metodo2_vgg16_final.keras" ]; then
    echo "❌ ERRO: Modelo não encontrado!"
    echo "   Rode primeiro: ./run.sh"
    exit 1
fi

cd "$PROJECT_ROOT"
python -m Metodo_2.metrics

echo ""
echo "✅ Métricas geradas com sucesso!"