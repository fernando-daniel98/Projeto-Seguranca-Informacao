#!/bin/bash
# run_casia_to_nuaa.sh
# Executa todo o pipeline: extração LBP do CASIA, treino da SVM e teste no NUAA

echo "============================================================"
echo "Pipeline: Treinar SVM no CASIA e testar no NUAA"
echo "============================================================"

# Navegar para o diretório CASIA
cd "$(dirname "$0")"

echo ""
echo "[1/3] Extraindo features LBP do dataset CASIA..."
echo "------------------------------------------------------------"
python3 extract_lbp_casia.py
if [ $? -ne 0 ]; then
    echo "ERRO: Falha na extração de features do CASIA."
    exit 1
fi

echo ""
echo "[2/3] Treinando SVM com dataset CASIA..."
echo "------------------------------------------------------------"
python3 train_svm_casia.py
if [ $? -ne 0 ]; then
    echo "ERRO: Falha no treinamento da SVM."
    exit 2
fi

echo ""
echo "[3/3] Testando modelo no dataset NUAA..."
echo "------------------------------------------------------------"
python3 test_nuaa.py
if [ $? -ne 0 ]; then
    echo "ERRO: Falha no teste do modelo."
    exit 3
fi

echo ""
echo "============================================================"
echo "Pipeline concluído com sucesso!"
echo "============================================================"
echo ""
echo "Resultados disponíveis em:"
echo "  - results_casia/results_casia_to_nuaa.txt"
echo "  - results_casia/confusion_matrix_casia_to_nuaa.png"
echo "  - results_casia/roc_curve_casia_to_nuaa.png"
echo ""
