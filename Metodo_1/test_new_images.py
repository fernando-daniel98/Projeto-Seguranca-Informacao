import argparse
from pathlib import Path
import cv2
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# Import feature extraction from the existing module
from extract_lbp import extract_lbp_features

def load_images_from_folder(folder):
    images = []
    filenames = []
    img_exts = (".jpg", ".jpeg", ".png", ".bmp")
    
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith(img_exts):
            img_path = folder / fname
            try:
                # Extract features immediately to save memory/time
                feat = extract_lbp_features(img_path)
                images.append(feat)
                filenames.append(fname)
            except Exception as e:
                print(f"[ERRO] Falha ao processar {fname}: {e}")
                
    return np.array(images), filenames

def main():
    parser = argparse.ArgumentParser(description="Testa o modelo SVM treinado com novas imagens.")
    parser.add_argument("--dir", type=str, required=True, help="Caminho para a pasta com as imagens.")
    args = parser.parse_args()

    target_dir = Path(args.dir)
    if not target_dir.exists():
        print(f"Erro: O diretório '{target_dir}' não existe.")
        return

    # Load model
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "models" / "metodo1_lbp_svm.pkl"
    
    if not model_path.exists():
        print(f"Erro: Modelo não encontrado em {model_path}")
        print("Execute svm.py primeiro para treinar o modelo.")
        return

    print(f"Carregando modelo de {model_path}...")
    model = joblib.load(model_path)

    # Check structure for Validation Mode (real/fake folders)
    real_dir = target_dir / "real"
    fake_dir = target_dir / "fake"

    if real_dir.exists() and fake_dir.exists():
        print(f"\n=== MODO VALIDAÇÃO (Pastas 'real' e 'fake' encontradas) ===")
        import os # Ensure os is imported for os.walk/listdir if needed, though pathlib is mostly used
        
        X_list = []
        y_list = []
        
        # Process Real (Label 1)
        print(f"Processando 'real' em {real_dir}...")
        for root, _, files in os.walk(real_dir):
            for fname in files:
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    try:
                        feat = extract_lbp_features(Path(root) / fname)
                        X_list.append(feat)
                        y_list.append(1)
                    except Exception as e:
                        print(f"Erro em {fname}: {e}")

        # Process Fake (Label 0)
        print(f"Processando 'fake' em {fake_dir}...")
        for root, _, files in os.walk(fake_dir):
            for fname in files:
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    try:
                        feat = extract_lbp_features(Path(root) / fname)
                        X_list.append(feat)
                        y_list.append(0)
                    except Exception as e:
                        print(f"Erro em {fname}: {e}")

        if not X_list:
            print("Nenhuma imagem válida encontrada.")
            return

        X = np.stack(X_list)
        y = np.array(y_list)
        
        print(f"\nTotal de imagens: {len(y)}")
        print(f"Real: {np.sum(y==1)}, Fake: {np.sum(y==0)}")
        
        # Predict
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        
        print(f"\nAcurácia: {acc:.4f}")
        print("\nRelatório de Classificação:")
        print(classification_report(y, y_pred, target_names=["fake (0)", "real (1)"]))

    else:
        print(f"\n=== MODO INFERÊNCIA (Imagens soltas na pasta) ===")
        import os
        
        X_list = []
        filenames = []
        
        print(f"Lendo imagens de {target_dir}...")
        for fname in sorted(os.listdir(target_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                img_path = target_dir / fname
                try:
                    feat = extract_lbp_features(img_path)
                    X_list.append(feat)
                    filenames.append(fname)
                except Exception as e:
                    print(f"Erro ao ler {fname}: {e}")
        
        if not X_list:
            print("Nenhuma imagem encontrada.")
            return
            
        X = np.stack(X_list)
        
        # Predict probabilities
        probs = model.predict_proba(X)
        preds = model.predict(X)
        
        print(f"\n{'Imagem':<30} | {'Predição':<10} | {'Confiança (Real)':<10}")
        print("-" * 60)
        
        for i, fname in enumerate(filenames):
            pred_label = "REAL" if preds[i] == 1 else "FAKE"
            prob_real = probs[i][1]
            print(f"{fname:<30} | {pred_label:<10} | {prob_real:.4f}")

if __name__ == "__main__":
    main()
