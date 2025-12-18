import numpy as np
from sklearn.svm import SVC
import joblib
import os

SEEDS = [42]

for s in SEEDS:
    print(f"Treinando SVM para SEED {s}...")
    data_path = f'./Metodo_3/database2/data/seed_{s}'
    X_train = np.load(f'{data_path}/X_train.npy')
    y_train = np.load(f'{data_path}/y_train.npy')

    svm_clf = SVC(kernel='rbf', C=1.0, probability=True, random_state=s)
    svm_clf.fit(X_train, y_train)

    model_path = f'./Metodo_3/database2/models/seed_{s}'
    os.makedirs(model_path, exist_ok=True)
    joblib.dump(svm_clf, f'{model_path}/metodo3_svm.pkl')