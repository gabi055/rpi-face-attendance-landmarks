import os, glob
import numpy as np
from caracteristicas import caracteristicas
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

def cargar(usar_z):
    X, y = [], []
    for persona in sorted(os.listdir("dataset")):
        for a in glob.glob(os.path.join("dataset", persona, "*.npy")):
            X.append(caracteristicas(np.load(a), usar_z=usar_z))
            y.append(persona)
    return np.array(X), np.array(y)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
modelos = {
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3)),
    "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf")),
    "SVM lineal": make_pipeline(StandardScaler(), SVC(kernel="linear")),
    "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
}
for usar_z in (False, True):
    X, y = cargar(usar_z)
    print(f"\n--- {'x,y,z' if usar_z else 'solo x,y'} ({X.shape[1]} características) ---")
    for nombre, m in modelos.items():
        print(f"{nombre}: {cross_val_score(m, X, y, cv=cv).mean() * 100:.1f}%")
