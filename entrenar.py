import os, glob
import numpy as np
import joblib
from caracteristicas import caracteristicas
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression

X, y = [], []
for persona in sorted(os.listdir("dataset")):
    for archivo in glob.glob(os.path.join("dataset", persona, "*.npy")):
        X.append(caracteristicas(np.load(archivo), usar_z=True))
        y.append(persona)
X, y = np.array(X), np.array(y)

personas = sorted(set(y))
print(f"{len(X)} muestras, {len(personas)} personas: {personas}")
if len(personas) < 2:
    raise SystemExit("Necesitas al menos 2 personas para entrenar.")

modelos = {
    "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "SVM lineal": make_pipeline(StandardScaler(), SVC(kernel="linear", probability=True)),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
mejor, mejor_acc = None, -1
for nombre, modelo in modelos.items():
    acc = cross_val_score(modelo, X, y, cv=cv).mean()
    print(f"{nombre}: precisión {acc * 100:.1f}%")
    if acc > mejor_acc:
        mejor, mejor_acc, mejor_nombre = modelo, acc, nombre

mejor.fit(X, y)
joblib.dump(mejor, "modelo.joblib")
print(f"Guardado modelo.joblib ({mejor_nombre})")
