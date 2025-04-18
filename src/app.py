# Código extraído desde explore.ipynb


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import requests


url= "https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv"
response_url= requests.get(url)

with open("/workspaces/Finarosalina_diabetes_machine-learning-py-template/data/raw/diabetes.csv", "wb") as file:
    file.write(response_url.content)

df = pd.read_csv('/workspaces/Finarosalina_diabetes_machine-learning-py-template/data/raw/diabetes.csv')

df.head()


df.info()

df.describe()

# Verificar si hay filas duplicadas
duplicados = df.duplicated()
print(df[duplicados])


# Correlaciones 
print("\n🔹 Correlación con la variable objetivo (Outcome):")
correlation = df.corr(numeric_only=True)
target_corr = correlation["Outcome"].sort_values(ascending=False)
print(target_corr)


# Mapa de calor
plt.figure(figsize=(14,10))
sns.heatmap(correlation, cmap='coolwarm', center=0, annot=True, fmt=".2f", linewidths=0.5)
plt.title("Mapa de calor de correlaciones")
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

# Crear un gráfico de caja para cada columna del DataFrame
plt.figure(figsize=(16,12))
for i, column in enumerate(df.columns):
    plt.subplot(3, 3, i+1)  # 3 filas, 3 columnas de gráficos
    sns.boxplot(x=df[column])
    plt.title(f"Outliers en {column}")

plt.tight_layout()
plt.show()


X = df.drop("Outcome", axis = 1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

from sklearn.feature_selection import SelectKBest, f_classif

selection_model = SelectKBest(score_func=f_classif, k=7)

selection_model.fit(X_train, y_train)
selected_columns = X_train.columns[selection_model.get_support()]

X_train_sel = pd.DataFrame(selection_model.transform(X_train), columns=selected_columns)

X_test_sel = pd.DataFrame(selection_model.transform(X_test), columns=selected_columns)


X_train_sel.head()

X_train_sel["Outcome"] = y_train.values
X_train_sel.head()

X_test_sel.head()


X_test_sel["Outcome"] = y_test.values
X_test_sel.head()

X_train_sel.to_csv("/workspaces/Finarosalina_diabetes_machine-learning-py-template/data/processed/clean_train.csv", index = False)
X_test_sel.to_csv("/workspaces/Finarosalina_diabetes_machine-learning-py-template/data/processed/clean_test.csv", index = False)

X_train = X_train_sel.drop(["Outcome"], axis = 1)
y_train = X_train_sel["Outcome"]
X_test = X_test_sel.drop(["Outcome"], axis = 1)
y_test = X_test_sel["Outcome"]

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=5, random_state = 42)
model.fit(X_train, y_train)

from sklearn import tree

fig = plt.figure(figsize=(9,7))

tree.plot_tree(model, feature_names = list(X_train.columns), class_names = ["0", "1", "2"], filled = True)

plt.show()

y_pred = model.predict(X_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

from pickle import dump

dump(model, open("/workspaces/Finarosalina_diabetes_machine-learning-py-template/models/decision_tree_classifier_default_42.sav", "wb"))

import json

# Rutas de archivo
notebook_path = "/workspaces/Finarosalina_diabetes_machine-learning-py-template/src/explore.ipynb"
output_path = "/workspaces/Finarosalina_diabetes_machine-learning-py-template/src/app.py"

# Leer el archivo .ipynb como JSON
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Extraer el código de las celdas tipo "code"
code_cells = []
for cell in notebook.get('cells', []):
    if cell.get('cell_type') == 'code':
        code = ''.join(cell.get('source', []))
        code_cells.append(code)

# Combinar el código y escribirlo en el archivo .py
with open(output_path, 'w', encoding='utf-8') as f:
    f.write("# Código extraído desde explore.ipynb\n\n")
    f.write("\n\n".join(code_cells))

print("✅ Código copiado exitosamente a app.py")
