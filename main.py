import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt

# Usamos 'r' para que Windows reconozca bien la ruta de las carpetas
ruta = r"C:\Users\angel\Desktop\Mineria de Datos\Dataset Tarea 1\insurance.csv"
df = pd.read_csv(ruta)

# Preparamos los datos (Convertimos 'smoker' a números: yes=1, no=0)
df['smoker_num'] = df['smoker'].map({'yes': 1, 'no': 0})

# Seleccionamos variables: Edad y si es Fumador para predecir Cargos
X = df[['age', 'smoker_num']]
y = df['charges']

# Creamos y entrenamos el modelo
model = LinearRegression()
model.fit(X, y)

# Mostramos los resultados clave
print(f"Intercepción (Costo base): {model.intercept_:.2f}")
print(f"Coeficiente Edad: {model.coef_[0]:.2f}")
print(f"Coeficiente Fumador: {model.coef_[1]:.2f}")

sns.lmplot(x='age', y='charges', hue='smoker', data=df, aspect=1.5)
plt.title("Regresión Lineal: Edad vs Cargos por Hábito de Fumar")
plt.show()
