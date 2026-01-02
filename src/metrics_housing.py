import pandas as pd
import os

# Cargar dataset
df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'housing.csv'))

print(df.head())
print(df.info())

#############################################################################################################
# Target (lo que queremos predecir)
y = df["price"]
# Features (los parametros que le pasamos para que prediga. todo menos el precio)
X = df.drop(columns=["price"])

#############################################################################################################
# Separar train y test. 80% aprende, 20% examinado.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#############################################################################################################
# Los modelos no entienden texto (yes, no, furnished). Solución: One-Hot Encoding
# Las columnas categóricas se convierten en 0/1, Las numéricas pasan tal cual

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

categorical_cols = X.select_dtypes(include="object").columns
numeric_cols = X.select_dtypes(exclude="object").columns

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

#############################################################################################################
# Modelos a comparar
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
}

#############################################################################################################
# Entrenar y evaluar con métricas

from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

results = []

for name, model in models.items():

    pipe = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    results.append([name, mae, rmse, r2])

results_df = pd.DataFrame(
    results, columns=["Model", "MAE", "RMSE", "R2"]
)
#print(results_df)




# #############################################################################################################
# #
# Cómo leer este gráfico:

# línea diagonal → predicción perfecta
# puntos cerca → buen modelo
# nube dispersa → error
# forma curva → modelo mal ajustado

plt.scatter(y_test, preds, alpha=0.6)
plt.xlabel("Precio real")
plt.ylabel("Precio predicho")
plt.title("Real vs Predicho")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         "--")
plt.show()

# #############################################################################################################
# #
# centro en 0 → sin sesgo
# cola larga → outliers
# asimetría → modelo subestima o sobreestima

# DISTRIBUCIÓN DE ERRORES
errors = y_test - preds
plt.hist(errors, bins=30)
plt.title("Distribución de errores")
plt.xlabel("Error (real - predicho)")
plt.show()

# LAS CASAS DONDE FALLA MÁS
error_df = X_test.copy()
error_df["real_price"] = y_test
error_df["pred_price"] = preds
error_df["abs_error"] = abs(y_test - preds)

print(error_df.sort_values("abs_error", ascending=False).head(10))

# ERROR RELATIVO
error_df["error_pct"] = error_df["abs_error"] / error_df["real_price"] * 100
print(error_df.sort_values("error_pct", ascending=False).head(10))

# Vamos a traducir observación a variables claras.
X["many_bedrooms"] = (X["bedrooms"] >= 3).astype(int)
X["multi_stories"] = (X["stories"] >= 2).astype(int)

X["large_and_multi"] = (
    (X["bedrooms"] >= 3) &
    (X["stories"] >= 2)
).astype(int)

# Reentrenar el pipeline
X_fe = X.copy()

X_fe["many_bedrooms"] = (X_fe["bedrooms"] >= 3).astype(int)
X_fe["multi_stories"] = (X_fe["stories"] >= 2).astype(int)
X_fe["large_and_multi"] = (
    (X_fe["bedrooms"] >= 3) &
    (X_fe["stories"] >= 2)
).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X_fe, y, test_size=0.2, random_state=42
)

#Vuelve a entrenar el mejor modelo (por ejemplo Random Forest
pipe.fit(X_train, y_train)
preds = pipe.predict(X_test)

print("MAE:", mean_absolute_error(y_test, preds))
print("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))
print("R2:", r2_score(y_test, preds))

# Repetir el análisis de errores SOLO para casas problemáticas
mask = (X_test["bedrooms"] >= 3) & (X_test["stories"] >= 2)
print("MAE casas grandes:",
      mean_absolute_error(y_test[mask], preds[mask]))


