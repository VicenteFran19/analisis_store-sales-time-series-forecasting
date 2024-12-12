import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import glob

# Paso 1: Cargar todos los archivos CSV
csv_files = glob.glob("*.csv")
dataframes = {}

for file in csv_files:
    df_name = file.split(".")[0]
    dataframes[df_name] = pd.read_csv(file)
    print(f"Archivo {file} cargado con {dataframes[df_name].shape[0]} filas y {dataframes[df_name].shape[1]} columnas.")

# Paso 2: Unir la información del archivo 'train.csv' con otros archivos relevantes
sales_data = dataframes['train']

# Incorporar información de otros archivos (si aplica)
if 'oil' in dataframes:
    oil_data = dataframes['oil']
    oil_data['date'] = pd.to_datetime(oil_data['date'])
    sales_data['date'] = pd.to_datetime(sales_data['date'])
    sales_data = sales_data.merge(oil_data, on='date', how='left')

if 'holidays_events' in dataframes:
    holidays = dataframes['holidays_events']
    holidays['date'] = pd.to_datetime(holidays['date'])
    holidays['holiday_flag'] = 1
    holidays = holidays[['date', 'holiday_flag']].drop_duplicates()
    sales_data = sales_data.merge(holidays, on='date', how='left')
    sales_data['holiday_flag'] = sales_data['holiday_flag'].fillna(0)

# Paso 3: Análisis Exploratorio de Datos (EDA)
print("\nInformación general del dataset unificado:")
sales_data.info()

# Distribución de ventas diarias
sns.histplot(sales_data['sales'], kde=True)
plt.title('Distribución de las Ventas Diarias')
plt.show()

# Verificación de valores nulos
def check_missing_values(df):
    print("\nValores nulos por columna:\n", df.isnull().sum())

check_missing_values(sales_data)

# Imputación o eliminación de valores nulos
sales_data = sales_data.fillna(0)  # Imputación con ceros (puedes cambiarla si lo prefieres)

# Paso 4: Conversión de fechas a componentes temporales
sales_data['year'] = sales_data['date'].dt.year
sales_data['month'] = sales_data['date'].dt.month
sales_data['day'] = sales_data['date'].dt.day
sales_data['day_of_week'] = sales_data['date'].dt.dayofweek

# Paso 5: Codificación de variables categóricas
label_encoders = {}
for col in ['store_nbr', 'family']:
    le = LabelEncoder()
    sales_data[col] = le.fit_transform(sales_data[col])
    label_encoders[col] = le

# Paso 6: Selección de variables predictoras y objetivo
y = sales_data['sales']
X = sales_data.drop(columns=['id', 'date', 'sales'])

# División del dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Paso 7: Normalización de los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Paso 8: Modelo 1 - Regresión Lineal
print("\nEntrenando el modelo de Regresión Lineal...")
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# Predicciones y evaluación de la Regresión Lineal
y_pred_reg = reg_model.predict(X_test)
print("\nMétricas de rendimiento para la Regresión Lineal:")
mse_reg = mean_squared_error(y_test, y_pred_reg)
rmse_reg = np.sqrt(mse_reg)
r2_reg = r2_score(y_test, y_pred_reg)
print(f'MSE: {mse_reg}, RMSE: {rmse_reg}, R2: {r2_reg}')

# Paso 9: Modelo 2 - Red Neuronal (MLP Regressor)
print("\nEntrenando la Red Neuronal...")
nn_model = MLPRegressor(hidden_layer_sizes=(64, 32, 16), max_iter=500, learning_rate_init=0.01, random_state=42)
nn_model.fit(X_train, y_train)

# Predicciones y evaluación de la Red Neuronal
y_pred_nn = nn_model.predict(X_test)
print("\nMétricas de rendimiento para la Red Neuronal:")
mse_nn = mean_squared_error(y_test, y_pred_nn)
rmse_nn = np.sqrt(mse_nn)
r2_nn = r2_score(y_test, y_pred_nn)
print(f'MSE: {mse_nn}, RMSE: {rmse_nn}, R2: {r2_nn}')

# Comparación de los modelos
print("\nComparación de los modelos:")
print(f"Regresión Lineal - MSE: {mse_reg}, RMSE: {rmse_reg}, R2: {r2_reg}")
print(f"Red Neuronal - MSE: {mse_nn}, RMSE: {rmse_nn}, R2: {r2_nn}")

# Visualización de los errores
error_df = pd.DataFrame({'Modelo': ['Regresión Lineal', 'Red Neuronal'], 
                         'MSE': [mse_reg, mse_nn], 
                         'RMSE': [rmse_reg, rmse_nn], 
                         'R2': [r2_reg, r2_nn]})

sns.barplot(x='Modelo', y='MSE', data=error_df)
plt.title('Comparación de MSE de los Modelos')
plt.show()

sns.barplot(x='Modelo', y='RMSE', data=error_df)
plt.title('Comparación de RMSE de los Modelos')
plt.show()

sns.barplot(x='Modelo', y='R2', data=error_df)
plt.title('Comparación de R2 de los Modelos')
plt.show()

# Reflexión sobre los modelos
if r2_nn > r2_reg:
    print("\nLa Red Neuronal muestra un mejor rendimiento predictivo en términos de R2.")
else:
    print("\nEl modelo de Regresión Lineal muestra un mejor rendimiento predictivo en términos de R2.")
