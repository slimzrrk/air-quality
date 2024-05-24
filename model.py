import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

# Spécifiez le chemin du fichier
file_path = '/content/sample_data/AirQualityUCI.csv'

# Charger les données en ignorant les erreurs initiales
df = pd.read_csv(file_path, sep=';', on_bad_lines='skip', skipinitialspace=True)

# Afficher les noms de colonnes originaux
print("Noms de colonnes avant nettoyage:")
print(df.columns)

# Renommer les colonnes pour enlever les espaces et caractères spéciaux
df.columns = df.columns.str.strip()  # Supprime les espaces autour des noms de colonnes
df.columns = df.columns.str.replace(' ', '_')  # Remplace les espaces internes par des underscores
df.columns = df.columns.str.replace('.', '')  # Supprime les points
df.columns = df.columns.str.replace('(', '')  # Supprime les parenthèses ouvrantes
df.columns = df.columns.str.replace(')', '')  # Supprime les parenthèses fermantes
df.columns = df.columns.str.replace('/', '_')  # Remplace les slashes par des underscores

# Afficher les noms de colonnes nettoyés
print("Noms de colonnes après nettoyage:")
print(df.columns)

# Correction des valeurs dans les colonnes numériques
for col in df.columns:
    df[col] = df[col].astype(str).str.strip().replace('', np.nan)
    if col not in ['Date', 'Time']:
        df[col] = df[col].str.replace(',', '.').astype(float)

# Afficher les premières lignes après la correction
print(df.head())

# 1. Vérification et traitement des valeurs manquantes
print(df.isnull().sum())

# Remplacer les valeurs manquantes par la moyenne de chaque colonne (si pertinent)
for column in df.columns:
    if df[column].dtype in [np.float64, np.int64]:
        df[column].fillna(df[column].mean(), inplace=True)
    else:
        df[column].fillna(df[column].mode()[0], inplace=True)

# 2. Correction des valeurs anormales
def detect_outliers(df, columns):
    outliers = {}
    for column in columns:
        if df[column].dtype in [np.float64, np.int64]:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers[column] = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

outliers = detect_outliers(df, df.columns)
for column, rows in outliers.items():
    median = df[column].median()
    df.loc[rows.index, column] = median

# 3. Uniformisation des formats de date et heure
df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H.%M.%S', errors='coerce')

# Suppression des anciennes colonnes de date et heure
df.drop(columns=['Date', 'Time'], inplace=True)

# 4. Suppression des données redondantes
df.drop_duplicates(inplace=True)

# 5. Normalisation des données
columns_to_normalize = ['COGT', 'PT08S1CO', 'NMHCGT', 'C6H6GT', 'PT08S2NMHC', 'NOxGT', 'PT08S3NOx', 'NO2GT', 'PT08S4NO2', 'PT08S5O3', 'T', 'RH', 'AH']
scaler = MinMaxScaler()
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# Afficher le DataFrame nettoyé
print(df.head())

# Enregistrer le DataFrame nettoyé dans un nouveau fichier CSV
df.to_csv('air_quality_data_cleaned.csv', index=False)
# Supprimer les colonnes spécifiées
df.drop(columns=['Unnamed:_15', 'Unnamed:_16', 'NMHCGT'], inplace=True)
df.head()


#Régression linéaire
# Séparation des données en ensemble d'entraînement et ensemble de test
X = df.drop(columns=['DateTime', 'COGT'])  # Features
y = df['COGT']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création et entraînement du modèle de régression linéaire
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred_linear = linear_model.predict(X_test)

# Évaluation du modèle
mse_linear = mean_squared_error(y_test, y_pred_linear)
print("Mean Squared Error (Régression linéaire):", mse_linear)

#Random Forest
from sklearn.ensemble import RandomForestRegressor

# Création et entraînement du modèle de Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred_rf = rf_model.predict(X_test)

# Évaluation du modèle
mse_rf = mean_squared_error(y_test, y_pred_rf)
print("Mean Squared Error (Random Forest):", mse_rf)


#Réseau de neurones
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Création du modèle séquentiel
nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1)
])

# Compilation du modèle
nn_model.compile(optimizer='adam', loss='mse')

# Entraînement du modèle
history = nn_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Évaluation du modèle
mse_nn = nn_model.evaluate(X_test, y_test)
print("Mean Squared Error (Réseau de neurones):", mse_nn)
