import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, BaggingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 0. PRZYGOTOWANIE DANYCH
# ==========================================
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results = {}
models = {} # Słownik do przechowywania wytrenowanych modeli na potrzeby wykresów

print("--- ROZPOCZĘCIE ANALIZY ---")

# ==========================================
# 1. KROSWALIDACJA I OPTYMALIZACJA PARAMETRÓW
# ==========================================
print("\n[Krok 1] Optymalizacja modeli bazowych...")

# Sieć neuronowa (MLP)
mlp = MLPClassifier(max_iter=500, random_state=42)
param_mlp = {'hidden_layer_sizes': [(10,), (50,), (50, 20)]}
grid_mlp = GridSearchCV(mlp, param_mlp, cv=5, scoring='accuracy').fit(X_train, y_train)
best_mlp = grid_mlp.best_estimator_

# k-Najbliższych Sąsiadów (kNN)
knn = KNeighborsClassifier()
param_knn = {'n_neighbors': [3, 5, 7, 11]}
grid_knn = GridSearchCV(knn, param_knn, cv=5, scoring='accuracy').fit(X_train, y_train)
best_knn = grid_knn.best_estimator_

# Drzewo Decyzyjne (DT)
dt = DecisionTreeClassifier(random_state=42)
param_dt = {'max_depth': [3, 5, 10, None]}
grid_dt = GridSearchCV(dt, param_dt, cv=5, scoring='accuracy').fit(X_train, y_train)
best_dt = grid_dt.best_estimator_

# Zapisywanie wyników i modeli bazowych
models['MLP (Optymalny)'] = best_mlp
models['kNN (Optymalny)'] = best_knn
models['DT (Optymalny)'] = best_dt

for name, model in models.items():
    results[name] = accuracy_score(y_test, model.predict(X_test))

print(f"Najlepsze parametry NN: {grid_mlp.best_params_}")
print(f"Najlepsze parametry kNN: {grid_knn.best_params_}")
print(f"Najlepsze parametry DT: {grid_dt.best_params_}")

# ==========================================
# 2. KOMITETY (ENSEMBLE METHODS)
# ==========================================
print("\n[Krok 2] Budowanie komitetów...")

# Modele bazowe
estimators_opt = [('mlp', best_mlp), ('knn', best_knn), ('dt', best_dt)]

# a) Voting
voting_hard = VotingClassifier(estimators=estimators_opt, voting='hard').fit(X_train, y_train)
voting_soft = VotingClassifier(estimators=estimators_opt, voting='soft').fit(X_train, y_train)

# b) Bagging
bagging_def = BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42), n_estimators=50, random_state=42).fit(X_train, y_train)
bagging_opt = BaggingClassifier(estimator=best_dt, n_estimators=50, random_state=42).fit(X_train, y_train)

# c) Stacking
stacking_lr = StackingClassifier(estimators=estimators_opt, final_estimator=LogisticRegression(), cv=5).fit(X_train, y_train)
stacking_dt = StackingClassifier(estimators=estimators_opt, final_estimator=DecisionTreeClassifier(max_depth=3, random_state=42), cv=5).fit(X_train, y_train)

# Zapisywanie modeli komitetowych
models['Voting (Hard)'] = voting_hard
models['Voting (Soft)'] = voting_soft
models['Bagging (DT Domyślny)'] = bagging_def
models['Bagging (DT Optymalny)'] = bagging_opt
models['Stacking (Meta: Regresja Logistyczna)'] = stacking_lr
models['Stacking (Meta: Drzewo Decyzyjne)'] = stacking_dt

# Obliczanie wyników dla komitetów
for name in list(models.keys())[3:]: # Pomijamy pierwsze 3, bo już są w results
    results[name] = accuracy_score(y_test, models[name].predict(X_test))

# ==========================================
# 3. WYBÓR, KOŃCOWE TRENOWANIE I WIZUALIZACJA
# ==========================================
print("\n[Krok 3] Podsumowanie i wybór najlepszego modelu...")

# Prezentacja wyników w formie tabeli
df_results = pd.DataFrame(list(results.items()), columns=['Model / Komitet', 'Dokładność (Accuracy)'])
df_results = df_results.sort_values(by='Dokładność (Accuracy)', ascending=False)
print("\n", df_results.to_string(index=False))

best_model_name = df_results.iloc[0]['Model / Komitet']
best_accuracy = df_results.iloc[0]['Dokładność (Accuracy)']
print(f"\n-> Najlepszym modelem okazał się: {best_model_name} z dokładnością {best_accuracy:.4f}")

print("\n[Krok 4] Generowanie wykresów...")

# --- WYKRES 1: Porównanie dokładności modeli ---
plt.figure(figsize=(12, 6))
sns.barplot(
    x='Dokładność (Accuracy)', 
    y='Model / Komitet', 
    data=df_results, 
    palette='viridis',
    hue='Model / Komitet',
    legend=False
)
plt.title('Porównanie dokładności modeli klasyfikacyjnych', fontsize=14)
plt.xlabel('Dokładność (Accuracy)', fontsize=12)
plt.ylabel('Model', fontsize=12)
# Ustawienie limitu osi X, aby lepiej uwidocznić różnice (dane z reguły mają >0.85 accuracy)
plt.xlim(df_results['Dokładność (Accuracy)'].min() - 0.05, 1.0)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# --- WYKRES 2: Macierz pomyłek dla najlepszego modelu ---
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues', 
    xticklabels=data.target_names, 
    yticklabels=data.target_names
)
plt.title(f'Macierz pomyłek\nNajlepszy model: {best_model_name}', fontsize=14)
plt.xlabel('Przewidywana klasa', fontsize=12)
plt.ylabel('Rzeczywista klasa', fontsize=12)
plt.tight_layout()
plt.show()