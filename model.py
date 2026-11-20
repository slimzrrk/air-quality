"""
Air Quality — prédiction de la concentration de monoxyde de carbone (CO)
=======================================================================

Pipeline complet sur le dataset « Air Quality » de l'UCI :
    1. chargement et nettoyage des données brutes (format européen, valeurs -200) ;
    2. pré-traitement sans fuite de données (imputation, écrêtage des valeurs
       aberrantes, normalisation) appris uniquement sur le jeu d'entraînement ;
    3. entraînement et comparaison de trois modèles de régression :
       régression linéaire, Random Forest et réseau de neurones (Keras) ;
    4. export des données nettoyées, des métriques et d'un graphique de résultats.

Usage :
    python model.py                      # exécution complète
    python model.py --skip-nn            # sans TensorFlow (plus rapide)
    python model.py --data mon_fichier.csv --output-dir resultats/
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "AirQualityUCI.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs"

TARGET = "COGT"            # CO(GT) : concentration réelle de CO en mg/m³
MISSING_SENTINEL = -200    # code utilisé par le dataset UCI pour « valeur manquante »
DROPPED_COLUMNS = ["NMHCGT"]  # NMHC(GT) : ~90 % de valeurs manquantes, inexploitable
SEED = 42


# --------------------------------------------------------------------------- #
# 1. Chargement et nettoyage                                                   #
# --------------------------------------------------------------------------- #
def load_raw_data(path: Path) -> pd.DataFrame:
    """Lit le CSV UCI (séparateur ';' et virgule décimale)."""
    df = pd.read_csv(path, sep=";", decimal=",")
    # Le fichier se termine par des lignes vides et chaque ligne par ';;',
    # ce qui crée des lignes et deux colonnes « Unnamed » entièrement vides.
    return df.dropna(how="all").dropna(axis=1, how="all")


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """'PT08.S1(CO)' -> 'PT08S1CO', 'CO(GT)' -> 'COGT', etc."""
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.replace(" ", "_", regex=False)
        .str.replace(r"[.()]", "", regex=True)
        .str.replace("/", "_", regex=False)
    )
    return df


def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    df = clean_column_names(raw)

    # Les valeurs -200 ne sont pas des mesures mais des données manquantes.
    numeric_cols = df.columns.difference(["Date", "Time"])
    df[numeric_cols] = df[numeric_cols].replace(MISSING_SENTINEL, np.nan)

    df = df.drop(columns=DROPPED_COLUMNS)

    # Date (jj/mm/aaaa) + heure (hh.mm.ss) -> un seul horodatage.
    df["DateTime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format="%d/%m/%Y %H.%M.%S"
    )
    df = df.drop(columns=["Date", "Time"]).drop_duplicates()
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2. Pré-traitement                                                            #
# --------------------------------------------------------------------------- #
class IQRClipper(BaseEstimator, TransformerMixin):
    """Ramène les valeurs aberrantes dans [Q1 - k·IQR, Q3 + k·IQR].

    Les bornes sont apprises sur le jeu d'entraînement uniquement (fit), puis
    appliquées telles quelles au jeu de test (transform).
    """

    def __init__(self, k: float = 1.5):
        self.k = k

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1, q3 = np.nanpercentile(X, [25, 75], axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.k * iqr
        self.upper_ = q3 + self.k * iqr
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)


def build_preprocessor():
    return make_pipeline(
        SimpleImputer(strategy="median"),  # valeurs manquantes -> médiane
        IQRClipper(k=1.5),                 # valeurs aberrantes -> bornes IQR
        MinMaxScaler(),                    # normalisation dans [0, 1]
    )


# --------------------------------------------------------------------------- #
# 3. Modèles                                                                   #
# --------------------------------------------------------------------------- #
def evaluate(name: str, y_true, y_pred) -> dict:
    return {
        "Modèle": name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }


def train_neural_network(X_train, y_train, epochs: int):
    """Réseau dense 64-64-1 (Keras). Retourne None si TensorFlow est absent."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # masque les logs C++ de TF
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")
    try:
        # On importe Keras 3 directement plutôt que tf.keras : cela fonctionne
        # même si TF_USE_LEGACY_KERAS est défini dans l'environnement.
        import keras
    except ImportError:
        print("  TensorFlow/Keras n'est pas installé : réseau de neurones ignoré.")
        return None

    keras.utils.set_random_seed(SEED)
    model = keras.Sequential(
        [
            keras.Input(shape=(X_train.shape[1],)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0,
    )
    print(f"  Entraînement arrêté après {len(history.history['loss'])} époques.")
    return model


# --------------------------------------------------------------------------- #
# 4. Visualisation                                                             #
# --------------------------------------------------------------------------- #
THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e",
        "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
        "accent": "#2a78d6", "other": "#c3c2b7",
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7",
        "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
        "accent": "#3987e5", "other": "#52514e",
    },
}


def plot_results(metrics: pd.DataFrame, y_test, best_pred, out_dir: Path) -> list[Path]:
    """Deux panneaux : RMSE par modèle + prédit vs mesuré pour le meilleur modèle.

    Une version claire et une version sombre sont générées (pour le README).
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib n'est pas installé : graphique ignoré.")
        return []

    ranked = metrics.sort_values("RMSE", ascending=False)  # meilleur en haut
    best = metrics.loc[metrics["RMSE"].idxmin()]
    lim = float(max(np.max(y_test), np.max(best_pred))) * 1.05
    paths = []

    for theme_name, c in THEMES.items():
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(11, 4.4), dpi=160,
            gridspec_kw={"width_ratios": [1, 1.05], "wspace": 0.32},
        )
        fig.patch.set_facecolor(c["surface"])

        for ax in (ax1, ax2):
            ax.set_facecolor(c["surface"])
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(c["axis"])
                ax.spines[side].set_linewidth(0.8)
            ax.tick_params(colors=c["muted"], labelsize=9, length=0, pad=6)
            ax.set_axisbelow(True)

        # Panneau 1 : RMSE par modèle (le meilleur est mis en avant)
        colors = [c["accent"] if m == best["Modèle"] else c["other"] for m in ranked["Modèle"]]
        ax1.barh(ranked["Modèle"], ranked["RMSE"], height=0.38, color=colors)
        for y, value in enumerate(ranked["RMSE"]):
            ax1.text(value + ranked["RMSE"].max() * 0.02, y, f"{value:.3f}",
                     va="center", fontsize=9.5, color=c["ink2"])
        ax1.set_xlim(0, ranked["RMSE"].max() * 1.22)
        ax1.grid(axis="x", color=c["grid"], linewidth=0.8)
        ax1.tick_params(axis="y", colors=c["ink"], labelsize=10)
        ax1.spines["left"].set_visible(False)
        ax1.set_xlabel("RMSE sur le jeu de test (mg/m³) — plus bas = meilleur",
                       color=c["ink2"], fontsize=9.5, labelpad=8)
        ax1.set_title("Erreur par modèle", loc="left", color=c["ink"],
                      fontsize=12, fontweight="bold", pad=12)

        # Panneau 2 : prédit vs mesuré pour le meilleur modèle
        ax2.plot([0, lim], [0, lim], color=c["muted"], linewidth=1, zorder=1)
        ax2.scatter(y_test, best_pred, s=9, alpha=0.35, color=c["accent"],
                    linewidths=0, zorder=2)
        ax2.text(0.04, 0.95, f"R² = {best['R2']:.3f}\nMAE = {best['MAE']:.3f} mg/m³",
                 transform=ax2.transAxes, va="top", fontsize=9.5, color=c["ink2"])
        ax2.text(lim * 0.80, lim * 0.84, "prédiction parfaite", ha="right", va="bottom",
                 fontsize=8.5, color=c["muted"])
        ax2.set_xlim(0, lim)
        ax2.set_ylim(0, lim)
        ax2.grid(color=c["grid"], linewidth=0.8)
        ax2.set_xlabel("CO mesuré (mg/m³)", color=c["ink2"], fontsize=9.5, labelpad=8)
        ax2.set_ylabel("CO prédit (mg/m³)", color=c["ink2"], fontsize=9.5, labelpad=8)
        ax2.set_title(f"Prédit vs mesuré — {best['Modèle']}", loc="left",
                      color=c["ink"], fontsize=12, fontweight="bold", pad=12)

        path = out_dir / f"results_{theme_name}.png"
        fig.savefig(path, facecolor=c["surface"], bbox_inches="tight", pad_inches=0.25)
        plt.close(fig)
        paths.append(path)
    return paths


# --------------------------------------------------------------------------- #
# Programme principal                                                          #
# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="chemin du CSV UCI (défaut : AirQualityUCI.csv)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="dossier de sortie (défaut : outputs/)")
    parser.add_argument("--epochs", type=int, default=200,
                        help="nombre max. d'époques du réseau de neurones (arrêt anticipé actif)")
    parser.add_argument("--skip-nn", action="store_true",
                        help="ne pas entraîner le réseau de neurones")
    parser.add_argument("--no-plot", action="store_true",
                        help="ne pas générer le graphique de résultats")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("1. Chargement et nettoyage des données")
    raw = load_raw_data(args.data)
    df = clean_data(raw)
    print(f"  {len(raw)} mesures horaires du {df['DateTime'].min():%d/%m/%Y} "
          f"au {df['DateTime'].max():%d/%m/%Y}")
    missing = (df.isna().mean() * 100).round(1)
    print("  Valeurs manquantes (%) :",
          ", ".join(f"{col} {pct}" for col, pct in missing.items() if pct > 0))
    cleaned_path = args.output_dir / "air_quality_cleaned.csv"
    df.to_csv(cleaned_path, index=False)
    print(f"  Données nettoyées -> {cleaned_path}")

    # On n'entraîne que sur les heures où la cible a réellement été mesurée :
    # imputer la cible reviendrait à apprendre au modèle des valeurs inventées.
    data = df.dropna(subset=[TARGET])
    X = data.drop(columns=["DateTime", TARGET])
    y = data[TARGET].to_numpy()
    print(f"  {len(data)} mesures avec CO renseigné, {X.shape[1]} variables explicatives")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(X_train)  # appris sur le train uniquement
    X_test = preprocessor.transform(X_test)

    print("\n2. Entraînement des modèles")
    results, predictions = [], {}

    print("  Régression linéaire…")
    linear = LinearRegression().fit(X_train, y_train)
    predictions["Régression linéaire"] = linear.predict(X_test)

    print("  Random Forest…")
    forest = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
    forest.fit(X_train, y_train)
    predictions["Random Forest"] = forest.predict(X_test)

    if not args.skip_nn:
        print("  Réseau de neurones…")
        nn = train_neural_network(X_train, y_train, args.epochs)
        if nn is not None:
            predictions["Réseau de neurones"] = nn.predict(X_test, verbose=0).ravel()

    for name, y_pred in predictions.items():
        results.append(evaluate(name, y_test, y_pred))
    metrics = pd.DataFrame(results)

    print("\n3. Résultats sur le jeu de test (CO en mg/m³)")
    print(metrics.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)

    best = metrics.loc[metrics["RMSE"].idxmin(), "Modèle"]
    print(f"\n  Meilleur modèle : {best}")

    importances = pd.Series(forest.feature_importances_, index=X.columns)
    print("\n  Variables les plus importantes (Random Forest) :")
    for col, imp in importances.sort_values(ascending=False).head(5).items():
        print(f"    {col:<12} {imp:.3f}")

    if not args.no_plot:
        for path in plot_results(metrics, y_test, predictions[best], args.output_dir):
            print(f"  Graphique -> {path}")


if __name__ == "__main__":
    main()
