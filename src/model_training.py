"""
Entraînement du modèle prédictif d'engagement et sauvegarde sur disque.
Le modèle prédit un score d'engagement RELATIF (par rapport à la baseline
médiane de la page pour sa période), afin d'isoler l'effet du moment de
publication et du type de contenu, indépendamment de la taille d'audience
ou des dérives temporelles (ex: baisse générale d'engagement observée en 2022).

Deux familles de modèles sont entraînées et sauvegardées :
- Un modèle PAR ENTREPRISE (usage principal, plus précis) -> engagement_model_{entreprise}.pkl
- Un modèle GLOBAL avec "entreprise" en feature (comparaison concurrentielle) -> engagement_model_global.pkl
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERIC_FEATURES = ['hour', 'day_of_week', 'month', 'message_length',
                     'has_hashtag', 'has_link', 'has_emoji']
BOOL_FEATURES = ['is_weekend', 'is_business_hours']
CATEGORICAL_FEATURES_GLOBAL = ['entreprise', 'content_categorie', 'time_slot_id']
CATEGORICAL_FEATURES_PER_ENT = ['content_categorie', 'time_slot_id']
FEATURES_GLOBAL = NUMERIC_FEATURES + BOOL_FEATURES + CATEGORICAL_FEATURES_GLOBAL
FEATURES_PER_ENT = NUMERIC_FEATURES + BOOL_FEATURES + CATEGORICAL_FEATURES_PER_ENT
TARGET = 'total_engagement'

# Hyperparamètres retenus après comparaison par validation croisée (5-fold)
# entre configuration manuelle et RandomizedSearchCV -> le tuné gagne en moyenne.
XGB_PARAMS = dict(
    n_estimators=500, max_depth=8, learning_rate=0.01,
    subsample=0.7, colsample_bytree=1.0,
    min_child_weight=3, reg_alpha=0, reg_lambda=0.5,
    random_state=42, n_jobs=-1,
)


def prepare_model_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Prépare le dataframe pour la modélisation : features propres + cible normalisée."""
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['year'] = df['timestamp'].dt.year

    model_df = df[FEATURES_GLOBAL + [TARGET, 'timestamp', 'year']].copy()
    model_df = model_df.dropna(subset=FEATURES_GLOBAL + [TARGET])
    for c in BOOL_FEATURES:
        model_df[c] = model_df[c].astype(int)
    model_df['content_categorie'] = model_df['content_categorie'].fillna('unknown')

    # Normalisation : engagement relatif à la baseline (médiane) entreprise x année
    baseline = model_df.groupby(['entreprise', 'year'])[TARGET].transform('median')
    baseline = baseline.replace(0, np.nan)
    model_df['engagement_relative'] = model_df[TARGET] / baseline
    model_df = model_df.dropna(subset=['engagement_relative'])
    model_df['target_log'] = np.log1p(model_df['engagement_relative'])

    return model_df


def _build_pipeline(categorical_features: list) -> Pipeline:
    preprocessor = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ], remainder='passthrough')
    return Pipeline([
        ('prep', preprocessor),
        ('model', xgb.XGBRegressor(**XGB_PARAMS)),
    ])


def _fit_and_evaluate(model_df: pd.DataFrame, features: list, categorical_features: list,
                       stratify_col: str = None) -> tuple:
    """Entraîne + évalue un pipeline sur un split 80/20, retourne (pipeline, metrics)."""
    stratify = model_df[stratify_col] if stratify_col else None
    train_df, test_df = train_test_split(
        model_df, test_size=0.2, random_state=42, stratify=stratify
    )
    X_train, X_test = train_df[features], test_df[features]
    y_train_log, y_test_log = train_df['target_log'], test_df['target_log']
    y_test_raw = test_df['engagement_relative']

    pipeline = _build_pipeline(categorical_features)
    pipeline.fit(X_train, y_train_log)

    pred_log = pipeline.predict(X_test)
    pred_raw = np.clip(np.expm1(pred_log), 0, None)

    metrics = {
        'r2_log': float(r2_score(y_test_log, pred_log)),
        'rmse_log': float(np.sqrt(mean_squared_error(y_test_log, pred_log))),
        'mae_log': float(mean_absolute_error(y_test_log, pred_log)),
        'r2_raw': float(r2_score(y_test_raw, pred_raw)),
        'rmse_raw': float(np.sqrt(mean_squared_error(y_test_raw, pred_raw))),
        'mae_raw': float(mean_absolute_error(y_test_raw, pred_raw)),
        'n_train': int(len(train_df)),
        'n_test': int(len(test_df)),
    }
    return pipeline, metrics


def train_and_save_all_models(prepared_df: pd.DataFrame, output_dir: str = '../models') -> dict:
    """
    Entraîne et sauvegarde :
    - un modèle par entreprise (usage principal)
    - un modèle global (comparaison concurrentielle)

    Returns:
        dict récapitulatif de toutes les métriques.
    """
    model_df = prepare_model_dataframe(prepared_df)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_metrics = {'trained_at': pd.Timestamp.now().isoformat(), 'per_entreprise': {}, 'global': {}}

    # --- Modèles par entreprise ---
    for ent in sorted(model_df['entreprise'].unique()):
        sub = model_df[model_df['entreprise'] == ent].reset_index(drop=True)
        pipeline, metrics = _fit_and_evaluate(sub, FEATURES_PER_ENT, CATEGORICAL_FEATURES_PER_ENT)
        joblib.dump(pipeline, output_path / f'engagement_model_{ent}.pkl')
        all_metrics['per_entreprise'][ent] = metrics
        print(f"  ✓ Modèle {ent} sauvegardé — R² brut: {metrics['r2_raw']:.3f} "
              f"(n={len(sub)})")

    # --- Modèle global ---
    pipeline_global, metrics_global = _fit_and_evaluate(
        model_df, FEATURES_GLOBAL, CATEGORICAL_FEATURES_GLOBAL, stratify_col='entreprise'
    )
    joblib.dump(pipeline_global, output_path / 'engagement_model_global.pkl')
    metrics_global['entreprises'] = sorted(model_df['entreprise'].unique().tolist())
    all_metrics['global'] = metrics_global
    print(f"  ✓ Modèle global sauvegardé — R² brut: {metrics_global['r2_raw']:.3f}")

    # --- Métadonnées et données d'entraînement (pour pattern_generator) ---
    defaults = {
        'message_length': float(model_df['message_length'].median()),
        'has_hashtag': int(model_df['has_hashtag'].mode()[0]),
        'has_link': int(model_df['has_link'].mode()[0]),
        'has_emoji': int(model_df['has_emoji'].mode()[0]),
    }
    with open(output_path / 'model_metadata.json', 'w') as f:
        json.dump({
            'metrics': all_metrics,
            'features_global': FEATURES_GLOBAL,
            'features_per_entreprise': FEATURES_PER_ENT,
            'target': TARGET,
            'feature_defaults': defaults,
        }, f, indent=2, ensure_ascii=False)

    model_df.to_csv(output_path / 'model_training_data.csv', index=False)

    return all_metrics


# Alias rétrocompatible (ancien nom utilisé précédemment)
def train_and_save_model(prepared_df: pd.DataFrame, output_dir: str = '../models') -> dict:
    return train_and_save_all_models(prepared_df, output_dir)


if __name__ == '__main__':
    from data_pipeline import load_and_prepare_data

    files = {
        'Ooredoo': '../data/raw/posts_dataOOFB0.csv',
        'Orange': '../data/raw/posts_dataOrangeFB0.csv',
        'TT': '../data/raw/posts_dataTTFB0.csv',
    }
    prepared_df = load_and_prepare_data(files)
    print("Entraînement en cours...\n")
    all_metrics = train_and_save_all_models(prepared_df, output_dir='../models')

    print("\n✓ Tous les modèles entraînés et sauvegardés dans ../models/")
    print(json.dumps(all_metrics, indent=2, ensure_ascii=False))