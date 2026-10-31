"""
Pipeline de préparation des données réseaux sociaux (Facebook).
Fusionne plusieurs fichiers d'entreprises concurrentes, nettoie, et
enrichit avec des features temporelles et de contenu.
"""

import pandas as pd
import numpy as np
from datetime import datetime


TIME_SLOTS = {
    'night_late': {'start': '00:00', 'end': '06:00', 'id': 0},
    'early_morning': {'start': '06:00', 'end': '09:00', 'id': 1},
    'work_morning': {'start': '09:00', 'end': '12:00', 'id': 2},
    'lunch_break': {'start': '12:00', 'end': '15:00', 'id': 3},
    'work_afternoon': {'start': '15:00', 'end': '18:00', 'id': 4},
    'evening_prime': {'start': '18:00', 'end': '21:00', 'id': 5},
    'night_wind': {'start': '21:00', 'end': '23:59', 'id': 6},
}

DROP_COLUMNS = [
    'created_at', 'updated_at', 'is_deleted', 'is_real', 'url', 'message', 'story',
    'picture', 'profile_cover', 'local_media', 'data', 'views_organic', 'views_paid',
    'likes', 'wow', 'sad', 'haha', 'angry', 'love', 'thankful', 'none',
    'comments_ad', 'views_ad', 'reach_ad', 'likes_ad', 'saved_ad', 'shares_ad',
    'reactions_new', 'score_new', 'tag', 'type', 'id',
]

REACTION_COLUMNS = ['likes', 'wow', 'sad', 'haha', 'angry', 'love', 'thankful']
METRIC_COLUMNS = ['reactions', 'comments', 'shares']

MISSING_THRESHOLD_PCT = 90


def assign_time_slot(timestamp: pd.Timestamp) -> int:
    """Retourne l'id du créneau horaire (0-6) correspondant à un timestamp."""
    if isinstance(timestamp, str):
        timestamp = pd.to_datetime(timestamp)
    ts_time = timestamp.time()
    for slot_info in TIME_SLOTS.values():
        start_time = datetime.strptime(slot_info['start'], '%H:%M').time()
        end_time = datetime.strptime(slot_info['end'], '%H:%M').time()
        if slot_info['id'] < 6:
            if start_time <= ts_time < end_time:
                return slot_info['id']
        else:
            if start_time <= ts_time <= end_time:
                return slot_info['id']
    return -1


def load_and_prepare_data(files_dict: dict) -> pd.DataFrame:
    """
    Charge, fusionne et nettoie les données Facebook de plusieurs entreprises.

    Args:
        files_dict: dict {nom_entreprise: chemin_fichier_csv}

    Returns:
        DataFrame nettoyé et enrichi de features temporelles/contenu.
    """
    dfs = []
    for name, path in files_dict.items():
        tmp = pd.read_csv(path, encoding='latin-1', on_bad_lines='skip', engine='python')
        tmp['entreprise'] = name
        dfs.append(tmp)
    df = pd.concat(dfs, ignore_index=True)

    # Réactions
    for col in REACTION_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['reactions'] = df[REACTION_COLUMNS].sum(axis=1, min_count=1)

    # Features texte (avant suppression du message)
    df['message_length'] = df['message'].fillna('').apply(len)
    df['has_hashtag'] = df['message'].fillna('').str.contains('#').astype(int)
    df['has_link'] = df['message'].fillna('').str.contains(r'http|www\.').astype(int)
    df['has_emoji'] = df['message'].fillna('').str.contains(
        r'[\U0001F300-\U0001FAFF\U00002600-\U000027BF]', regex=True
    ).astype(int)

    # Suppression colonnes inutiles (reach_organic/reach_paid conservées volontairement)
    df.drop(columns=DROP_COLUMNS, inplace=True, errors='ignore')
    df.dropna(axis=1, how='all', inplace=True)

    missing_pct = df.isnull().mean() * 100
    cols_to_drop = missing_pct[missing_pct > MISSING_THRESHOLD_PCT].index.tolist()
    df.drop(columns=cols_to_drop, inplace=True)
    df.drop(columns=['social_id'], inplace=True, errors='ignore')

    # Timestamp
    df = df.rename(columns={'creation_time': 'timestamp'})
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%m/%Y %H:%M', errors='coerce')
    df = df.dropna(subset=['timestamp']).reset_index(drop=True)

    # Renommage
    df = df.rename(columns={'status_type': 'content_categorie', 'social_type': 'platform'})
    df.drop(columns=['page_id'], inplace=True, errors='ignore')

    # Features temporelles
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_name'] = df['timestamp'].dt.day_name()
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year
    df['is_weekend'] = df['day_of_week'].isin([5, 6])
    df['is_business_hours'] = (df['hour'] > 8) & (df['hour'] < 18)
    df['time_slot_id'] = df['timestamp'].apply(assign_time_slot)

    # Métriques finales (NaN = 0, absence réelle plutôt qu'inconnue)
    for col in METRIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    if 'post_clicks_by_type' in df.columns:
        df['post_clicks_by_type'] = df['post_clicks_by_type'].ffill()

    df['total_engagement'] = df['reactions'] + df['comments'] + df['shares']

    return df


if __name__ == '__main__':
    import os

    files = {
        'Ooredoo': '../data/raw/posts_dataOOFB0.csv',
        'Orange': '../data/raw/posts_dataOrangeFB0.csv',
        'TT': '../data/raw/posts_dataTTFB0.csv',
    }
    result = load_and_prepare_data(files)
    os.makedirs('../data/prepared', exist_ok=True)
    result.to_csv('../data/prepared/resultPreparation.csv', index=False)
    print(f"✓ Export terminé : {result.shape}")