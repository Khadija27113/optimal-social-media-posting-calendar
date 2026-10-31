"""
Génération d'un calendrier de publication hebdomadaire optimisé.

Pour une entreprise donnée, ce module :
1. Détermine un rythme de publication hebdomadaire réaliste (basé sur son
   historique récent, sur une période d'observation configurable).
2. Répartit ce volume entre les types de contenu selon leurs proportions
   historiques (pour varier le contenu plutôt que répéter un seul type).
3. Utilise le modèle entraîné pour classer tous les créneaux (jour x heure)
   possibles par score d'engagement relatif prédit.
4. Ne retient que les créneaux suffisamment appuyés par l'historique réel
   (garde-fou anti-extrapolation sur des combinaisons rares).
5. Construit un calendrier final sans doublon exact de créneau.
"""

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from data_pipeline import assign_time_slot, TIME_SLOTS

DAY_NAMES_FR = {0: 'Lundi', 1: 'Mardi', 2: 'Mercredi', 3: 'Jeudi',
                4: 'Vendredi', 5: 'Samedi', 6: 'Dimanche'}

MIN_SLOT_SUPPORT = 5   # nb minimum de posts historiques pour faire confiance à un créneau
MIN_CONTENT_SUPPORT = 30  # nb minimum de posts historiques pour considérer un type de contenu


def _load_artifacts(models_dir: str):
    models_dir = Path(models_dir)
    with open(models_dir / 'model_metadata.json') as f:
        metadata = json.load(f)
    training_data = pd.read_csv(models_dir / 'model_training_data.csv')
    training_data['timestamp'] = pd.to_datetime(training_data['timestamp'])
    return metadata, training_data


def _weekly_post_frequency(training_data: pd.DataFrame, entreprise: str,
                            observation_weeks: int = 78) -> float:
    """Fréquence hebdomadaire moyenne récente (defaut ~18 mois d'historique)."""
    sub = training_data[training_data['entreprise'] == entreprise].copy()
    cutoff = sub['timestamp'].max() - pd.Timedelta(weeks=observation_weeks)
    recent = sub[sub['timestamp'] >= cutoff]
    if len(recent) < 10:  # pas assez de données récentes, on prend tout l'historique
        recent = sub
    n_weeks = max((recent['timestamp'].max() - recent['timestamp'].min()).days / 7, 1)
    return len(recent) / n_weeks, recent


def _content_type_allocation(recent_df: pd.DataFrame, weekly_total: int,
                              observation_weeks: int) -> dict:
    """Répartit le volume hebdomadaire entre types de contenu, au prorata de
    leur poids historique récent (méthode des plus grands restes pour arrondir
    proprement à un total entier)."""
    counts = recent_df['content_categorie'].value_counts()
    # seuil de fiabilité proportionnel à la VRAIE fenêtre d'observation choisie
    # (avant: valeur fixe qui ignorait le paramètre observation_weeks -> bug)
    min_content_threshold = max(2, MIN_CONTENT_SUPPORT * observation_weeks / 78)
    valid = counts[counts >= min_content_threshold]
    if valid.empty:
        valid = counts  # fallback si rien ne passe le seuil
    proportions = valid / valid.sum()

    raw_alloc = proportions * weekly_total
    alloc = raw_alloc.astype(int)
    remainder = weekly_total - alloc.sum()
    # distribue le reste aux types avec les plus gros restes fractionnaires
    fractional = (raw_alloc - alloc).sort_values(ascending=False)
    for content_type in fractional.index[:remainder]:
        alloc[content_type] += 1
    return {k: int(v) for k, v in alloc.items() if v > 0}


def get_baseline_reference(entreprise: str, models_dir: str = '../models',
                            observation_weeks: int = 78) -> float:
    """
    Renvoie l'engagement médian de référence (baseline brute, pas le score
    relatif) sur la période d'observation choisie -> permet de convertir un
    score relatif en estimation concrète (ex: score 5 x baseline 90 = ~450).
    """
    metadata, training_data = _load_artifacts(models_dir)
    _, recent_df = _weekly_post_frequency(training_data, entreprise, observation_weeks)
    return float(recent_df['total_engagement'].median())



def generate_weekly_calendar(entreprise: str, models_dir: str = '../models',
                              observation_weeks: int = 78) -> pd.DataFrame:
    """
    Génère le calendrier de publication hebdomadaire recommandé pour une entreprise.

    Args:
        entreprise: nom de l'entreprise (doit correspondre à un modèle sauvegardé)
        models_dir: dossier contenant les modèles et métadonnées
        observation_weeks: taille de la fenêtre d'observation récente (en semaines)
                            utilisée pour calculer le rythme de publication et
                            les proportions de contenu. ~78 semaines = 18 mois.

    Returns:
        DataFrame du calendrier recommandé, trié par jour de la semaine.
    """
    metadata, training_data = _load_artifacts(models_dir)
    model = joblib.load(Path(models_dir) / f'engagement_model_{entreprise}.pkl')
    defaults = metadata['feature_defaults']
    features = metadata['features_per_entreprise']

    weekly_freq, recent_df = _weekly_post_frequency(training_data, entreprise, observation_weeks)
    weekly_total = max(round(weekly_freq), 1)

    if len(recent_df) < 20:
        print(f"  ⚠ {entreprise}: seulement {len(recent_df)} posts dans la période "
              f"d'observation choisie ({observation_weeks} semaines) — résultats peu fiables. "
              f"Une fenêtre plus large est recommandée.")

    content_alloc = _content_type_allocation(recent_df, weekly_total, observation_weeks)

    ent_df = training_data[training_data['entreprise'] == entreprise]

    # Fiabilité calculée sur la PÉRIODE D'OBSERVATION choisie (recent_df),
    # pas sur tout l'historique -> le calendrier reflète le comportement
    # récent/actuel de la page, pas des habitudes vieilles de 10 ans.
    reliability_df = recent_df

    # Grille de toutes les combinaisons jour x heure x contenu, moyennée sur les mois
    rows = []
    for day in range(7):
        for hour in range(24):
            slot_id = assign_time_slot(pd.Timestamp(year=2024, month=1, day=1, hour=hour))
            is_weekend = day in [5, 6]
            is_business_hours = 8 < hour < 18
            for content in content_alloc:
                for month in range(1, 13):
                    rows.append({
                        'day_of_week': day, 'hour': hour, 'content_categorie': content,
                        'month': month, 'message_length': defaults['message_length'],
                        'has_hashtag': defaults['has_hashtag'], 'has_link': defaults['has_link'],
                        'has_emoji': defaults['has_emoji'], 'is_weekend': is_weekend,
                        'is_business_hours': is_business_hours, 'time_slot_id': slot_id,
                    })
    grid = pd.DataFrame(rows)
    grid['pred_relative'] = np.clip(np.expm1(model.predict(grid[features])), 0, None)
    grid['time_slot_id'] = grid['hour'].apply(
        lambda h: assign_time_slot(pd.Timestamp(year=2024, month=1, day=1, hour=h))
    )
    # Agrégation au niveau CRÉNEAU (pas l'heure exacte) : la fiabilité historique
    # est mesurée par créneau, donc la recommandation doit l'être aussi, sinon
    # plusieurs heures voisines se partagent artificiellement le même petit
    # groupe de posts historiques et semblent chacune "bien appuyées" à tort.
    grid_by_slot = grid.groupby(
        ['day_of_week', 'time_slot_id', 'content_categorie']
    )['pred_relative'].mean().reset_index()
    # heure représentative du créneau (la plus proche du centre, pour l'affichage)
    slot_center_hour = {info['id']: (
        (datetime.strptime(info['start'], '%H:%M').hour +
         (datetime.strptime(info['end'], '%H:%M').hour if info['end'] != '23:59' else 24)) // 2
    ) % 24 for info in TIME_SLOTS.values()}
    grid_by_slot['hour'] = grid_by_slot['time_slot_id'].map(slot_center_hour)
    grid = grid_by_slot

    # Fiabilité : nb de posts historiques réels dans (jour x créneau x contenu),
    # mesurée sur la période d'observation choisie
    support = reliability_df.groupby(
        ['content_categorie', 'day_of_week', 'time_slot_id']
    ).size().reset_index(name='n_posts_slot')
    grid = grid.merge(support, on=['content_categorie', 'day_of_week', 'time_slot_id'], how='left')
    grid['n_posts_slot'] = grid['n_posts_slot'].fillna(0)

    # Seuil de fiabilité proportionnel à la taille de la fenêtre d'observation :
    # avec une période courte, exiger 5 posts fixes serait presque toujours
    # impossible -> on adapte (plancher de 2, pour éviter le pur hasard).
    adaptive_min_support = max(2, round(MIN_SLOT_SUPPORT * observation_weeks / 78))
    grid_reliable = grid[grid['n_posts_slot'] >= adaptive_min_support].copy()

    # Construction du calendrier final : pour chaque type de contenu, on prend
    # ses N meilleurs créneaux (N = allocation), sans réutiliser un créneau
    # (jour+créneau) déjà pris par un autre type, pour étaler la semaine.
    used_slots = set()
    calendar_rows = []
    for content, n_needed in sorted(content_alloc.items(), key=lambda x: -x[1]):
        candidates = grid_reliable[grid_reliable['content_categorie'] == content].sort_values(
            'pred_relative', ascending=False
        )
        picked = 0
        for _, r in candidates.iterrows():
            slot_key = (r['day_of_week'], r['time_slot_id'])
            if slot_key in used_slots:
                continue
            used_slots.add(slot_key)
            calendar_rows.append(r)
            picked += 1
            if picked >= n_needed:
                break
        if picked < n_needed:
            print(f"  ⚠ {entreprise}/{content}: seulement {picked}/{n_needed} créneaux "
                  f"fiables disponibles (pas assez d'historique pour recommander plus)")

    calendar = pd.DataFrame(calendar_rows)
    if calendar.empty:
        return calendar
    calendar['jour'] = calendar['day_of_week'].map(DAY_NAMES_FR)
    calendar = calendar.sort_values(['day_of_week', 'hour']).reset_index(drop=True)
    calendar['entreprise'] = entreprise

    cols = ['entreprise', 'jour', 'hour', 'content_categorie', 'pred_relative', 'n_posts_slot']
    return calendar[cols].rename(columns={
        'hour': 'heure', 'content_categorie': 'type_contenu',
        'pred_relative': 'score_relatif_predit', 'n_posts_slot': 'fiabilite_n_posts'
    })


if __name__ == '__main__':
    metadata, training_data = _load_artifacts('../models')
    entreprises = sorted(training_data['entreprise'].unique())

    # Démonstration : deux périodes d'observation différentes sur la même entreprise,
    # pour vérifier que le paramètre change vraiment le résultat.
    print(f"\n{'#'*70}\n# DÉMONSTRATION : impact de la période d'observation (Orange)\n{'#'*70}")
    for weeks, label in [(12, "3 mois"), (78, "18 mois (défaut)")]:
        print(f"\n--- Fenêtre: {weeks} semaines ({label}) ---")
        cal = generate_weekly_calendar('Orange', models_dir='../models', observation_weeks=weeks)
        print(f"  Nb de créneaux recommandés: {len(cal)}")
        if not cal.empty:
            print(cal[['jour', 'heure', 'type_contenu', 'fiabilite_n_posts']].to_string(index=False))

    print(f"\n{'#'*70}\n# CALENDRIERS FINAUX (fenêtre par défaut: 78 semaines / 18 mois)\n{'#'*70}")
    for ent in entreprises:
        print(f"\n{'='*70}\n🏢 CALENDRIER RECOMMANDÉ — {ent}\n{'='*70}")
        cal = generate_weekly_calendar(ent, models_dir='../models')
        if cal.empty:
            print("  Aucun créneau suffisamment fiable trouvé.")
            continue
        print(cal.to_string(index=False))
        cal.to_csv(f'../outputs/calendrier_{ent}.csv', index=False)

    print("\n✓ Calendriers générés et exportés dans ../outputs/")