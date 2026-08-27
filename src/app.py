"""
Interface client Streamlit — Calendrier de publication optimisé.

Permet au client de choisir une entreprise et une période d'observation,
et affiche le calendrier hebdomadaire recommandé avec l'évolution
d'engagement attendue pour chaque créneau.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from pattern_generator import _load_artifacts, generate_weekly_calendar  # noqa: E402

MODELS_DIR = str(Path(__file__).parent.parent / 'models')

DAY_ORDER = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']

CONTENT_LABELS_FR = {
    'added_photos': 'Photo',
    'added_video': 'Vidéo',
    'storie_photo': 'Story (photo)',
    'storie_video': 'Story (vidéo)',
    'mobile_status_update': 'Statut mobile',
    'shared_story': 'Partage',
}

st.set_page_config(page_title="Calendrier de publication optimisé", page_icon="📅", layout="wide")


@st.cache_data(show_spinner=False)
def get_available_entreprises():
    _, training_data = _load_artifacts(MODELS_DIR)
    return sorted(training_data['entreprise'].unique())


@st.cache_data(show_spinner=False)
def compute_calendar(entreprise: str, observation_weeks: int):
    return generate_weekly_calendar(entreprise, models_dir=MODELS_DIR, observation_weeks=observation_weeks)


def score_to_color(score: float) -> str:
    """Vert plus le score dépasse 1 (baseline), rouge plus il est en dessous."""
    if score >= 1:
        intensity = min((score - 1) / 4, 1)  # sature autour de score=5
        r = int(220 - intensity * 140)
        g = int(230)
        b = int(220 - intensity * 140)
    else:
        intensity = min((1 - score) / 1, 1)
        r = int(230)
        g = int(220 - intensity * 140)
        b = int(220 - intensity * 140)
    return f"rgb({r},{g},{b})"


def reliability_label(n_posts: float) -> str:
    if n_posts >= 20:
        return "🟢 Fiable"
    elif n_posts >= 8:
        return "🟡 Modérée"
    else:
        return "🟠 Limitée"


# ----------------------------------------------------------------
# Barre latérale : contrôles client
# ----------------------------------------------------------------
st.sidebar.title("Paramètres")

entreprises = get_available_entreprises()
entreprise = st.sidebar.selectbox("Entreprise", entreprises)

observation_weeks = st.sidebar.slider(
    "Période d'observation (en semaines)",
    min_value=4, max_value=104, value=78, step=2,
    help="Fenêtre récente utilisée pour calculer le rythme de publication et "
         "la fiabilité des recommandations. Une fenêtre courte est plus réactive "
         "aux tendances actuelles mais repose sur moins de données ; une fenêtre "
         "longue est plus robuste mais moins réactive."
)
st.sidebar.caption(f"≈ {observation_weeks / 4.33:.1f} mois")

generate = st.sidebar.button(" Générer le calendrier", type="primary", width='stretch')

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Comment lire le score ?**\n\n"
    "Un score de `5.0` signifie que ce créneau devrait générer environ "
    "**5× l'engagement habituel** de la page. Un score de `0.5` signifie "
    "2× moins que d'habitude."
)

# ----------------------------------------------------------------
# Corps principal
# ----------------------------------------------------------------
st.title(" Calendrier de publication optimisé")
st.markdown(
    f"Recommandations pour **{entreprise}**, basées sur les **{observation_weeks} dernières semaines** "
    f"d'activité (~{observation_weeks / 4.33:.1f} mois)."
)

with st.spinner("Génération du calendrier..."):
    calendar = compute_calendar(entreprise, observation_weeks)

if calendar.empty:
    st.error(
        "Aucun créneau suffisamment fiable n'a pu être identifié sur cette période "
        "d'observation. Essayez d'élargir la fenêtre (plus de semaines)."
    )
    st.stop()

# ----------------------------------------------------------------
# Indicateurs clés
# ----------------------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Posts recommandés / semaine", len(calendar))
col2.metric("Score moyen prédit", f"{calendar['score_relatif_predit'].mean():.2f}×")
best_row = calendar.loc[calendar['score_relatif_predit'].idxmax()]
col3.metric(
    "Meilleur créneau",
    f"{best_row['jour']} {int(best_row['heure'])}h",
    f"×{best_row['score_relatif_predit']:.2f}"
)

st.markdown("---")

# ----------------------------------------------------------------
# Vue calendrier hebdomadaire (7 colonnes, une par jour)
# ----------------------------------------------------------------
st.subheader("Vue hebdomadaire")

cols = st.columns(7)
for i, day in enumerate(DAY_ORDER):
    with cols[i]:
        st.markdown(f"**{day}**")
        day_posts = calendar[calendar['jour'] == day].sort_values('heure')
        if day_posts.empty:
            st.caption("—")
            continue
        for _, row in day_posts.iterrows():
            content_label = CONTENT_LABELS_FR.get(row['type_contenu'], row['type_contenu'])
            score = row['score_relatif_predit']
            color = score_to_color(score)
            reliab = reliability_label(row['fiabilite_n_posts'])
            st.markdown(
                f"""
                <div style="background-color:{color}; border-radius:8px; padding:8px;
                            margin-bottom:8px; border:1px solid rgba(0,0,0,0.08);">
                    <div style="font-weight:600; font-size:0.95em;">{int(row['heure'])}h — {content_label}</div>
                    <div style="font-size:1.3em; font-weight:700; margin-top:2px;">×{score:.2f}</div>
                    <div style="font-size:0.75em; color:#555; margin-top:2px;">
                        {reliab} ({int(row['fiabilite_n_posts'])} posts hist.)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.markdown("---")

# ----------------------------------------------------------------
# Explication de l'évolution d'engagement attendue
# ----------------------------------------------------------------
st.subheader("Évolution d'engagement attendue")
st.markdown(
    "Chaque score représente un **multiplicateur par rapport à l'engagement habituel** "
    "de la page (calculé sur sa propre médiane récente). Par exemple :"
)
example = calendar.sort_values('score_relatif_predit', ascending=False).iloc[0]
st.info(
    f"Le créneau **{example['jour']} à {int(example['heure'])}h** ({CONTENT_LABELS_FR.get(example['type_contenu'], example['type_contenu'])}) "
    f"a un score prédit de **×{example['score_relatif_predit']:.2f}** : si l'engagement habituel de "
    f"{entreprise} est par exemple de 100 (likes + commentaires + partages), publier à ce moment-là "
    f"devrait générer environ **{int(example['score_relatif_predit'] * 100)}** — soit "
    f"{example['score_relatif_predit']:.1f}× plus qu'un post moyen."
)

# ----------------------------------------------------------------
# Tableau détaillé + export
# ----------------------------------------------------------------
with st.expander(" Voir le tableau détaillé"):
    display_df = calendar.copy()
    display_df['type_contenu'] = display_df['type_contenu'].map(
        lambda x: CONTENT_LABELS_FR.get(x, x)
    )
    display_df['fiabilité'] = display_df['fiabilite_n_posts'].apply(reliability_label)
    display_df = display_df.rename(columns={
        'jour': 'Jour', 'heure': 'Heure', 'type_contenu': 'Type de contenu',
        'score_relatif_predit': 'Score prédit', 'fiabilite_n_posts': 'Nb posts historiques',
    })[['Jour', 'Heure', 'Type de contenu', 'Score prédit', 'Nb posts historiques', 'fiabilité']]
    st.dataframe(display_df, width='stretch', hide_index=True)

    csv = calendar.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Télécharger le calendrier (CSV)", data=csv,
        file_name=f"calendrier_{entreprise}_{observation_weeks}sem.csv", mime="text/csv",
    )

st.caption(
    "⚠️ Les scores sont des prédictions basées sur l'historique de publication. "
    "Un score élevé avec une fiabilité 'Limitée' repose sur peu de données passées "
    "et doit être interprété avec prudence."
)

# ----------------------------------------------------------------
# Heatmap : évolution du score sur toute la semaine (vue d'ensemble)
# ----------------------------------------------------------------
st.markdown("---")
st.subheader(" Évolution de l'engagement prédit sur la semaine")
st.markdown(
    "Cette carte de chaleur montre le score moyen prédit pour **tous les "
    "créneaux possibles** (pas seulement ceux retenus dans le calendrier "
    "ci-dessus), tous types de contenu confondus — utile pour repérer les "
    "grandes tendances de la semaine."
)

import json

import joblib
import plotly.graph_objects as go

from data_pipeline import TIME_SLOTS, assign_time_slot

SLOT_LABELS_FR = {
    0: '00h-06h (nuit)', 1: '06h-09h (matin tôt)', 2: '09h-12h (matinée)',
    3: '12h-15h (déjeuner)', 4: '15h-18h (après-midi)', 5: '18h-21h (soirée)',
    6: '21h-00h (fin soirée)',
}


@st.cache_data(show_spinner=False)
def compute_full_week_heatmap(entreprise: str):
    with open(str(Path(MODELS_DIR) / 'model_metadata.json')) as f:
        metadata = json.load(f)
    _, training_data = _load_artifacts(MODELS_DIR)
    model = joblib.load(Path(MODELS_DIR) / f'engagement_model_{entreprise}.pkl')
    defaults = metadata['feature_defaults']
    features = metadata['features_per_entreprise']
    content_types = training_data[training_data.entreprise == entreprise]['content_categorie'].unique()

    rows = []
    for day in range(7):
        for slot_info in TIME_SLOTS.values():
            hour = int(slot_info['start'].split(':')[0])
            for content in content_types:
                rows.append({
                    'day_of_week': day, 'hour': hour, 'content_categorie': content,
                    'month': 6, 'message_length': defaults['message_length'],
                    'has_hashtag': defaults['has_hashtag'], 'has_link': defaults['has_link'],
                    'has_emoji': defaults['has_emoji'], 'is_weekend': day in [5, 6],
                    'is_business_hours': 8 < hour < 18, 'time_slot_id': slot_info['id'],
                })
    grid = pd.DataFrame(rows)
    grid['pred'] = np.clip(np.expm1(model.predict(grid[features])), 0, None)
    return grid.groupby(['day_of_week', 'time_slot_id'])['pred'].mean().reset_index()


with st.spinner("Calcul de la carte de chaleur..."):
    heatmap_data = compute_full_week_heatmap(entreprise)

slot_ids_sorted = sorted(TIME_SLOTS.values(), key=lambda x: x['id'])
z = []
for slot in slot_ids_sorted:
    row_vals = []
    for day in range(7):
        v = heatmap_data[(heatmap_data.day_of_week == day) &
                          (heatmap_data.time_slot_id == slot['id'])]['pred']
        row_vals.append(round(float(v.values[0]), 2) if len(v) else None)
    z.append(row_vals)

fig = go.Figure(data=go.Heatmap(
    z=z,
    x=DAY_ORDER,
    y=[SLOT_LABELS_FR[s['id']] for s in slot_ids_sorted],
    colorscale='RdYlGn',
    zmid=1,
    text=z,
    texttemplate="%{text}×",
    colorbar=dict(title="Score"),
    hovertemplate="%{x}, %{y}<br>Score: %{z:.2f}×<extra></extra>",
))
fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "🟢 Vert = engagement au-dessus de la normale · 🔴 Rouge = en dessous · "
    "Score = 1 → performance habituelle de la page."
)