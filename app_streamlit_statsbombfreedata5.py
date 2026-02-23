import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import Pitch, VerticalPitch, FontManager
import numpy as np
from statsbombpy import sb
import warnings
import sys
import subprocess
from datetime import datetime
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
import seaborn as sns
import math
from matplotlib import patches
import matplotlib.patheffects as path_effects

# Supprimer les warnings d'authentification
warnings.filterwarnings("ignore", message="credentials were not supplied")

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Complete StatsBomb Football Analysis by Oussama Boutrouft (@Oussama_Boutrouft)",
    page_icon="⚽",
    layout="wide"
)

# Title of the application
st.title("⚽ Complete StatsBomb Football Analysis Platform by Oussama Boutrouft (@Oussama_Boutrouft)")
st.markdown("""
## Professional Football Analysis Dashboard
This application integrates ALL analyses from your notebook with advanced features.
""")

# Function to install dependencies
def install_dependencies():
    dependencies = ['statsbombpy', 'mplsoccer', 'scipy', 'seaborn']
    
    for package in dependencies:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            st.info(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            st.success(f"{package} installed successfully!")

# Install dependencies
install_dependencies()

# Initialize session state
if 'competitions_df' not in st.session_state:
    st.session_state.competitions_df = None
if 'matches_df' not in st.session_state:
    st.session_state.matches_df = None
if 'events_df' not in st.session_state:
    st.session_state.events_df = None
if 'selected_competition_id' not in st.session_state:
    st.session_state.selected_competition_id = None
if 'selected_season_id' not in st.session_state:
    st.session_state.selected_season_id = None
if 'selected_match_id' not in st.session_state:
    st.session_state.selected_match_id = None
if 'selected_match_info' not in st.session_state:
    st.session_state.selected_match_info = None
if 'all_analysis_data' not in st.session_state:
    st.session_state.all_analysis_data = {}

# ============================================
# NEW ANALYSIS FUNCTIONS TO ADD
# ============================================

# 14. Positional Heatmap (Juego de Posición)
def plot_positional_heatmap(events_df, team_name, ax, cmap_color='#e32221'):
    """Calcule et affiche la heatmap de Juego de Posición pour une équipe."""
    # 1. Configuration du terrain et colormap
    pitch = Pitch(pitch_type='statsbomb', line_zorder=3,
                  pitch_color='#1a1a1a', line_color='#606060')
    custom_cmap = LinearSegmentedColormap.from_list("team_cmap", ['#1a1a1a', cmap_color], N=10)

    # 2. Filtrage des données
    team_events = events_df[events_df['team'] == team_name].copy()
    team_events = team_events[team_events['location'].notna()].copy()
    team_events['x'] = team_events['location'].apply(lambda loc: loc[0])
    team_events['y'] = team_events['location'].apply(lambda loc: loc[1])

    if team_events.empty:
        ax.text(0.5, 0.5, f"No data for {team_name}", color='white', ha='center')
        return

    # 3. Calcul statistique (Juego de Posición - 'full' pour les 30 zones)
    bin_stat = pitch.bin_statistic_positional(team_events.x, team_events.y,
                                              statistic='count', positional='full',
                                              normalize=True)

    # 4. Dessin
    pitch.draw(ax=ax)
    pitch.heatmap_positional(bin_stat, ax=ax, cmap=custom_cmap, edgecolors='#1a1a1a')

    # 5. Annotation des pourcentages
    pitch.label_heatmap(bin_stat, color='#f4edf0', fontsize=14,
                        ax=ax, ha='center', va='center',
                        str_format='{:.0%}')

    ax.set_title(f"{team_name}", color=cmap_color, fontsize=20, fontweight='bold', pad=10)

def run_full_positional_analysis(events_df, team_a, team_b, color_a='#e32221', color_b='#009444'):
    """Fonction 'Master' pour générer le rapport comparatif complet."""
    # Création de la figure
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(15, 16))
    fig.set_facecolor('#1a1a1a')

    # Exécution de l'analyse pour chaque équipe
    plot_positional_heatmap(events_df, team_a, axs[0], cmap_color=color_a)
    plot_positional_heatmap(events_df, team_b, axs[1], cmap_color=color_b)

    plt.suptitle(f'Tactical Analysis: Zone Occupation',
                 color='white', fontsize=25, fontweight='bold', y=0.96)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

# 15. Individual Player Positional Grid
def plot_player_positional_grid(events_df, team_name, cmap_color='#e32221'):
    """Génère une grille de heatmaps de Juego de Posición pour chaque joueur de l'équipe."""
    # 1. Récupération et nettoyage
    team_events = events_df[(events_df['team'] == team_name) & (events_df['location'].notna())].copy()
    team_events['x'] = team_events['location'].apply(lambda x: x[0])
    team_events['y'] = team_events['location'].apply(lambda x: x[1])

    # Liste des joueurs par nombre d'actions (top 11 ou plus)
    players = team_events['player'].value_counts().head(12).index.tolist()

    # 2. Configuration de la grille
    n_players = len(players)
    n_cols = 3
    n_rows = int(np.ceil(n_players / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 5))
    fig.set_facecolor('#1a1a1a')
    axes = axes.flatten()

    pitch = Pitch(pitch_type='statsbomb', line_zorder=3,
                  pitch_color='#1a1a1a', line_color='#606060')
    custom_cmap = LinearSegmentedColormap.from_list("player_cmap", ['#1a1a1a', cmap_color], N=10)

    # 3. Boucle de génération par joueur
    for i, player in enumerate(players):
        ax = axes[i]
        player_df = team_events[team_events['player'] == player]

        if len(player_df) > 0:
            # Calcul positionnel (full = 30 zones tactiques)
            bin_stat = pitch.bin_statistic_positional(player_df.x, player_df.y,
                                                      statistic='count', positional='full',
                                                      normalize=True)

            # Dessin du terrain et de la heatmap
            pitch.draw(ax=ax)
            pitch.heatmap_positional(bin_stat, ax=ax, cmap=custom_cmap, edgecolors='#1a1a1a')

            # Labels des % (on affiche seulement si > 5% pour ne pas surcharger)
            pitch.label_heatmap(bin_stat, color='#f4edf0', fontsize=10,
                                ax=ax, ha='center', va='center',
                                str_format='{:.0%}', exclude_zeros=True)

            ax.set_title(f"{player.split()[-1]} ({len(player_df)} acts)",
                         color='white', fontsize=16, fontweight='bold')
        else:
            ax.text(0.5, 0.5, "No data", color='white', ha='center', va='center')

    # Nettoyage des axes vides
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f"Individual Tactical Occupation - {team_name}",
                 color='white', fontsize=28, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

# 16. Expected Threat (xT) Analysis
class ExpectedThreatAnalyser:
    def __init__(self, bins=(16, 12)):
        self.bins = bins
        self.xt_grid = None

    def compute_xt_matrix(self, events_df):
        """Calcule la matrice xT."""
        pitch = Pitch(pitch_type='statsbomb')
        events_df = events_df.copy()
        events_df['is_shot'] = events_df['type'] == 'Shot'
        events_df['is_move'] = events_df['type'].isin(['Pass', 'Carry'])
        events_df['is_goal'] = False
        
        if 'shot_outcome' in events_df.columns:
            events_df.loc[events_df['shot_outcome'] == 'Goal', 'is_goal'] = True
        
        # Extract coordinates
        events_df = events_df[events_df['location'].notna()].copy()
        events_df['x'] = events_df['location'].apply(lambda loc: loc[0])
        events_df['y'] = events_df['location'].apply(lambda loc: loc[1])

        shot_prob = pitch.bin_statistic(events_df.x, events_df.y, values=events_df.is_shot,
                                        statistic='mean', bins=self.bins)['statistic']
        move_prob = pitch.bin_statistic(events_df.x, events_df.y, values=events_df.is_move,
                                        statistic='mean', bins=self.bins)['statistic']

        shots = events_df[events_df.is_shot]
        goal_prob = np.zeros(self.bins)
        if len(shots) > 0:
            goal_prob_stats = pitch.bin_statistic(shots.x, shots.y, values=shots.is_goal,
                                                  statistic='mean', bins=self.bins)
            goal_prob = np.nan_to_num(goal_prob_stats['statistic'])

        xt = np.multiply(shot_prob, goal_prob)
        for _ in range(10):
            xt = np.multiply(shot_prob, goal_prob) + np.multiply(move_prob, xt)

        self.xt_grid = xt
        return xt

def plot_xt_for_teams(events_df, team_a, team_b):
    """Plot xT analysis for both teams."""
    # 1. Analyse xT
    analyser = ExpectedThreatAnalyser(bins=(16, 12))
    analyser.compute_xt_matrix(events_df)

    # 2. Visualisation (2 lignes pour comparer les 2 équipes)
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(15, 18))
    fig.set_facecolor('#1a1a1a')

    # Configuration du terrain
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#1a1a1a', line_color='white',
                  linewidth=1.5, line_zorder=3)

    teams = [team_a, team_b]
    colors = ['Reds', 'Greens']

    for i, team in enumerate(teams):
        ax = axs[i]
        pitch.draw(ax=ax)

        # On récupère les stats de la grille
        # On utilise une statistique bidon pour la structure, puis on injecte le xT
        team_events = events_df[events_df['team'] == team].copy()
        team_events = team_events[team_events['location'].notna()].copy()
        team_events['x'] = team_events['location'].apply(lambda loc: loc[0])
        team_events['y'] = team_events['location'].apply(lambda loc: loc[1])
        
        if len(team_events) > 0:
            bin_stats = pitch.bin_statistic(team_events.x, team_events.y, bins=(16, 12))
            bin_stats['statistic'] = analyser.xt_grid

            # Heatmap avec alpha
            pitch.heatmap(bin_stats, ax=ax, cmap=colors[i], edgecolors='#222222', alpha=0.7, zorder=2)

            # Labels
            pitch.label_heatmap(bin_stats, ax=ax, str_format='{:.2%}', color='white',
                                fontsize=9, va='center', ha='center', zorder=4)

        ax.set_title(f"Expected Threat (xT) - {team}", color='white', fontsize=20, fontweight='bold', pad=10)

    plt.tight_layout()
    return fig

# 17. xT & Action Chains Analysis
def run_combined_xt_chains_analysis(events_df, team_name, bins=(16, 12)):
    """Combined xT and action chains analysis."""
    # 1. Logique xT (Simplifiée et robuste pour l'affichage)
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#1a1a1a', line_color='white', line_zorder=4)

    # Filter events with location
    events_with_loc = events_df[events_df['location'].notna()].copy()
    events_with_loc['x'] = events_with_loc['location'].apply(lambda loc: loc[0])
    events_with_loc['y'] = events_with_loc['location'].apply(lambda loc: loc[1])

    # Calcul des probabilités pour la grille de fond
    events_with_loc['is_shot'] = events_with_loc['type'] == 'Shot'
    events_with_loc['is_goal'] = False
    if 'shot_outcome' in events_with_loc.columns:
        events_with_loc.loc[events_with_loc['shot_outcome'] == 'Goal', 'is_goal'] = True

    shot_prob = pitch.bin_statistic(events_with_loc.x, events_with_loc.y, 
                                   values=events_with_loc.is_shot, statistic='mean', bins=bins)['statistic']
    
    shots = events_with_loc[events_with_loc.is_shot]
    goal_prob = np.zeros(bins)
    if len(shots) > 0:
        goal_prob_stats = pitch.bin_statistic(shots.x, shots.y,
                                            values=shots.is_goal, statistic='mean', bins=bins)
        goal_prob = np.nan_to_num(goal_prob_stats['statistic'])
    
    xt_grid = np.nan_to_num(shot_prob * goal_prob)
    # On normalise un peu pour l'esthétique du contraste
    xt_grid = xt_grid / (xt_grid.max() if xt_grid.max() != 0 else 1)

    # 3. Extraction des chaînes d'actions (5 dernières avant un tir)
    team_events = events_df[(events_df['team'] == team_name) & (events_df['type'] == 'Shot')].copy()
    sequence_list = []
    
    for _, shot in team_events.iterrows():
        if 'possession' in events_df.columns:
            pos_id = shot['possession']
            chain = events_df[(events_df['possession'] == pos_id) & 
                            (events_df.index <= shot.name)].tail(5)
            sequence_list.append(chain)
    
    if sequence_list:
        df_chains = pd.concat(sequence_list)
        
        # Coordonnées de fin pour les flèches
        def get_end_coords(row):
            if row['type'] == 'Pass' and isinstance(row.get('pass_end_location'), list): 
                return row['pass_end_location']
            if row['type'] == 'Carry' and isinstance(row.get('carry_end_location'), list): 
                return row['carry_end_location']
            if 'location' in row and isinstance(row.get('location'), list):
                return row['location']
            return [np.nan, np.nan]
        
        df_chains[['end_x', 'end_y']] = pd.DataFrame(df_chains.apply(get_end_coords, axis=1).tolist(), 
                                                     index=df_chains.index)
        df_chains = df_chains[df_chains['location'].notna()].copy()
        df_chains['x'] = df_chains['location'].apply(lambda loc: loc[0])
        df_chains['y'] = df_chains['location'].apply(lambda loc: loc[1])
    else:
        df_chains = pd.DataFrame()

    # 4. Visualisation
    fig, ax = pitch.draw(figsize=(15, 10))
    fig.set_facecolor('#1a1a1a')

    # A. La Grille xT (Le "Fond de danger")
    if len(events_with_loc) > 0:
        bin_stats = pitch.bin_statistic(events_with_loc.x, events_with_loc.y, bins=bins)
        bin_stats['statistic'] = xt_grid
        pitch.heatmap(bin_stats, ax=ax, cmap='inferno', alpha=0.5, zorder=1, edgecolors='#222222')

    # B. Les Flèches d'actions (Le "Mouvement")
    if not df_chains.empty:
        moves = df_chains[df_chains['type'].isin(['Pass', 'Carry'])]
        if not moves.empty and 'end_x' in moves.columns and 'end_y' in moves.columns:
            pitch.arrows(moves.x, moves.y, moves.end_x, moves.end_y,
                         color='cyan', alpha=0.8, width=2, headwidth=3, zorder=5, ax=ax, 
                         label='Ball movement')

    # C. Les Tirs
    shots_df = df_chains[df_chains['type'] == 'Shot'] if not df_chains.empty else pd.DataFrame()
    if not shots_df.empty:
        pitch.scatter(shots_df.x, shots_df.y, s=400, marker='*', c='yellow', 
                     edgecolors='white', zorder=6, ax=ax, label='Final shot')

    ax.set_title(f"Expected Threat & Action Chains\n{team_name}",
                 color='white', fontsize=22, fontweight='bold', pad=20)
    ax.legend(facecolor='#1a1a1a', edgecolor='white', labelcolor='white', loc='lower left')

    return fig

# 18. Pass Sonar Analysis
def get_sonar_data(events_df, team_name):
    """Récupère et prépare les données StatsBomb sans erreur."""
    mask = (events_df['team'] == team_name) & (events_df['type'] == 'Pass') & (events_df['location'].notna())
    df = events_df[mask].copy()

    if len(df) > 0:
        # Extraction propre des coordonnées
        df['x'] = df['location'].apply(lambda x: x[0])
        df['y'] = df['location'].apply(lambda x: x[1])
        df['end_x'] = df['pass_end_location'].apply(lambda x: x[0])
        df['end_y'] = df['pass_end_location'].apply(lambda x: x[1])

    return df

def plot_enhanced_team_sonar(df, team_name, ax, pitch, color='#e32221'):
    """Dessine le sonar avec une méthode d'extraction des bords universelle."""
    if len(df) == 0:
        ax.text(0.5, 0.5, f"No pass data for {team_name}", color='white', ha='center', va='center')
        return
    
    # 1. Calcul des angles et distances
    angle, distance = pitch.calculate_angle_and_distance(df.x, df.y, df.end_x, df.end_y)

    # 2. Calcul du Sonar
    bs_sonar = pitch.bin_statistic_sonar(df.x, df.y, angle, bins=(6, 4, 8), center=True)

    # 3. Calcul manuel des limites des zones
    x_edges = np.linspace(0, 120, 7) # 6 bins
    y_edges = np.linspace(0, 80, 5)  # 4 bins

    # Centres pour placer le texte
    cx = (x_edges[:-1] + x_edges[1:]) / 2
    cy = (y_edges[:-1] + y_edges[1:]) / 2

    # 4. Dessin du terrain et du sonar
    pitch.draw(ax=ax)
    pitch.sonar_grid(bs_sonar, ax=ax, width=15, fc=color, ec='white', alpha=0.6, zorder=2)

    # 5. Identification des joueurs dominants
    for i in range(len(x_edges)-1):
        for j in range(len(y_edges)-1):
            # Filtrage des passes dans le rectangle actuel
            mask = (df.x >= x_edges[i]) & (df.x < x_edges[i+1]) & \
                   (df.y >= y_edges[j]) & (df.y < y_edges[j+1])
            subset = df[mask]

            if not subset.empty:
                # Top player dans cette zone
                top_player = subset['player'].value_counts().idxmax()
                last_name = top_player.split()[-1] if isinstance(top_player, str) else "N/A"

                # On place le texte au centre calculé manuellement
                ax.text(cx[i], cy[j] + 5, last_name,
                        color='white', fontsize=8, ha='center', va='center',
                        fontweight='bold', alpha=0.8, zorder=5)

    ax.set_title(f"{team_name} | Pass Sonar & Zone Leaders", color='white', fontsize=18, pad=15)

def run_full_sonar_comparison(events_df, team_a, team_b):
    """Génère la comparaison sonar pour deux équipes."""
    fig, axs = plt.subplots(2, 1, figsize=(16, 20))
    fig.set_facecolor('#1a1a1a')

    pitch = Pitch(pitch_type='statsbomb', pitch_color='#1a1a1a', line_color='#606060', linewidth=1.5)

    # Team 1
    df_lev = get_sonar_data(events_df, team_a)
    plot_enhanced_team_sonar(df_lev, team_a, axs[0], pitch, color='#e32221')

    # Team 2
    df_bre = get_sonar_data(events_df, team_b)
    plot_enhanced_team_sonar(df_bre, team_b, axs[1], pitch, color='#009444')

    plt.tight_layout()
    return fig

# 19. Pass Flow Analysis
def get_pass_flow_data(events_df, team_name):
    """Récupère les passes et prépare les coordonnées."""
    # Filtrer les passes réussies (outcome NaN dans StatsBomb signifie réussite)
    mask = (events_df['team'] == team_name) & (events_df['type'] == 'Pass') & (events_df['location'].notna())
    df = events_df[mask].copy()

    if len(df) > 0:
        # Extraction des coordonnées x, y
        df['x'] = df['location'].apply(lambda x: x[0])
        df['y'] = df['location'].apply(lambda x: x[1])
        df['end_x'] = df['pass_end_location'].apply(lambda x: x[0])
        df['end_y'] = df['pass_end_location'].apply(lambda x: x[1])

    return df

def plot_team_pass_flow(events_df, team_name, ax, color_map='Reds', arrow_color='cyan'):
    """Génère le Pass Flow Plot pour une équipe spécifique."""
    df_pass = get_pass_flow_data(events_df, team_name)

    if len(df_pass) == 0:
        ax.text(0.5, 0.5, f"No pass data for {team_name}", color='white', ha='center', va='center')
        return

    # Configuration du Pitch
    pitch = Pitch(pitch_type='statsbomb', line_zorder=3,
                  line_color='#c7d5cc', pitch_color='#1a1a1a')

    # Division du terrain (6x4 zones)
    bins = (6, 4)

    # 1. Dessiner le terrain
    pitch.draw(ax=ax)

    # 2. Dessiner la heatmap (Densité d'origine des passes)
    bs_heatmap = pitch.bin_statistic(df_pass.x, df_pass.y, statistic='count', bins=bins)
    pitch.heatmap(bs_heatmap, ax=ax, cmap=color_map, alpha=0.5, zorder=1)

    # 3. Dessiner le "Flow" (Les flèches de tendance)
    pitch.flow(df_pass.x, df_pass.y, df_pass.end_x, df_pass.end_y,
               color=arrow_color, arrow_type='same',
               arrow_length=5, bins=bins, ax=ax, zorder=4)

    ax.set_title(f'Pass Flow: {team_name}', fontsize=22, color='white', fontweight='bold')

def run_comparison_pass_flow(events_df, team_a, team_b):
    """Compare le flux de passes des deux équipes."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 20))
    fig.set_facecolor('#1a1a1a')

    # Team A (Rouge / Flèches Cyan pour le contraste)
    plot_team_pass_flow(events_df, team_a, ax1, color_map='Reds', arrow_color='#00f2ff')

    # Team B (Vert / Flèches Blanches)
    plot_team_pass_flow(events_df, team_b, ax2, color_map='Greens', arrow_color='white')

    plt.tight_layout()
    return fig

# 20. Vertical Shot Map
def get_shot_data(events_df, team_name):
    """Récupère les tirs et les xG pour une équipe."""
    mask = (events_df['team'] == team_name) & (events_df['type'] == 'Shot')
    df_shots = events_df[mask].copy()

    if len(df_shots) > 0:
        # Extraction des coordonnées
        df_shots = df_shots[df_shots['location'].notna()].copy()
        locations = df_shots['location'].apply(lambda x: x if isinstance(x, list) else [np.nan, np.nan])
        df_shots[['x', 'y']] = pd.DataFrame(locations.tolist(), index=df_shots.index)

        # Séparation Buts / Non-Buts
        df_goals = df_shots[df_shots['shot_outcome'] == 'Goal'].copy()
        df_no_goals = df_shots[df_shots['shot_outcome'] != 'Goal'].copy()
    else:
        df_goals = pd.DataFrame()
        df_no_goals = pd.DataFrame()

    return df_goals, df_no_goals

def plot_vertical_half_shot_map(events_df, team_name, ax, pitch, main_color='#e32221'):
    """Dessine la carte des tirs sur un demi-terrain vertical."""
    df_goals, df_no_goals = get_shot_data(events_df, team_name)

    if len(df_no_goals) == 0 and len(df_goals) == 0:
        ax.text(0.5, 0.5, f"No shot data for {team_name}", color='white', ha='center', va='center')
        return

    # 1. Dessiner les tirs manqués (Cercles hachurés)
    if len(df_no_goals) > 0:
        # Use a default xG value if not available
        if 'shot_statsbomb_xg' not in df_no_goals.columns:
            df_no_goals['shot_statsbomb_xg'] = 0.1
        
        sizes = (df_no_goals['shot_statsbomb_xg'] * 1900) + 100
        pitch.scatter(df_no_goals['x'], df_no_goals['y'],
                      s=sizes,
                      edgecolors=main_color,
                      c='None',
                      hatch='///',
                      marker='o',
                      ax=ax,
                      alpha=0.6,
                      label='Missed shots')

    # 2. Dessiner les buts (Marqueur Football)
    if len(df_goals) > 0:
        if 'shot_statsbomb_xg' not in df_goals.columns:
            df_goals['shot_statsbomb_xg'] = 0.5
        
        sizes = (df_goals['shot_statsbomb_xg'] * 1900) + 100
        pitch.scatter(df_goals['x'], df_goals['y'],
                      s=sizes,
                      edgecolors='white',
                      linewidths=0.6,
                      c='white',
                      marker='o',  # Using circle instead of football for compatibility
                      ax=ax,
                      zorder=5,
                      label='Goals')

    # Statistiques pour le titre
    total_xg = 0
    if len(df_no_goals) > 0 and 'shot_statsbomb_xg' in df_no_goals.columns:
        total_xg += df_no_goals['shot_statsbomb_xg'].sum()
    if len(df_goals) > 0 and 'shot_statsbomb_xg' in df_goals.columns:
        total_xg += df_goals['shot_statsbomb_xg'].sum()
    
    total_xg = round(total_xg, 2)
    
    ax.set_title(f"{team_name}\n{len(df_goals)} Goals | {total_xg} xG",
                 color='white', fontsize=18, pad=10, fontweight='bold')

def run_side_by_side_shot_comparison(events_df, team_a, team_b):
    """Génère la comparaison face-à-face sur demi-terrains verticaux."""
    # On utilise VerticalPitch avec half=True
    pitch = VerticalPitch(pitch_type='statsbomb', half=True, line_zorder=2,
                          line_color='#c7d5cc', pitch_color='#1a1a1a')

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(18, 12))
    fig.set_facecolor('#1a1a1a')

    # Team A (À gauche)
    pitch.draw(ax=axs[0])
    plot_vertical_half_shot_map(events_df, team_a, axs[0], pitch, main_color='#e32221')

    # Team B (À droite)
    pitch.draw(ax=axs[1])
    plot_vertical_half_shot_map(events_df, team_b, axs[1], pitch, main_color='#009444')

    # Titre général
    plt.suptitle(f"Shot Map Comparison: High Performance Analysis",
                 color='white', fontsize=24, fontweight='bold', y=0.98)

    # Légende unique pour la figure
    if len(axs[0].get_legend_handles_labels()[0]) > 0:
        axs[0].legend(loc='lower left', facecolor='#1a1a1a', labelcolor='white', fontsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

# 21. Convex Hull Analysis (Individual Players)
def plot_player_hull(events_df, player_name, team_name, ax, color_theme):
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#0e1111', line_color='#444444')
    pitch.draw(ax=ax)

    # Filtrer les actions du joueur avec des coordonnées
    player_df = events_df[(events_df['player'] == player_name) & (events_df['location'].notna())].copy()
    player_df['x'] = player_df['location'].apply(lambda loc: loc[0])
    player_df['y'] = player_df['location'].apply(lambda loc: loc[1])

    if len(player_df) > 3: # Il faut au moins 3 points pour une enveloppe
        try:
            # Calcul du Convex Hull
            hull = pitch.convexhull(player_df.x, player_df.y)

            # Dessin du polygone (la zone d'influence)
            pitch.polygon(hull, ax=ax, edgecolor=color_theme, facecolor=color_theme, alpha=0.3, label='Action zone')

            # Dessin des points d'impact
            pitch.scatter(player_df.x, player_df.y, ax=ax, edgecolor='white', facecolor=color_theme, s=60, alpha=0.7)

            # Statistique rapide
            ax.text(2, 78, f"Actions: {len(player_df)}", color='white', fontsize=10)
        except:
            ax.text(60, 40, "Insufficient data for hull", color='gray', ha='center')
    else:
        ax.text(60, 40, "Not enough data", color='gray', ha='center')

    ax.set_title(f"CONVEX HULL : {player_name}", color='white', fontsize=18, fontweight='bold', pad=20)

# 22. Team Convex Hulls
def plot_team_hulls(events_df, team_name, color_theme):
    """Plot convex hulls for all players in a team."""
    # Filtrer les données et supprimer les lignes sans nom de joueur ou sans position
    team_df = events_df[(events_df['team'] == team_name) & (events_df['location'].notna()) & 
                       (events_df['player'].notna())].copy()
    team_df['x'] = team_df['location'].apply(lambda loc: loc[0])
    team_df['y'] = team_df['location'].apply(lambda loc: loc[1])

    players = team_df['player'].unique()
    n_players = len(players)

    # Configuration de la grille
    cols = 4
    rows = math.ceil(n_players / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 4.5), facecolor='#0e1111')
    axes = axes.flatten()

    pitch = Pitch(pitch_type='statsbomb', pitch_color='#1a1d21', line_color='#444444')

    for i, player in enumerate(players):
        if i >= len(axes):
            break
            
        ax = axes[i]
        pitch.draw(ax=ax)

        player_data = team_df[team_df['player'] == player]

        # Sécurité : Vérifier que player est bien une chaîne de caractères
        p_name = str(player).split()[-1] if pd.notnull(player) else "Unknown"

        if len(player_data) >= 3:
            try:
                # Calcul et tracé du Convex Hull
                hull = pitch.convexhull(player_data.x, player_data.y)
                pitch.polygon(hull, ax=ax, edgecolor=color_theme, facecolor=color_theme, alpha=0.3)
                # Points d'action
                pitch.scatter(player_data.x, player_data.y, ax=ax, s=8, color='white', alpha=0.4)
            except:
                pass

        ax.set_title(p_name, color='white', fontsize=14, fontweight='bold', pad=10)

    # Nettoyage des graphiques vides
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f"INFLUENCE ZONES (CONVEX HULL) : {team_name.upper()}",
                 color='white', fontsize=26, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig

# 23. Player Thirds Distribution
def plot_player_thirds(events_df, team_name, color_map='coolwarm'):
    """Plot player action distribution by thirds."""
    # Filtrage des joueurs avec positions
    team_df = events_df[(events_df['team'] == team_name) & (events_df['location'].notna()) & 
                       (events_df['player'].notna())].copy()
    team_df['x'] = team_df['location'].apply(lambda loc: loc[0])

    players = team_df['player'].unique()
    n_players = len(players)
    cols = 4
    rows = math.ceil(n_players / cols)

    # Préparation du pitch et de la statistique par tiers
    pitch = Pitch(line_zorder=2, line_color='#444444', pitch_color='#1a1d21')
    fig, axs = pitch.grid(nrows=rows, ncols=cols, figheight=rows*5,
                          grid_height=0.85, title_height=0.05, axis=False, endnote_height=0)
    fig.set_facecolor('#0e1111')

    for i, player in enumerate(players):
        if i >= len(axs['pitch'].flat):
            break
            
        ax = axs['pitch'].flat[i]
        player_data = team_df[team_df['player'] == player]

        # Calcul du pourcentage par tiers (StatsBomb X va de 0 à 120)
        # Tiers 1: 0-40, Tiers 2: 40-80, Tiers 3: 80-120
        def_3rd = len(player_data[player_data.x <= 40])
        mid_3rd = len(player_data[(player_data.x > 40) & (player_data.x <= 80)])
        att_3rd = len(player_data[player_data.x > 80])

        total = len(player_data)
        if total > 0:
            stats = np.array([[def_3rd/total*100, mid_3rd/total*100, att_3rd/total*100]])
        else:
            stats = np.array([[0, 0, 0]])

        # On crée une structure de bin_statistic factice pour le heatmap
        bin_stat = pitch.bin_statistic(player_data.x, [40]*len(player_data), bins=(3, 1))
        bin_stat['statistic'] = stats

        # Affichage
        heatmap = pitch.heatmap(bin_stat, ax=ax, cmap=color_map, vmin=0, vmax=100, alpha=0.7)
        pitch.label_heatmap(bin_stat, color='white', fontsize=20, ax=ax,
                            str_format='{:.0f}%', ha='center', va='center', fontweight='bold')

        p_name = str(player).split()[-1] if pd.notnull(player) else "Unknown"
        ax.text(60, -10, p_name, ha='center', color='white', fontsize=18, fontweight='bold')

    # Supprimer les axes inutilisés
    for j in range(i + 1, len(axs['pitch'].flat)):
        axs['pitch'].flat[j].remove()

    title = f"ACTION DISTRIBUTION BY THIRDS : {team_name.upper()}"
    fig.text(0.5, 0.95, title, color='white', fontsize=30, ha='center', fontweight='bold')
    return fig

# 24. Defensive Block Analysis - CORRECTED
def get_defensive_actions_statsbomb(df):
    """Get defensive actions from StatsBomb data."""
    def_types = ['Ball Recovery', 'Interception', 'Tackle', 'Clearance', 'Error', 'Foul Committed', 'Block']
    df_def = df[df['type'].isin(def_types)].copy()

    # Extraction des coordonnées
    df_def['x'] = df_def['location'].apply(lambda loc: loc[0] if isinstance(loc, list) else np.nan)
    df_def['y'] = df_def['location'].apply(lambda loc: loc[1] if isinstance(loc, list) else np.nan)
    return df_def.dropna(subset=['x', 'y'])

def plot_defensive_block(events_df, team_name, ax, col):
    """Plot defensive block analysis."""
    # Filtrage
    team_df = events_df[events_df['team'] == team_name].copy()

    # Identifier les titulaires (First Eleven) via les tactiques
    starters = team_df[team_df['minute'] == 0]['player'].unique()

    # Actions défensives
    def_actions = get_defensive_actions_statsbomb(team_df)

    if len(def_actions) == 0:
        ax.text(0.5, 0.5, f"No defensive actions for {team_name}", color='white', ha='center', va='center')
        return

    # CORRECTION : Calcul des positions médianes par joueur
    player_stats = def_actions.groupby('player').agg(
        x_median=('x', 'median'),
        y_median=('y', 'median')
    ).reset_index()
    player_stats['count'] = def_actions.groupby('player').size().values
    player_stats['is_starter'] = player_stats['player'].isin(starters)

    pitch = Pitch(pitch_type='statsbomb', pitch_color='#0e1111', line_color='#ffffff', line_zorder=2)
    pitch.draw(ax=ax)

    # Heatmap (KDE)
    flamingo_cmap = LinearSegmentedColormap.from_list("custom", ['#0e1111', col], N=100)
    
    try:
        pitch.kdeplot(def_actions.x, def_actions.y, ax=ax, fill=True, levels=100, 
                     thresh=0.2, cmap=flamingo_cmap, alpha=0.6)
    except:
        pass

    # Scatter des actions individuelles
    ax.scatter(def_actions.x, def_actions.y, s=10, marker='x', color='yellow', alpha=0.1, zorder=3)

    # Nodes des joueurs
    if len(player_stats) > 0:
        MAX_SIZE = 2000
        if player_stats['count'].max() > 0:
            player_stats['marker_size'] = (player_stats['count'] / player_stats['count'].max()) * MAX_SIZE
        else:
            player_stats['marker_size'] = 100

        for _, row in player_stats.iterrows():
            # Exclure le gardien si identifié (ici par position moyenne très basse)
            if row.x_median < 15: 
                continue

            marker = 'o' if row.is_starter else 's'
            pitch.scatter(row.x_median, row.y_median, s=row.marker_size + 200, marker=marker,
                          color='#0e1111', edgecolor=col, linewidth=2, zorder=4, ax=ax)

            # Nom abrégé
            short_name = str(row['player']).split()[-1][:3].upper()
            ax.text(row.x_median, row.y_median, short_name, color='white', fontsize=8, 
                   ha='center', va='center', zorder=5)

        # Ligne de hauteur (Defensive Actions Height)
        dah = player_stats.x_median.mean()
        ax.axvline(x=dah, color=col, linestyle='--', alpha=0.8, linewidth=2)
        # CORRECTION DES PATH_EFFECTS :
        ax.text(dah + 1, 78, f"HEIGHT: {dah:.1f}m", color='white', fontsize=12, 
               fontweight='bold', path_effects=[path_effects.Stroke(linewidth=3, foreground='black'),
                                               path_effects.Normal()])

    ax.set_title(f"DEFENSIVE BLOCK: {team_name}", color='white', fontsize=20, fontweight='bold', pad=20)

# 25. Elegant Defensive Block - CORRECTED
def plot_elegant_defensive_block(events_df, team_name, ax, main_color):
    """Plot elegant defensive block visualization."""
    team_df = events_df[events_df['team'] == team_name].copy()
    
    # Identification des titulaires
    starters = team_df[team_df['minute'] == 0]['player'].unique()
    def_actions = get_defensive_actions_statsbomb(team_df)

    if len(def_actions) == 0:
        ax.text(0.5, 0.5, f"No defensive actions for {team_name}", color='white', ha='center', va='center')
        return

    # CORRECTION : Statistiques par joueur
    player_stats = def_actions.groupby('player').agg(
        x_median=('x', 'median'),
        y_median=('y', 'median')
    ).reset_index()
    player_stats['count'] = def_actions.groupby('player').size().values
    player_stats['is_starter'] = player_stats['player'].isin(starters)

    # Dessin du terrain
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#0e1111', line_color='#ffffff', 
                  line_zorder=3, linewidth=1.5, goal_type='box')
    pitch.draw(ax=ax)

    # HEATMAP (KDE)
    custom_cmap = LinearSegmentedColormap.from_list("club_cmap", ['#0e1111', main_color], N=100)

    try:
        pitch.kdeplot(def_actions.x, def_actions.y, ax=ax, fill=True,
                      levels=100, thresh=0.1, cmap=custom_cmap, alpha=0.8, zorder=2)
    except:
        pass

    # NODES DES JOUEURS
    if len(player_stats) > 0:
        MAX_SIZE = 1800
        if player_stats['count'].max() > 0:
            player_stats['marker_size'] = (player_stats['count'] / player_stats['count'].max()) * MAX_SIZE
        else:
            player_stats['marker_size'] = 100

        for _, row in player_stats.iterrows():
            if row.x_median < 15: 
                continue  # Skip goalkeeper

            marker = 'o' if row.is_starter else 's'
            pitch.scatter(row.x_median, row.y_median, s=row.marker_size + 200, marker=marker,
                          color='#0e1111', edgecolor=main_color, linewidth=2, zorder=4, ax=ax)

            # Nom abrégé
            p_name = str(row['player']).split()[-1][:3].upper() if pd.notnull(row['player']) else "???"
            ax.text(row.x_median, row.y_median, p_name, color='white', fontsize=8,
                    ha='center', va='center', fontweight='bold', zorder=5)

        # LIGNE DE HAUTEUR DU BLOC (DAH)
        dah = player_stats.x_median.mean()
        ax.axvline(x=dah, color='white', linestyle='--', alpha=0.6, linewidth=2, zorder=4)
        # CORRECTION DES PATH_EFFECTS :
        ax.text(dah + 1, 5, f"BLOCK: {dah:.1f}m", color='white', fontsize=12,
                fontweight='bold', path_effects=[path_effects.Stroke(linewidth=3, foreground='black'),
                                                path_effects.Normal()], zorder=5)

    ax.set_title(f"{team_name.upper()}", color='white', fontsize=22, fontweight='bold', pad=25)
    ax.text(60, 85, "DENSITY ANALYSIS", color=main_color, fontsize=9, ha='center', alpha=0.6)

# 26. Danger Passes Analysis
def get_successful_passes(events_df, team_name):
    """Get successful passes for a team."""
    # Filtrage des passes réussies (hors corners)
    passes = events_df[(events_df['team'] == team_name) & 
                      (events_df['type'] == 'Pass') & 
                      (events_df['pass_outcome'].isna())].copy()

    if len(passes) == 0:
        return passes

    # Coordonnées StatsBomb
    passes['x'] = passes['location'].apply(lambda loc: loc[0] if isinstance(loc, list) else np.nan)
    passes['y'] = passes['location'].apply(lambda loc: loc[1] if isinstance(loc, list) else np.nan)
    passes['end_x'] = passes['pass_end_location'].apply(lambda loc: loc[0] if isinstance(loc, list) else np.nan)
    passes['end_y'] = passes['pass_end_location'].apply(lambda loc: loc[1] if isinstance(loc, list) else np.nan)

    # Exclure les corners (si l'info est présente dans pass_type)
    if 'pass_type' in passes.columns:
        passes = passes[passes['pass_type'] != 'Corner']
    
    return passes.dropna(subset=['x', 'y', 'end_x', 'end_y'])

def draw_danger_passes(ax, df, team_name, col):
    """Draw dangerous passes analysis."""
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#0e1111', line_color='#ffffff', linewidth=2)
    pitch.draw(ax=ax)

    if len(df) == 0:
        ax.text(0.5, 0.5, f"No pass data for {team_name}", color='white', ha='center', va='center')
        return

    # Compteurs
    z14 = 0
    hs = 0

    # Définition des zones StatsBomb (Terrain 120x80)
    # Zone 14 : x[100-120], y[30-50]
    # Half Spaces : x[80-120], y[18-30] et y[50-62]

    for _, row in df.iterrows():
        # ZONE 14
        if 100 <= row['end_x'] <= 120 and 30 <= row['end_y'] <= 50:
            arrow = patches.FancyArrowPatch((row['x'], row['y']), (row['end_x'], row['end_y']),
                                            arrowstyle='->', alpha=0.6, mutation_scale=15, 
                                            color='orange', linewidth=1)
            ax.add_patch(arrow)
            z14 += 1
        # HALF SPACES
        elif row['end_x'] >= 80:
            if (18 <= row['end_y'] <= 30) or (50 <= row['end_y'] <= 62):
                arrow = patches.FancyArrowPatch((row['x'], row['y']), (row['end_x'], row['end_y']),
                                                arrowstyle='->', alpha=0.6, mutation_scale=15, 
                                                color=col, linewidth=1)
                ax.add_patch(arrow)
                hs += 1

    # Coloriage des zones sur le pitch
    # Zone 14
    ax.add_patch(plt.Rectangle((100, 30), 18, 20, color='orange', alpha=0.15, label='Zone 14'))
    # Half-Spaces
    ax.add_patch(plt.Rectangle((80, 18), 38, 12, color=col, alpha=0.1, label='Half-Spaces'))
    ax.add_patch(plt.Rectangle((80, 50), 38, 12, color=col, alpha=0.1))

    # Affichage des compteurs (Hexagones)
    ax.scatter(20, 15, color=col, s=8000, marker='h', edgecolor='white', linewidth=2)
    ax.scatter(20, 65, color='orange', s=8000, marker='h', edgecolor='white', linewidth=2)

    # CORRECTION DES PATH_EFFECTS :
    ax.text(20, 15, f"{hs}\nHS", fontsize=18, color='white', ha='center', va='center', 
           fontweight='bold', path_effects=[path_effects.Stroke(linewidth=3, foreground='black'),
                                           path_effects.Normal()])
    ax.text(20, 65, f"{z14}\nZ14", fontsize=18, color='white', ha='center', va='center', 
           fontweight='bold', path_effects=[path_effects.Stroke(linewidth=3, foreground='black'),
                                           path_effects.Normal()])

    # Titre et Direction
    ax.set_title(f"{team_name.upper()}\nDANGEROUS PASSES", color='white', fontsize=22, fontweight='bold', pad=20)
    ax.annotate('', xy=(110, -5), xytext=(10, -5), 
                arrowprops=dict(arrowstyle="->", color=col, lw=2), annotation_clip=False)

# 27. Pass End Zone Distribution
def get_pass_end_data(events_df, team_name):
    """Get pass end location data."""
    # Filtrage passes réussies
    mask = (events_df['team'] == team_name) & (events_df['type'] == 'Pass') & (events_df['pass_outcome'].isna())
    passes = events_df[mask].copy()
    
    if len(passes) == 0:
        return passes
    
    # Extraction des coordonnées d'arrivée
    passes['end_x'] = passes['pass_end_location'].apply(lambda loc: loc[0] if isinstance(loc, list) else np.nan)
    passes['end_y'] = passes['pass_end_location'].apply(lambda loc: loc[1] if isinstance(loc, list) else np.nan)
    
    return passes.dropna(subset=['end_x', 'end_y'])

def plot_pass_end_zone(ax, df, team_name, main_color):
    """Plot pass end zone distribution."""
    if len(df) == 0:
        ax.text(0.5, 0.5, f"No pass data for {team_name}", color='white', ha='center', va='center')
        return
    
    # Création du pitch
    pitch = Pitch(pitch_type='statsbomb', line_color='#ffffff', pitch_color='#0e1111',
                  line_zorder=2, linewidth=2, goal_type='box')
    pitch.draw(ax=ax)

    # Colormap personnalisée
    custom_cmap = LinearSegmentedColormap.from_list("end_zone_cmap", ['#0e1111', main_color], N=50)

    # 1. Calcul des statistiques par zones positionnelles
    bin_statistic = pitch.bin_statistic_positional(df.end_x, df.end_y, statistic='count',
                                                   positional='full', normalize=True)

    # 2. Dessin de la Heatmap
    pitch.heatmap_positional(bin_statistic, ax=ax, cmap=custom_cmap, edgecolors='#222222', alpha=0.9)

    # 3. Ajout des points d'arrivée
    pitch.scatter(df.end_x, df.end_y, c='white', s=2, alpha=0.2, ax=ax)

    # 4. Ajout des labels en pourcentage - CORRECTION DES PATH_EFFECTS :
    pitch.label_heatmap(bin_statistic, color='#ffffff', fontsize=18, ax=ax,
                       ha='center', va='center', str_format='{:.0%}',
                       fontweight='bold', 
                       path_effects=[path_effects.Stroke(linewidth=3, foreground='black'),
                                   path_effects.Normal()])

    # Direction de l'attaque
    ax.annotate('', xy=(110, -7), xytext=(10, -7),
                arrowprops=dict(arrowstyle="->", color=main_color, lw=2),
                annotation_clip=False)

    ax.set_title(f"{team_name.upper()}\nPASS END ZONE DISTRIBUTION",
                 color='white', fontsize=24, fontweight='bold', pad=30)

# 28. Goal Post Analysis
def plot_goal_post_analysis(events_df, team_a, team_b):
    """Goal post analysis showing shot placements."""
    bg_color = '#0e1111'
    line_color = '#ffffff'
    hcol, acol = '#ff002e', '#00ff87'
    
    shots = events_df[events_df['type'] == 'Shot'].copy()
    
    if len(shots) == 0:
        fig, ax = plt.subplots(figsize=(12, 14), facecolor=bg_color)
        ax.text(0.5, 0.5, "No shot data available", color='white', ha='center', va='center')
        return fig
    
    # Extraction des coordonnées du "Goal Mouth"
    def get_mouth_y(loc):
        if isinstance(loc, list) and len(loc) > 1:
            return (44 - loc[1]) * 11.25  # Scale pour passer de 8m à 90 unités
        return 45

    def get_mouth_z(loc):
        if isinstance(loc, list) and len(loc) > 2:
            return loc[2] * 10  # Scale pour la hauteur
        return 0
    
    shots['goalMouthY'] = shots['shot_end_location'].apply(lambda x: get_mouth_y(x) if isinstance(x, list) else 45)
    shots['goalMouthZ'] = shots['shot_end_location'].apply(lambda x: get_mouth_z(x) if isinstance(x, list) else 0)
    
    # Séparation des équipes
    teams = events_df['team'].unique()
    if len(teams) >= 2:
        h_shots = shots[shots['team'] == teams[0]].copy()
        a_shots = shots[shots['team'] == teams[1]].copy()
    else:
        h_shots = shots.copy()
        a_shots = pd.DataFrame()
    
    # Ajustement des positions Z pour le deuxième but
    if len(h_shots) > 0:
        h_shots['goalMouthZ'] = h_shots['goalMouthZ'] + 38

    # Création de la figure
    fig, ax = plt.subplots(figsize=(12, 14), facecolor=bg_color)
    
    # Pitch invisible pour le repère
    pitch = Pitch(pitch_type='statsbomb', pitch_color=bg_color, line_color=bg_color)
    pitch.draw(ax=ax)

    # --- DESSIN DES BUTS (TEMPLATE) ---
    # Away goalpost (Bas)
    ax.plot([0, 0], [-1, 30], color=line_color, linewidth=5)
    ax.plot([0, 90], [30, 30], color=line_color, linewidth=5)
    ax.plot([90, 90], [30, -1], color=line_color, linewidth=5)
    ax.plot([-2, 92], [-2, -2], color=line_color, linewidth=3)

    # Home goalpost (Haut)
    ax.plot([0, 0], [37, 68], color=line_color, linewidth=5)
    ax.plot([0, 90], [68, 68], color=line_color, linewidth=5)
    ax.plot([90, 90], [68, 37], color=line_color, linewidth=5)
    ax.plot([-2, 92], [36, 36], color=line_color, linewidth=3)

    # Filtrage des outcomes
    def draw_scatters(df, color_team, is_home=True):
        if len(df) == 0:
            return
        
        # Saved shots
        saved = df[df['shot_outcome'] == 'Saved']
        if len(saved) > 0:
            ax.scatter(saved['goalMouthY'], saved['goalMouthZ'],
                       marker='o', c='None', edgecolor=color_team, hatch='/////', s=400,
                       label='Saved' if is_home else '')
        
        # Goals
        goals = df[df['shot_outcome'] == 'Goal']
        if len(goals) > 0:
            ax.scatter(goals['goalMouthY'], goals['goalMouthZ'],
                       marker='o', c='green', edgecolors='white', s=500, linewidth=2, zorder=5,
                       label='Goals' if is_home else '')
        
        # Posts
        posts = df[df['shot_outcome'].isin(['Post', 'Saved to Post'])]
        if len(posts) > 0:
            ax.scatter(posts['goalMouthY'], posts['goalMouthZ'],
                       marker='o', c='None', edgecolors='orange', hatch='/////', s=400,
                       label='Post' if is_home else '')

    draw_scatters(h_shots, acol, True)  # Shots faced by Home
    draw_scatters(a_shots, hcol, False)  # Shots faced by Away

    # --- TEXTES ET STATS - CORRECTION DES PATH_EFFECTS ---
    ax.text(0, 72, f"Home team GK saves", color=hcol, fontsize=25, ha='left', 
           fontweight='bold', path_effects=[path_effects.Stroke(linewidth=3, foreground='black'),
                                           path_effects.Normal()])
    ax.text(0, -8, f"Away team GK saves", color=acol, fontsize=25, ha='left', 
           fontweight='bold', path_effects=[path_effects.Stroke(linewidth=3, foreground='black'),
                                           path_effects.Normal()])

    # xG calculations
    hxgot = round(h_shots['shot_statsbomb_xg'].sum(), 2) if 'shot_statsbomb_xg' in h_shots.columns else 0
    axgot = round(a_shots['shot_statsbomb_xg'].sum(), 2) if 'shot_statsbomb_xg' in a_shots.columns else 0

    ax.text(95, 60, f"xG Faced: {axgot}\nGoals: {len(a_shots[a_shots['shot_outcome']=='Goal'])}", 
           color=hcol, fontsize=12)
    ax.text(95, 20, f"xG Faced: {hxgot}\nGoals: {len(h_shots[h_shots['shot_outcome']=='Goal'])}", 
           color=acol, fontsize=12)

    ax.axis('off')
    ax.set_title("GOAL POST ANALYSIS\nShot Placement Visualization", 
                color='white', fontsize=20, fontweight='bold', pad=20)
    
    return fig

# ============================================
# ORIGINAL ANALYSIS FUNCTIONS (1-13)
# ============================================

# 1. Basic Pass Map Analysis
def create_pass_map(passes_df, team_name, ax, color):
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#c7d5cc',
                  linewidth=2, line_zorder=2)
    pitch.draw(ax=ax)

    # Extract coordinates
    x_start = [loc[0] for loc in passes_df['location']]
    y_start = [loc[1] for loc in passes_df['location']]
    x_end = [loc[0] for loc in passes_df['pass_end_location']]
    y_end = [loc[1] for loc in passes_df['pass_end_location']]

    # Draw passes
    pitch.arrows(x_start, y_start, x_end, y_end,
                width=2, headwidth=5, headlength=5,
                color=color, ax=ax, alpha=0.3, zorder=1)

    # Add starting points
    pitch.scatter(x_start, y_start, s=30, color=color,
                 edgecolors='white', linewidth=1.5, alpha=0.8, ax=ax, zorder=3)

    ax.set_title(f'{team_name}\n{len(passes_df)} successful passes',
                fontsize=16, fontweight='bold', color='white', pad=20)

def analyze_passes(events):
    # Filter successful passes
    passes = events[events['type'] == 'Pass'].copy()
    passes = passes[passes['pass_outcome'].isna()]  # Successful passes only
    
    # Separate by team
    teams = passes['team'].dropna().unique()
    
    team_passes = {}
    for team in teams[:2]:  # Take first two teams
        team_passes[team] = passes[passes['team'] == team]
    
    return team_passes, passes

def generate_pass_statistics(team_passes, all_passes):
    stats = {}
    
    for team, passes_df in team_passes.items():
        team_all_passes = all_passes[all_passes['team'] == team]
        success_rate = len(passes_df) / len(team_all_passes) * 100 if len(team_all_passes) > 0 else 0
        
        # Top 5 passers
        top_passers = passes_df['player'].value_counts().head(5)
        
        stats[team] = {
            'successful_passes': len(passes_df),
            'success_rate': success_rate,
            'top_passers': top_passers
        }
    
    return stats

# 2. Individual Player Pass Maps
def create_player_pass_map(passes_df, player_name, team_name, ax, color):
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#c7d5cc',
                  linewidth=2, line_zorder=2)
    pitch.draw(ax=ax)

    # Filter player passes
    player_passes = passes_df[passes_df['player'] == player_name]

    if len(player_passes) > 0:
        # Extract coordinates
        x_start = [loc[0] for loc in player_passes['location']]
        y_start = [loc[1] for loc in player_passes['location']]
        x_end = [loc[0] for loc in player_passes['pass_end_location']]
        y_end = [loc[1] for loc in player_passes['pass_end_location']]

        # Draw passes
        pitch.arrows(x_start, y_start, x_end, y_end,
                    width=2, headwidth=5, headlength=5,
                    color=color, ax=ax, alpha=0.4, zorder=1)

        # Add starting points
        pitch.scatter(x_start, y_start, s=40, color=color,
                     edgecolors='white', linewidth=2, alpha=0.9, ax=ax, zorder=3)

    # Title with number of passes
    ax.set_title(f'{player_name}\n{len(player_passes)} passes',
                fontsize=11, fontweight='bold', color='white', pad=10)

# 3. Vertical Zone Analysis
def create_player_zone_map(passes_df, player_name, color, ax):
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#1a1a1a', line_color='#555555', linewidth=1)
    pitch.draw(ax=ax)

    # Vertical zone demarcation lines (StatsBomb y goes from 0 to 80)
    # Left: 0-26.6 | Center: 26.6-53.3 | Right: 53.3-80
    ax.axvline(26.6, color='white', linestyle='--', alpha=0.2)
    ax.axvline(53.3, color='white', linestyle='--', alpha=0.2)

    player_p = passes_df[passes_df['player'] == player_name].copy()

    if len(player_p) > 0:
        player_p['x'] = player_p.location.apply(lambda x: x[0])
        player_p['y'] = player_p.location.apply(lambda x: x[1])
        player_p['end_x'] = player_p.pass_end_location.apply(lambda x: x[0])
        player_p['end_y'] = player_p.pass_end_location.apply(lambda x: x[1])

        # Calculate zone percentages
        total = len(player_p)
        left_pct = (player_p[player_p['y'] < 26.6].shape[0] / total) * 100
        center_pct = (player_p[(player_p['y'] >= 26.6) & (player_p['y'] <= 53.3)].shape[0] / total) * 100
        right_pct = (player_p[player_p['y'] > 53.3].shape[0] / total) * 100

        # Draw passes
        pitch.arrows(player_p.x, player_p.y, player_p.end_x, player_p.end_y,
                     color=color, alpha=0.3, width=1, ax=ax)
        pitch.scatter(player_p.x, player_p.y, s=20, color=color, edgecolors='white', ax=ax)

        # Display zone statistics
        stats_text = f"L: {left_pct:.0f}% | C: {center_pct:.0f}% | R: {right_pct:.0f}%"
        ax.set_xlabel(stats_text, color='white', fontsize=9, fontweight='bold')

    ax.set_title(f"{player_name.split()[-1]} ({len(player_p)} P)", color='white', fontsize=12)

# 4. Cross Analysis with Comet Effect
def plot_team_crosses(df, team_name, ax):
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#0e1111', line_color='#555555')
    pitch.draw(ax=ax)

    team_df = df[df['team'] == team_name].copy()

    if len(team_df) > 0:
        # Prepare coordinates
        x = team_df.location.apply(lambda x: x[0]).values
        y = team_df.location.apply(lambda x: x[1]).values
        x_end = team_df.pass_end_location.apply(lambda x: x[0]).values
        y_end = team_df.pass_end_location.apply(lambda x: x[1]).values

        # Separate Outcomes: Successful (NaN) vs Failed (Incomplete, Out, etc.)
        mask_complete = team_df.pass_outcome.isna()

        # COMET EFFECT LINES
        # 1. Successful crosses (Viridis / Cyan)
        if mask_complete.any():
            pitch.lines(x[mask_complete], y[mask_complete],
                        x_end[mask_complete], y_end[mask_complete],
                        comet=True, transparent=True,
                        alpha_start=0.1, alpha_end=0.6,
                        lw=4, cmap='viridis', ax=ax, label='Successful')

        # 2. Failed crosses (Red/Dark for contrast)
        if (~mask_complete).any():
            pitch.lines(x[~mask_complete], y[~mask_complete],
                        x_end[~mask_complete], y_end[~mask_complete],
                        comet=True, transparent=True,
                        alpha_start=0.05, alpha_end=0.2,
                        lw=2, color='#e74c3c', ax=ax, label='Incomplete')

    ax.set_title(f"{team_name} : Cross Analysis", color='white', fontsize=15, pad=10)

# 5. Halfspace Analysis
def plot_halfspace_analysis(df, team_name, ax, team_color):
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#0e1111', line_color='#555555')
    pitch.draw(ax=ax)

    # Draw Halfspace zones visually
    # Left: 18 to 30 | Right: 50 to 62
    ax.axhline(18, color='white', linestyle='--', alpha=0.1)
    ax.axhline(30, color='white', linestyle='--', alpha=0.1)
    ax.axhline(50, color='white', linestyle='--', alpha=0.1)
    ax.axhline(62, color='white', linestyle='--', alpha=0.1)

    # Filter by team
    team_df = df[df['team'] == team_name].copy()

    # Extract coordinates
    team_df['x'] = team_df.location.apply(lambda loc: loc[0])
    team_df['y'] = team_df.location.apply(lambda loc: loc[1])
    team_df['end_x'] = team_df.pass_end_location.apply(lambda loc: loc[0])
    team_df['end_y'] = team_df.pass_end_location.apply(lambda loc: loc[1])

    # Isolate passes STARTING in halfspaces
    hs_passes = team_df[team_df['y'].apply(lambda y: (18 <= y <= 30) or (50 <= y <= 62))].copy()

    # Outcomes
    mask_complete = hs_passes.pass_outcome.isna()

    # COMET TRACING
    if len(hs_passes) > 0:
        # 1. Successful (Comet Viridis)
        if mask_complete.any():
            pitch.lines(hs_passes[mask_complete].x, hs_passes[mask_complete].y,
                        hs_passes[mask_complete].end_x, hs_passes[mask_complete].end_y,
                        comet=True, transparent=True, alpha_start=0.1, alpha_end=0.6,
                        lw=3, cmap='viridis', ax=ax, label='Success')

        # 2. Failed (Red Comet)
        if (~mask_complete).any():
            pitch.lines(hs_passes[~mask_complete].x, hs_passes[~mask_complete].y,
                        hs_passes[~mask_complete].end_x, hs_passes[~mask_complete].end_y,
                        comet=True, transparent=True, alpha_start=0.1, alpha_end=0.3,
                        lw=1.5, color='#e74c3c', ax=ax, label='Failure')

    ax.set_title(f"{team_name}\nPasses from Halfspaces", color='white', fontsize=14)

# 6. Offensive Halfspace Analysis
def plot_offensive_halfspaces(df, team_name, ax, color_cmap):
    # Draw only offensive half (60-120)
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#0e1111', line_color='#444444')
    pitch.draw(ax=ax)

    # Draw Halfspace corridors
    ax.axhline(18, color='white', linestyle=':', alpha=0.3)
    ax.axhline(30, color='white', linestyle=':', alpha=0.3)
    ax.axhline(50, color='white', linestyle=':', alpha=0.3)
    ax.axhline(62, color='white', linestyle=':', alpha=0.3)

    # Filter by team
    team_df = df[df['team'] == team_name].copy()

    # Extract coordinates
    team_df['x'] = team_df.location.apply(lambda loc: loc[0])
    team_df['y'] = team_df.location.apply(lambda loc: loc[1])
    team_df['end_x'] = team_df.pass_end_location.apply(lambda loc: loc[0])
    team_df['end_y'] = team_df.pass_end_location.apply(lambda loc: loc[1])

    # STRICT FILTERS
    # 1. In opponent half (x > 60)
    # 2. In halfspaces (y)
    # 3. Forward pass (end_x > x)
    mask_offensive = (team_df.x >= 60) & \
                     (((team_df.y >= 18) & (team_df.y <= 30)) | ((team_df.y >= 50) & (team_df.y <= 62))) & \
                     (team_df.end_x > team_df.x)

    hs_df = team_df[mask_offensive].copy()
    mask_complete = hs_df.pass_outcome.isna()

    # COMET TRACING
    if len(hs_df) > 0:
        # Successful passes
        pitch.lines(hs_df[mask_complete].x, hs_df[mask_complete].y,
                    hs_df[mask_complete].end_x, hs_df[mask_complete].end_y,
                    comet=True, transparent=True, alpha_start=0.1, alpha_end=0.8,
                    lw=4, cmap=color_cmap, ax=ax)

        # Failed passes (thin red)
        pitch.lines(hs_df[~mask_complete].x, hs_df[~mask_complete].y,
                    hs_df[~mask_complete].end_x, hs_df[~mask_complete].end_y,
                    comet=True, transparent=True, alpha_start=0.1, alpha_end=0.2,
                    lw=1, color='#ff3333', ax=ax)

    # Opponent goal indicator (Direction)
    ax.annotate('', xy=(115, 40), xytext=(105, 40),
                arrowprops=dict(arrowstyle="->", color='white', lw=2))
    ax.text(110, 35, "OPPONENT GOAL", color='white', fontsize=8, ha='center', fontweight='bold')

    ax.set_title(f"{team_name}\nAttack via Offensive Halfspaces", color='white', fontsize=15)

# 7. Zone 14 to Box Penetration
def plot_zone14_to_keeper_box(df, team_name, ax, color_name):
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#0e1111', line_color='#444444')
    pitch.draw(ax=ax)

    # Visual Zone 14 (Rectangle 80-102 in X, 30-50 in Y)
    rect = plt.Rectangle((80, 20), 22, 40, color='white', alpha=0.05, linestyle='--')
    ax.add_patch(rect)

    team_df = df[df['team'] == team_name].copy()

    # Secure coordinate extraction
    team_df['x'] = team_df.location.apply(lambda loc: loc[0])
    team_df['y'] = team_df.location.apply(lambda loc: loc[1])
    team_df['end_x'] = team_df.pass_end_location.apply(lambda loc: loc[0])
    team_df['end_y'] = team_df.pass_end_location.apply(lambda loc: loc[1])

    # FILTER: Start Zone 14 -> Arrival Box (end_x > 102)
    mask = (team_df.x >= 80) & (team_df.x <= 102) & \
           (team_df.y >= 20) & (team_df.y <= 60) & \
           (team_df.end_x >= 102)

    z14_df = team_df[mask].copy()

    if len(z14_df) > 0:
        mask_complete = z14_df.pass_outcome.isna()

        # Successful comet tracing
        if mask_complete.any():
            pitch.lines(z14_df[mask_complete].x, z14_df[mask_complete].y,
                        z14_df[mask_complete].end_x, z14_df[mask_complete].end_y,
                        comet=True, transparent=True, alpha_start=0.2, alpha_end=0.9,
                        lw=5, color=color_name, ax=ax, label='Successful')

        # Failed (discreet red)
        if (~mask_complete).any():
            pitch.lines(z14_df[~mask_complete].x, z14_df[~mask_complete].y,
                        z14_df[~mask_complete].end_x, z14_df[~mask_complete].end_y,
                        comet=True, transparent=True, alpha_start=0.1, alpha_end=0.3,
                        lw=2, color='#ff4757', ax=ax, label='Failed')
    else:
        ax.text(60, 40, "No recorded passes", color='gray', ha='center')

    ax.set_title(f"{team_name}", color='white', fontsize=18, fontweight='bold', pad=15)

# 8. Full Tactical Map (5 vertical corridors)
def plot_full_tactical_map(df, team_name, ax, color_name):
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#0e1111', line_color='#444444')
    pitch.draw(ax=ax)

    # DELIMITATION OF 5 TACTICAL CORRIDORS
    ax.axhline(18, color='white', linestyle='--', alpha=0.2, lw=1)
    ax.axhline(30, color='white', linestyle='--', alpha=0.3, lw=1.5)
    ax.axhline(50, color='white', linestyle='--', alpha=0.3, lw=1.5)
    ax.axhline(62, color='white', linestyle='--', alpha=0.2, lw=1)

    # Filter by team
    team_df = df[df['team'] == team_name].copy()

    # Extract coordinates
    team_df['x'] = team_df.location.apply(lambda loc: loc[0])
    team_df['y'] = team_df.location.apply(lambda loc: loc[1])
    team_df['end_x'] = team_df.pass_end_location.apply(lambda loc: loc[0])
    team_df['end_y'] = team_df.pass_end_location.apply(lambda loc: loc[1])

    mask_complete = team_df.pass_outcome.isna()

    # TRACE ALL PASSES
    if len(team_df) > 0:
        # 1. Successful passes (Comet effect with team color)
        pitch.lines(team_df[mask_complete].x, team_df[mask_complete].y,
                    team_df[mask_complete].end_x, team_df[mask_complete].end_y,
                    comet=True, transparent=True, alpha_start=0.05, alpha_end=0.4,
                    lw=2, color=color_name, ax=ax, label='Successful')

        # 2. Failed passes (Discreet red)
        if (~mask_complete).any():
            pitch.lines(team_df[~mask_complete].x, team_df[~mask_complete].y,
                        team_df[~mask_complete].end_x, team_df[~mask_complete].end_y,
                        comet=True, transparent=True, alpha_start=0.02, alpha_end=0.1,
                        lw=1, color='#ff4757', ax=ax, label='Failed')

    # Zone annotations for clarity
    ax.text(2, 9, "LEFT WING", color='gray', fontsize=8, va='center')
    ax.text(2, 24, "LEFT HS", color='gray', fontsize=8, va='center')
    ax.text(2, 40, "CENTER", color='gray', fontsize=8, va='center')
    ax.text(2, 56, "RIGHT HS", color='gray', fontsize=8, va='center')
    ax.text(2, 71, "RIGHT WING", color='gray', fontsize=8, va='center')

    ax.set_title(f"{team_name}", color='white', fontsize=20, fontweight='bold', pad=15)

# 9. Player Heatmaps
def create_player_heatmap(events_df, player_name, team_name, ax, cmap='hot'):
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#c7d5cc',
                  linewidth=2, line_zorder=2)
    pitch.draw(ax=ax)

    # Filter player events
    player_events = events_df[events_df['player'] == player_name]

    if len(player_events) > 0:
        # Extract coordinates
        x = [loc[0] for loc in player_events['location']]
        y = [loc[1] for loc in player_events['location']]

        # Create hexbin heatmap
        pitch.hexbin(x, y, ax=ax, edgecolors='#22312b', gridsize=15,
                    cmap=cmap, alpha=0.8, zorder=1, linewidths=0.5)

    # Title with number of actions
    ax.set_title(f'{player_name}\n{len(player_events)} actions',
                fontsize=11, fontweight='bold', color='white', pad=10)

# 10. Juego de Posición Heatmap
def create_juego_posicion_heatmap(events_df, team_name, ax, cmap='hot'):
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#c7d5cc',
                  linewidth=2, line_zorder=2)
    pitch.draw(ax=ax)

    # Filter events with location
    team_events = events_df[events_df['team'] == team_name].copy()
    team_events = team_events[team_events['location'].notna()].copy()
    
    if len(team_events) == 0:
        ax.text(0.5, 0.5, f"No location data for {team_name}", color='white', ha='center', va='center')
        return

    # Extract coordinates of all events
    x = np.array([loc[0] for loc in team_events['location']])
    y = np.array([loc[1] for loc in team_events['location']])

    # Create hexbin heatmap
    hexbin = pitch.hexbin(x, y, ax=ax, edgecolors='#22312b', gridsize=20,
                cmap=cmap, alpha=0.85, zorder=1, linewidths=0.5)

    # Add Juego de Posición division lines
    ax.plot([40, 40], [0, 80], color='yellow', linewidth=2.5, linestyle='--', alpha=0.6, zorder=3)
    ax.plot([80, 80], [0, 80], color='yellow', linewidth=2.5, linestyle='--', alpha=0.6, zorder=3)

    ax.plot([0, 120], [16, 16], color='yellow', linewidth=2.5, linestyle='--', alpha=0.6, zorder=3)
    ax.plot([0, 120], [32, 32], color='yellow', linewidth=2.5, linestyle='--', alpha=0.6, zorder=3)
    ax.plot([0, 120], [48, 48], color='yellow', linewidth=2.5, linestyle='--', alpha=0.6, zorder=3)
    ax.plot([0, 120], [64, 64], color='yellow', linewidth=2.5, linestyle='--', alpha=0.6, zorder=3)

    # Zone labels
    zone_labels = {
        'Zone 1 (Defensive)': (20, 75),
        'Zone 2 (Midfield)': (60, 75),
        'Zone 3 (Offensive)': (100, 75),
    }

    for label, (x_pos, y_pos) in zone_labels.items():
        ax.text(x_pos, y_pos, label, fontsize=9, color='white',
               ha='center', va='center', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7),
               zorder=4)

    # Colorbar
    cbar = plt.colorbar(hexbin, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Action Density', rotation=270, labelpad=20, color='white', fontweight='bold')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    # Title
    ax.set_title(f'{team_name}\nJuego de Posición - {len(team_events)} total actions',
                fontsize=14, fontweight='bold', color='white', pad=15)

# 11. Goal Scoring Zones Heatmap (Advanced)
def plot_goal_scoring_zones(match_events, team_name, opponent_name, ax):
    """Heatmap of goal scoring chances by zone with percentages"""
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#1a1a1a', line_color='#606060')
    pitch.draw(ax=ax)
    
    # Filter shots for the selected team
    team_shots = match_events[(match_events['team'] == team_name) & 
                              (match_events['type'] == 'Shot')].copy()
    
    if len(team_shots) > 0:
        # Extract coordinates
        team_shots = team_shots[team_shots['location'].notna()].copy()
        x = np.array([loc[0] for loc in team_shots['location']])
        y = np.array([loc[1] for loc in team_shots['location']])
        
        if len(x) > 0:
            # Create hexbin heatmap
            hexbin = pitch.hexbin(x, y, ax=ax, edgecolors='#1a1a1a', gridsize=12,
                                cmap='Reds', alpha=0.8, zorder=1, linewidths=0.5)
            
            # Colorbar
            cbar = plt.colorbar(hexbin, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Number of Shots', rotation=270, labelpad=20, color='white')
    
    # Title with opponent info
    ax.set_title(f'{team_name} vs {opponent_name}\nGoal Scoring Zones',
                color='white', fontsize=14, fontweight='bold')

# 12. Set Pieces Analysis
def analyze_set_pieces(match_events, team_name, ax):
    """Analyze set pieces (corners, free kicks) with created chances"""
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#1a1a1a', line_color='#606060')
    pitch.draw(ax=ax)
    
    # Filter set pieces
    corners = match_events[(match_events['team'] == team_name) & 
                          (match_events['type'] == 'Pass') &
                          (match_events['pass_type'] == 'Corner')].copy()
    
    free_kicks = match_events[(match_events['team'] == team_name) &
                             (match_events['type'].isin(['Pass', 'Shot'])) &
                             (match_events['play_pattern'].isin(['From Free Kick', 'From Corner']))].copy()
    
    # Filter shots created from set pieces
    shots_from_set_pieces = match_events[(match_events['team'] == team_name) &
                                        (match_events['type'] == 'Shot') &
                                        (match_events['shot_type'].isin(['Free Kick', 'Open Play']))].copy()
    
    # Plot corners
    if len(corners) > 0:
        corners = corners[corners['location'].notna()].copy()
        x_corner = [loc[0] for loc in corners['location']]
        y_corner = [loc[1] for loc in corners['location']]
        ax.scatter(x_corner, y_corner, s=50, color='yellow', 
                  edgecolors='black', alpha=0.7, label='Corners', zorder=3)
    
    # Plot free kicks
    if len(free_kicks) > 0:
        free_kicks = free_kicks[free_kicks['location'].notna()].copy()
        x_fk = [loc[0] for loc in free_kicks['location']]
        y_fk = [loc[1] for loc in free_kicks['location']]
        ax.scatter(x_fk, y_fk, s=40, color='orange', 
                  edgecolors='black', alpha=0.7, label='Free Kicks', zorder=3)
    
    # Plot shots from set pieces
    if len(shots_from_set_pieces) > 0:
        shots_from_set_pieces = shots_from_set_pieces[shots_from_set_pieces['location'].notna()].copy()
        x_shot = [loc[0] for loc in shots_from_set_pieces['location']]
        y_shot = [loc[1] for loc in shots_from_set_pieces['location']]
        ax.scatter(x_shot, y_shot, s=60, color='red', 
                  edgecolors='white', alpha=0.8, label='Shots', zorder=4)
    
    # Calculate statistics
    total_set_pieces = len(corners) + len(free_kicks)
    shots_created = len(shots_from_set_pieces)
    conversion_rate = (shots_created / total_set_pieces * 100) if total_set_pieces > 0 else 0
    
    # Add statistics text
    stats_text = f"""
    Total Set Pieces: {total_set_pieces}
    Shots Created: {shots_created}
    Conversion Rate: {conversion_rate:.1f}%
    """
    ax.text(100, 10, stats_text, color='white', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
           zorder=5)
    
    ax.legend(loc='upper left', fontsize=9)
    ax.set_title(f'{team_name}\nSet Pieces Analysis', color='white', fontsize=14, fontweight='bold')

# 13. Competition-wide Goal Zone Analysis
def analyze_competition_goal_zones(competition_id, season_id, team_name):
    """Analyze goal scoring patterns across competition for a specific team"""
    # This function would need to load all matches for the competition
    # For now, we'll just return a placeholder
    st.info(f"This feature requires loading all matches for competition {competition_id}")
    return None

# ============================================
# DATA LOADING FUNCTIONS
# ============================================

# Function to load competitions (identical to notebook)
@st.cache_data
def load_competitions():
    try:
        competitions = sb.competitions()
        return competitions
    except Exception as e:
        st.error(f"Error loading competitions: {e}")
        return None

# Function to load matches (identical to notebook)
@st.cache_data
def load_matches(competition_id, season_id):
    try:
        matches = sb.matches(competition_id=competition_id, season_id=season_id)
        return matches
    except Exception as e:
        st.error(f"Error loading matches: {e}")
        return None

# Function to load events (identical to notebook)
@st.cache_data
def load_events(match_id):
    try:
        events = sb.events(match_id=match_id)
        return events
    except Exception as e:
        st.error(f"Error loading events: {e}")
        return None

# ============================================
# STREAMLIT APPLICATION WITH NEW ANALYSES
# ============================================

# Sidebar for navigation
st.sidebar.title("⚽ Navigation")
page = st.sidebar.radio(
    "Select a section:",
    ["🏆 1. Load Competitions", 
     "📋 2. Select Match", 
     "📊 3. Basic Analysis",
     "🎨 4. Advanced Visualizations",
     "🔥 5. Advanced Zone Analysis",
     "🎯 6. Goal & Set Pieces Analysis",
     "🏆 7. Competition Analysis",
     "🧠 8. Tactical Analysis (NEW)",
     "📈 9. Expected Threat (xT) Analysis (NEW)",
     "🌀 10. Pass Sonar & Flow (NEW)",
     "🎯 11. Shot & Goal Analysis (NEW)",
     "🛡️ 12. Defensive Analysis (NEW)",
     "🎪 13. Danger Zones Analysis (NEW)"]
)

# Page 1: Load Competitions
if page == "🏆 1. Load Competitions":
    st.header("🏆 AVAILABLE COMPETITIONS")
    st.markdown("=" * 80)
    
    if st.button("🔄 Load all competitions"):
        with st.spinner("Loading competitions..."):
            competitions_df = load_competitions()
            if competitions_df is not None:
                st.session_state.competitions_df = competitions_df
                st.success(f"✅ {len(competitions_df)} competitions loaded successfully!")
    
    if st.session_state.competitions_df is not None:
        # Display like in notebook
        st.dataframe(st.session_state.competitions_df, use_container_width=True)
        
        st.markdown("=" * 80)
        st.subheader("🎯 Competition Selection")
        
        # Selection interface
        col1, col2 = st.columns(2)
        
        with col1:
            competition_id = st.number_input(
                "Competition ID:", 
                min_value=0,
                value=223,
                help="Example: 223 for Copa America"
            )
        
        with col2:
            season_id = st.number_input(
                "Season ID:", 
                min_value=0,
                value=282,
                help="Example: 282 for season 2024"
            )
        
        if st.button("📥 Load matches for this competition"):
            st.session_state.selected_competition_id = int(competition_id)
            st.session_state.selected_season_id = int(season_id)
            st.session_state.selected_match_id = None
            st.session_state.events_df = None
            st.success(f"Competition {competition_id} - Season {season_id} selected!")

# Page 2: Select Match
elif page == "📋 2. Select Match":
    if st.session_state.selected_competition_id is None:
        st.warning("⚠️ Please first load a competition in step 1.")
    else:
        st.header(f"📋 AVAILABLE MATCHES FOR COMPETITION {st.session_state.selected_competition_id} - SEASON {st.session_state.selected_season_id}")
        st.markdown("=" * 80)
        
        with st.spinner(f"Loading matches for competition {st.session_state.selected_competition_id}..."):
            matches_df = load_matches(
                st.session_state.selected_competition_id,
                st.session_state.selected_season_id
            )
        
        if matches_df is not None:
            st.session_state.matches_df = matches_df
            
            # Display columns like in notebook
            display_cols = ['match_id', 'match_date', 'home_team', 'away_team', 'home_score', 'away_score']
            display_df = matches_df[display_cols].copy()
            
            st.dataframe(display_df, use_container_width=True)
            
            st.markdown("=" * 80)
            st.subheader("🎯 Match Selection")
            
            # Selection interface
            match_id = st.number_input(
                "Enter the Match ID you want to analyze:", 
                min_value=0,
                value=3895302,  # Default to Leverkusen vs Bremen
                help="Example: 3895302 for Bayer Leverkusen vs Werder Bremen"
            )
            
            # Get match info for display
            if len(matches_df) > 0:
                match_info = matches_df[matches_df['match_id'] == match_id]
                if len(match_info) > 0:
                    st.session_state.selected_match_info = match_info.iloc[0].to_dict()
            
            if st.button("⚽ Load and analyze this match"):
                st.session_state.selected_match_id = int(match_id)
                st.session_state.events_df = None
                st.session_state.all_analysis_data = {}
                st.success(f"Match {match_id} selected! Move to step 3 for analysis.")

# Page 3: Basic Analysis
elif page == "📊 3. Basic Analysis":
    if st.session_state.selected_match_id is None:
        st.warning("⚠️ Please first select a match in step 2.")
    else:
        st.header(f"📊 LOADING EVENTS FOR MATCH {st.session_state.selected_match_id}")
        st.markdown("=" * 80)
        
        if st.session_state.events_df is None:
            with st.spinner(f"Loading events for match {st.session_state.selected_match_id}..."):
                events_df = load_events(st.session_state.selected_match_id)
                if events_df is not None:
                    st.session_state.events_df = events_df
                    st.success("✅ Data loaded successfully!")
        
        if st.session_state.events_df is not None:
            events_df = st.session_state.events_df
            
            # Display information like in notebook
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total number of events", len(events_df))
            
            with col2:
                st.metric("Number of columns", len(events_df.columns))
            
            st.markdown("---")
            st.subheader("📋 Available columns (like in notebook):")
            st.write(f"```python\n{list(events_df.columns)}\n```")
            
            st.markdown("---")
            st.subheader("👀 Preview of first events (like in notebook):")
            
            # Select columns to display
            preview_cols = ['minute', 'second', 'team', 'player', 'type']
            available_cols = [col for col in preview_cols if col in events_df.columns]
            
            st.dataframe(events_df[available_cols].head(10), use_container_width=True)
            
            st.markdown("=" * 80)
            st.success("✅ You can now analyze the match!")
            
            # Pass analysis section (identical to notebook)
            st.markdown("---")
            st.header("🔍 Pass Analysis (Identical to notebook)")
            
            if st.button("📊 Run complete pass analysis"):
                with st.spinner("Analyzing passes..."):
                    # Analyze passes
                    team_passes, all_passes = analyze_passes(events_df)
                    
                    # Generate statistics
                    stats = generate_pass_statistics(team_passes, all_passes)
                    
                    # Store in session state
                    st.session_state.all_analysis_data['team_passes'] = team_passes
                    st.session_state.all_analysis_data['pass_stats'] = stats
                    
                    st.success("✅ Pass analysis completed!")
            
            # Display results if available
            if 'pass_stats' in st.session_state.all_analysis_data:
                stats = st.session_state.all_analysis_data['pass_stats']
                
                st.markdown("---")
                st.subheader("📊 PASS STATISTICS (Like in notebook)")
                
                for team, team_stats in stats.items():
                    st.markdown(f"**{team}:**")
                    st.write(f"  - Successful passes: {team_stats['successful_passes']}")
                    st.write(f"  - Success rate: {team_stats['success_rate']:.1f}%")
                    st.write("")
                
                st.markdown("---")
                st.subheader("🎯 TOP 5 PASSERS (Like in notebook)")
                
                for team, team_stats in stats.items():
                    st.markdown(f"**{team}:**")
                    top_passers = team_stats['top_passers']
                    for i, (player, count) in enumerate(top_passers.items(), 1):
                        st.write(f"  {i}. {player}: {count} passes")
                    st.write("")
            
            # Generate pass map visualization
            if 'team_passes' in st.session_state.all_analysis_data and len(st.session_state.all_analysis_data['team_passes']) >= 2:
                st.markdown("---")
                st.subheader("🗺️ Pass Map Visualization")
                
                team_passes = st.session_state.all_analysis_data['team_passes']
                
                # Create figure with 2 subplots
                fig, axes = plt.subplots(1, 2, figsize=(24, 10))
                fig.patch.set_facecolor('#22312b')
                
                # Colors for each team (red and green like in notebook)
                colors = ['#e30613', '#1d9053']
                
                teams = list(team_passes.keys())
                for idx, team in enumerate(teams[:2]):
                    create_pass_map(team_passes[team], team, axes[idx], colors[idx])
                
                # General title
                title = 'Pass Maps'
                if st.session_state.selected_match_info:
                    home = st.session_state.selected_match_info['home_team']
                    away = st.session_state.selected_match_info['away_team']
                    home_score = st.session_state.selected_match_info['home_score']
                    away_score = st.session_state.selected_match_info['away_score']
                    title += f' - {home} {home_score}-{away_score} {away}'
                
                fig.suptitle(title,
                            fontsize=20, fontweight='bold', color='white', y=0.98)
                
                plt.tight_layout()
                st.pyplot(fig)

# Page 4: Advanced Visualizations (Original)
elif page == "🎨 4. Advanced Visualizations":
    if st.session_state.events_df is None:
        st.warning("⚠️ Please first load and analyze a match in step 3.")
    else:
        st.header("🎨 Advanced Visualizations")
        
        # Tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs([
            "👤 Individual Player Analysis", 
            "🎯 Cross Analysis",
            "⚡ Halfspace Analysis",
            "🏟️ Zone 14 Penetration"
        ])
        
        with tab1:
            st.subheader("👤 Individual Player Pass Maps")
            
            # Get teams
            events_df = st.session_state.events_df
            passes = events_df[(events_df['type'] == 'Pass') & 
                              (events_df['pass_outcome'].isna())].copy()
            
            teams = passes['team'].dropna().unique()
            
            if len(teams) >= 2:
                team1, team2 = teams[0], teams[1]
                
                # Team selection
                selected_team = st.selectbox("Select team:", teams[:2])
                
                if selected_team:
                    team_passes = passes[passes['team'] == selected_team]
                    
                    # Top players
                    top_players = team_passes['player'].value_counts().head(12).index.tolist()
                    
                    # Create grid
                    n_players = len(top_players)
                    n_cols = 4
                    n_rows = int(np.ceil(n_players / n_cols))
                    
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows*5))
                    fig.patch.set_facecolor('#22312b')
                    axes = axes.flatten()
                    
                    # Color based on team
                    color = '#e30613' if selected_team == team1 else '#1d9053'
                    
                    for idx, player in enumerate(top_players):
                        create_player_pass_map(team_passes, player, selected_team,
                                              axes[idx], color)
                    
                    # Hide empty axes
                    for idx in range(len(top_players), len(axes)):
                        axes[idx].axis('off')
                    
                    fig.suptitle(f'{selected_team} - Individual Pass Maps',
                                fontsize=18, fontweight='bold', color='white', y=0.995)
                    plt.tight_layout()
                    st.pyplot(fig)
        
        with tab2:
            st.subheader("🎯 Cross Analysis with Comet Effect")
            
            events_df = st.session_state.events_df
            
            # Filter crosses
            crosses = events_df[(events_df['type'] == 'Pass') & 
                               (events_df['pass_cross'] == True)].copy()
            
            if len(crosses) > 0:
                teams = crosses['team'].unique()
                
                fig, axes = plt.subplots(1, 2, figsize=(22, 10))
                fig.patch.set_facecolor('#0e1111')
                
                for i, team in enumerate(teams[:2]):
                    plot_team_crosses(crosses, team, axes[i])
                
                plt.suptitle(f"CROSS ANALYSIS WITH COMET EFFECT",
                            color='#00f2ff', fontsize=20, y=0.95, fontweight='bold')
                st.pyplot(fig)
            else:
                st.info("No crosses found in this match.")
        
        with tab3:
            st.subheader("⚡ Halfspace Analysis")
            
            events_df = st.session_state.events_df
            all_passes = events_df[events_df['type'] == 'Pass'].copy()
            
            if len(all_passes) > 0:
                teams = all_passes['team'].unique()
                
                fig, axes = plt.subplots(1, 2, figsize=(22, 10))
                fig.patch.set_facecolor('#0e1111')
                
                colors = ['#e30613', '#1d9053']
                
                for i, team in enumerate(teams[:2]):
                    plot_halfspace_analysis(all_passes, team, axes[i], colors[i])
                
                plt.suptitle("HALFSPACE DOMINATION | COMET EFFECT",
                            color='#00f2ff', fontsize=22, fontweight='bold', y=0.98)
                st.pyplot(fig)
        
        with tab4:
            st.subheader("🏟️ Zone 14 to Box Penetration")
            
            events_df = st.session_state.events_df
            all_passes = events_df[events_df['type'] == 'Pass'].copy()
            
            if len(all_passes) > 0:
                teams = all_passes['team'].unique()
                
                fig, axes = plt.subplots(1, 2, figsize=(22, 12))
                fig.set_facecolor('#0e1111')
                
                # Colors
                plot_zone14_to_keeper_box(all_passes, teams[0], axes[0], '#00f2ff')
                plot_zone14_to_keeper_box(all_passes, teams[1], axes[1], '#10ac84')
                
                # Attack direction
                axes[1].annotate('', xy=(0.6, 0.03), xytext=(0.4, 0.03),
                                xycoords='figure fraction',
                                arrowprops=dict(arrowstyle="->", color='#00f2ff', lw=3))
                
                fig.text(0.5, 0.015, "ATTACK DIRECTION →",
                        color='#00f2ff', fontsize=12, ha='center', fontweight='bold')
                
                plt.suptitle("PENETRATION FROM ZONE 14 TO BOX",
                            color='white', fontsize=24, fontweight='bold', y=0.96)
                
                plt.tight_layout(rect=[0, 0.05, 1, 0.95])
                st.pyplot(fig)

# Page 5: Advanced Zone Analysis (Original)
elif page == "🔥 5. Advanced Zone Analysis":
    if st.session_state.events_df is None:
        st.warning("⚠️ Please first load and analyze a match in step 3.")
    else:
        st.header("🔥 Advanced Zone Analysis")
        
        tab1, tab2, tab3 = st.tabs([
            "📊 Tactical Map (5 Corridors)",
            "🔥 Player Heatmaps",
            "🏰 Juego de Posición"
        ])
        
        with tab1:
            st.subheader("📊 Full Tactical Map (5 Vertical Corridors)")
            
            events_df = st.session_state.events_df
            all_passes = events_df[events_df['type'] == 'Pass'].copy()
            
            if len(all_passes) > 0:
                teams = all_passes['team'].unique()
                
                fig, axes = plt.subplots(2, 1, figsize=(18, 22))
                fig.set_facecolor('#0e1111')
                
                # Team 1 in Red, Team 2 in Green
                plot_full_tactical_map(all_passes, teams[0], axes[0], '#e30613')
                plot_full_tactical_map(all_passes, teams[1], axes[1], '#10ac84')
                
                # Attack direction
                axes[1].annotate('', xy=(0.6, 0.03), xytext=(0.4, 0.03),
                                xycoords='figure fraction',
                                arrowprops=dict(arrowstyle="->", color='#00f2ff', lw=3))
                
                fig.text(0.5, 0.015, "ATTACK DIRECTION →",
                        color='#00f2ff', fontsize=12, ha='center', fontweight='bold')
                
                plt.suptitle("TACTICAL STRUCTURE AND BALL CIRCULATION\nDivision into 5 vertical corridors",
                            color='white', fontsize=26, fontweight='bold', y=0.97)
                
                plt.tight_layout(rect=[0, 0.05, 1, 0.95])
                st.pyplot(fig)
        
        with tab2:
            st.subheader("🔥 Individual Player Heatmaps")
            
            events_df = st.session_state.events_df
            events_with_location = events_df[events_df['location'].notna()].copy()
            
            teams = events_with_location['team'].dropna().unique()
            
            if len(teams) >= 2:
                selected_team = st.selectbox("Select team for heatmaps:", teams[:2])
                
                if selected_team:
                    team_events = events_with_location[events_with_location['team'] == selected_team]
                    
                    # Top players
                    top_players = team_events['player'].value_counts().head(12).index.tolist()
                    
                    # Create grid
                    n_players = len(top_players)
                    n_cols = 4
                    n_rows = int(np.ceil(n_players / n_cols))
                    
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows*5))
                    fig.patch.set_facecolor('#22312b')
                    axes = axes.flatten()
                    
                    # Color map based on team
                    cmap = 'Reds' if selected_team == teams[0] else 'Greens'
                    
                    for idx, player in enumerate(top_players):
                        create_player_heatmap(team_events, player, selected_team,
                                             axes[idx], cmap=cmap)
                    
                    # Hide empty axes
                    for idx in range(len(top_players), len(axes)):
                        axes[idx].axis('off')
                    
                    fig.suptitle(f'{selected_team} - Player Heatmaps (Activity Zones)',
                                fontsize=18, fontweight='bold', color='white', y=0.995)
                    plt.tight_layout()
                    st.pyplot(fig)
        
        with tab3:
            st.subheader("🏰 Juego de Posición Heatmap")
            
            events_df = st.session_state.events_df
            events_with_location = events_df[events_df['location'].notna()].copy()
            
            teams = events_with_location['team'].dropna().unique()
            
            if len(teams) >= 2:
                team1, team2 = teams[0], teams[1]
                
                fig, axes = plt.subplots(1, 2, figsize=(28, 12))
                fig.patch.set_facecolor('#22312b')
                
                # Heatmap for each team
                create_juego_posicion_heatmap(
                    events_with_location[events_with_location['team'] == team1],
                    team1, axes[0], cmap='Reds'
                )
                
                create_juego_posicion_heatmap(
                    events_with_location[events_with_location['team'] == team2],
                    team2, axes[1], cmap='Greens'
                )
                
                # General title
                match_info = ""
                if st.session_state.selected_match_info:
                    match_info = f"{st.session_state.selected_match_info['home_team']} {st.session_state.selected_match_info['home_score']}-{st.session_state.selected_match_info['away_score']} {st.session_state.selected_match_info['away_team']}"
                
                fig.suptitle(f'🏰 JUEGO DE POSICIÓN HEATMAP - Zone Occupation Analysis\n{match_info}',
                            fontsize=20, fontweight='bold', color='white', y=0.98)
                
                plt.tight_layout()
                st.pyplot(fig)

# Page 6: Goal & Set Pieces Analysis (Original)
elif page == "🎯 6. Goal & Set Pieces Analysis":
    if st.session_state.events_df is None:
        st.warning("⚠️ Please first load and analyze a match in step 3.")
    else:
        st.header("🎯 Goal Scoring & Set Pieces Analysis")
        
        tab1, tab2 = st.tabs([
            "⚽ Goal Scoring Zones",
            "🎯 Set Pieces Analysis"
        ])
        
        with tab1:
            st.subheader("⚽ Goal Scoring Zones Analysis")
            
            events_df = st.session_state.events_df
            
            # Get teams
            teams = events_df['team'].dropna().unique()
            
            if len(teams) >= 2:
                team1, team2 = teams[0], teams[1]
                
                # Create visualization
                fig, axes = plt.subplots(1, 2, figsize=(20, 10))
                fig.patch.set_facecolor('#1a1a1a')
                
                # Plot for each team
                plot_goal_scoring_zones(events_df, team1, team2, axes[0])
                plot_goal_scoring_zones(events_df, team2, team1, axes[1])
                
                plt.suptitle(f"GOAL SCORING ZONES ANALYSIS\n{team1} vs {team2}",
                            color='white', fontsize=18, fontweight='bold', y=0.98)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Display statistics
                st.markdown("---")
                st.subheader("📊 Shooting Statistics")
                
                for team in [team1, team2]:
                    team_shots = events_df[(events_df['team'] == team) & 
                                          (events_df['type'] == 'Shot')].copy()
                    team_goals = team_shots[team_shots['shot_outcome'] == 'Goal']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(f"{team} Total Shots", len(team_shots))
                    with col2:
                        st.metric(f"{team} Goals", len(team_goals))
                    with col3:
                        conversion = (len(team_goals) / len(team_shots) * 100) if len(team_shots) > 0 else 0
                        st.metric(f"{team} Conversion Rate", f"{conversion:.1f}%")
        
        with tab2:
            st.subheader("🎯 Set Pieces Analysis")
            
            events_df = st.session_state.events_df
            
            # Get teams
            teams = events_df['team'].dropna().unique()
            
            if len(teams) >= 2:
                selected_team = st.selectbox("Select team for set pieces analysis:", teams[:2])
                
                if selected_team:
                    # Create visualization
                    fig, ax = plt.subplots(figsize=(14, 10))
                    fig.patch.set_facecolor('#1a1a1a')
                    
                    analyze_set_pieces(events_df, selected_team, ax)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Additional statistics
                    st.markdown("---")
                    st.subheader("📊 Detailed Set Pieces Statistics")
                    
                    # Count different types of set pieces
                    corners = events_df[(events_df['team'] == selected_team) & 
                                       (events_df['type'] == 'Pass') &
                                       (events_df['pass_type'] == 'Corner')].copy()
                    
                    free_kicks = events_df[(events_df['team'] == selected_team) &
                                          (events_df['type'].isin(['Pass', 'Shot'])) &
                                          (events_df['play_pattern'].isin(['From Free Kick', 'From Corner']))].copy()
                    
                    shots_from_set_pieces = events_df[(events_df['team'] == selected_team) &
                                                     (events_df['type'] == 'Shot') &
                                                     (events_df['shot_type'].isin(['Free Kick', 'Open Play']))].copy()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Corners", len(corners))
                    with col2:
                        st.metric("Free Kicks", len(free_kicks))
                    with col3:
                        st.metric("Shots from Set Pieces", len(shots_from_set_pieces))

# Page 7: Competition Analysis (Original)
elif page == "🏆 7. Competition Analysis":
    if st.session_state.selected_competition_id is None:
        st.warning("⚠️ Please first load a competition in step 1.")
    else:
        st.header("🏆 Competition-wide Analysis")
        
        tab1, tab2 = st.tabs([
            "⚽ Team Goal Zones in Competition",
            "📈 Competition Statistics"
        ])
        
        with tab1:
            st.subheader("⚽ Team Goal Scoring Patterns in Competition")
            
            # Get all teams in the competition
            matches_df = st.session_state.matches_df
            
            if matches_df is not None:
                # Get unique teams
                all_teams = pd.concat([matches_df['home_team'], matches_df['away_team']]).unique()
                
                selected_team = st.selectbox("Select team for competition analysis:", all_teams)
                
                if selected_team:
                    st.info("This feature would load all matches for the team in the competition.")
                    st.info("For now, please use the match-specific analyses.")
                    # Note: The full implementation would require loading all match events
                    # which could be time-consuming and memory-intensive
        
        with tab2:
            st.subheader("📈 Competition Statistics")
            
            if st.session_state.matches_df is not None:
                matches_df = st.session_state.matches_df.copy()
                
                # Basic statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Matches", len(matches_df))
                with col2:
                    st.metric("Unique Teams", len(pd.concat([matches_df['home_team'], matches_df['away_team']]).unique()))
                with col3:
                    avg_goals = (matches_df['home_score'].sum() + matches_df['away_score'].sum()) / len(matches_df)
                    st.metric("Average Goals per Match", f"{avg_goals:.2f}")
                
                # Top scoring teams
                st.markdown("---")
                st.subheader("🏆 Top Scoring Teams")
                
                # Calculate goals for and against
                home_goals = matches_df.groupby('home_team')['home_score'].sum()
                away_goals = matches_df.groupby('away_team')['away_score'].sum()
                total_goals = home_goals.add(away_goals, fill_value=0).sort_values(ascending=False).head(10)
                
                # Create bar chart
                fig, ax = plt.subplots(figsize=(12, 6))
                total_goals.plot(kind='bar', ax=ax, color='skyblue')
                ax.set_xlabel('Team')
                ax.set_ylabel('Goals Scored')
                ax.set_title('Top 10 Scoring Teams')
                plt.xticks(rotation=45)
                st.pyplot(fig)
                
                # Match results distribution
                st.markdown("---")
                st.subheader("📊 Match Results Distribution")
                
                # Calculate results
                matches_df['result'] = np.where(matches_df['home_score'] > matches_df['away_score'], 'Home Win',
                                               np.where(matches_df['home_score'] < matches_df['away_score'], 'Away Win', 'Draw'))
                
                result_counts = matches_df['result'].value_counts()
                
                fig, ax = plt.subplots(figsize=(8, 8))
                result_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%', startangle=90)
                ax.set_ylabel('')
                ax.set_title('Distribution of Match Results')
                st.pyplot(fig)

# Page 8: NEW - Tactical Analysis
elif page == "🧠 8. Tactical Analysis (NEW)":
    if st.session_state.events_df is None:
        st.warning("⚠️ Please first load and analyze a match in step 3.")
    else:
        st.header("🧠 Advanced Tactical Analysis")
        
        tab1, tab2, tab3 = st.tabs([
            "📊 Positional Heatmaps",
            "👥 Individual Player Grids",
            "🎭 Player Thirds Distribution"
        ])
        
        with tab1:
            st.subheader("📊 Positional Heatmaps (Juego de Posición)")
            
            events_df = st.session_state.events_df
            teams = events_df['team'].dropna().unique()
            
            if len(teams) >= 2:
                team1, team2 = teams[0], teams[1]
                
                if st.button("Generate Positional Heatmaps"):
                    with st.spinner("Creating positional heatmaps..."):
                        fig = run_full_positional_analysis(
                            events_df, team1, team2, 
                            color_a='#e32221', color_b='#009444'
                        )
                        st.pyplot(fig)
                        
                        st.markdown("---")
                        st.info("""
                        **Positional Heatmap Analysis:**
                        - Shows where each team spends most time on the pitch
                        - Divided into 30 tactical zones
                        - Percentages show time spent in each zone
                        - Red/Green intensity indicates concentration of play
                        """)
        
        with tab2:
            st.subheader("👥 Individual Player Positional Grids")
            
            events_df = st.session_state.events_df
            teams = events_df['team'].dropna().unique()
            
            if len(teams) >= 2:
                selected_team = st.selectbox("Select team for individual analysis:", teams[:2])
                
                if selected_team:
                    if st.button(f"Generate {selected_team} Player Grids"):
                        with st.spinner(f"Creating player grids for {selected_team}..."):
                            color = '#e32221' if selected_team == teams[0] else '#009444'
                            fig = plot_player_positional_grid(events_df, selected_team, color)
                            st.pyplot(fig)
                            
                            st.markdown("---")
                            st.info("""
                            **Individual Player Grids:**
                            - Shows tactical positioning of each player
                            - Heatmap indicates zone occupation
                            - Percentage shows time spent in each zone
                            - Useful for analyzing player roles and positioning
                            """)
        
        with tab3:
            st.subheader("🎭 Player Action Distribution by Thirds")
            
            events_df = st.session_state.events_df
            teams = events_df['team'].dropna().unique()
            
            if len(teams) >= 2:
                selected_team = st.selectbox("Select team for thirds analysis:", teams[:2], key='thirds_team')
                
                if selected_team:
                    if st.button(f"Generate {selected_team} Thirds Distribution"):
                        with st.spinner(f"Creating thirds analysis for {selected_team}..."):
                            color_map = 'Reds' if selected_team == teams[0] else 'Greens'
                            fig = plot_player_thirds(events_df, selected_team, color_map)
                            st.pyplot(fig)
                            
                            st.markdown("---")
                            st.info("""
                            **Thirds Distribution Analysis:**
                            - Shows where each player's actions occur
                            - Defensive Third (0-40m): Defensive actions
                            - Middle Third (40-80m): Build-up play
                            - Attacking Third (80-120m): Final third actions
                            - Useful for understanding player roles and team structure
                            """)

# Page 9: NEW - Expected Threat (xT) Analysis
elif page == "📈 9. Expected Threat (xT) Analysis (NEW)":
    if st.session_state.events_df is None:
        st.warning("⚠️ Please first load and analyze a match in step 3.")
    else:
        st.header("📈 Expected Threat (xT) Analysis")
        
        tab1, tab2 = st.tabs([
            "⚡ Team xT Comparison",
            "🔗 xT & Action Chains"
        ])
        
        with tab1:
            st.subheader("⚡ Team Expected Threat Comparison")
            
            events_df = st.session_state.events_df
            teams = events_df['team'].dropna().unique()
            
            if len(teams) >= 2:
                team1, team2 = teams[0], teams[1]
                
                if st.button("Generate xT Analysis"):
                    with st.spinner("Calculating Expected Threat..."):
                        fig = plot_xt_for_teams(events_df, team1, team2)
                        st.pyplot(fig)
                        
                        st.markdown("---")
                        st.info("""
                        **Expected Threat (xT) Analysis:**
                        - Measures the probability of scoring from each zone
                        - Combines: Probability of shooting × Probability of scoring
                        - Higher values (red/green) indicate more dangerous zones
                        - Shows where teams create the most threat
                        """)
        
        with tab2:
            st.subheader("🔗 xT & Action Chains Analysis")
            
            events_df = st.session_state.events_df
            teams = events_df['team'].dropna().unique()
            
            if len(teams) >= 2:
                selected_team = st.selectbox("Select team for chains analysis:", teams[:2], key='xt_chains')
                
                if selected_team:
                    if st.button(f"Generate {selected_team} Action Chains"):
                        with st.spinner(f"Creating xT and action chains for {selected_team}..."):
                            fig = run_combined_xt_chains_analysis(events_df, selected_team)
                            st.pyplot(fig)
                            
                            st.markdown("---")
                            st.info("""
                            **xT & Action Chains Analysis:**
                            - Background: xT heatmap shows dangerous zones
                            - Yellow stars: Shot locations
                            - Cyan arrows: Last 5 actions before each shot
                            - Shows build-up patterns leading to shots
                            - Identifies dangerous passing sequences
                            """)

# Page 10: NEW - Pass Sonar & Flow
elif page == "🌀 10. Pass Sonar & Flow (NEW)":
    if st.session_state.events_df is None:
        st.warning("⚠️ Please first load and analyze a match in step 3.")
    else:
        st.header("🌀 Pass Sonar & Flow Analysis")
        
        tab1, tab2 = st.tabs([
            "🎵 Pass Sonar Analysis",
            "🌊 Pass Flow Patterns"
        ])
        
        with tab1:
            st.subheader("🎵 Pass Sonar Analysis")
            
            events_df = st.session_state.events_df
            teams = events_df['team'].dropna().unique()
            
            if len(teams) >= 2:
                team1, team2 = teams[0], teams[1]
                
                if st.button("Generate Pass Sonar Comparison"):
                    with st.spinner("Creating pass sonar visualization..."):
                        fig = run_full_sonar_comparison(events_df, team1, team2)
                        st.pyplot(fig)
                        
                        st.markdown("---")
                        st.info("""
                        **Pass Sonar Analysis:**
                        - Shows passing directions from different zones
                        - Wedges indicate pass direction distribution
                        - Zone leaders show dominant players in each area
                        - Useful for analyzing passing patterns and player influence
                        """)
        
        with tab2:
            st.subheader("🌊 Pass Flow Patterns")
            
            events_df = st.session_state.events_df
            teams = events_df['team'].dropna().unique()
            
            if len(teams) >= 2:
                team1, team2 = teams[0], teams[1]
                
                if st.button("Generate Pass Flow Comparison"):
                    with st.spinner("Creating pass flow visualization..."):
                        fig = run_comparison_pass_flow(events_df, team1, team2)
                        st.pyplot(fig)
                        
                        st.markdown("---")
                        st.info("""
                        **Pass Flow Analysis:**
                        - Heatmap: Density of pass origins
                        - Arrows: Direction and volume of passes
                        - Shows main passing corridors and tendencies
                        - Useful for analyzing team's passing strategy
                        """)

# Page 11: NEW - Shot & Goal Analysis
elif page == "🎯 11. Shot & Goal Analysis (NEW)":
    if st.session_state.events_df is None:
        st.warning("⚠️ Please first load and analyze a match in step 3.")
    else:
        st.header("🎯 Advanced Shot & Goal Analysis")
        
        tab1, tab2, tab3 = st.tabs([
            "⚽ Vertical Shot Maps",
            "🥅 Goal Post Analysis",
            "📊 Shot Statistics"
        ])
        
        with tab1:
            st.subheader("⚽ Vertical Shot Maps")
            
            events_df = st.session_state.events_df
            teams = events_df['team'].dropna().unique()
            
            if len(teams) >= 2:
                team1, team2 = teams[0], teams[1]
                
                if st.button("Generate Shot Map Comparison"):
                    with st.spinner("Creating shot map visualization..."):
                        fig = run_side_by_side_shot_comparison(events_df, team1, team2)
                        st.pyplot(fig)
                        
                        st.markdown("---")
                        st.info("""
                        **Vertical Shot Map Analysis:**
                        - Shows shot locations on vertical half-pitch
                        - Circle size: Expected Goals (xG) value
                        - Hatched circles: Missed shots
                        - White circles: Goals
                        - Shows shooting patterns and efficiency
                        """)
        
        with tab2:
            st.subheader("🥅 Goal Post Analysis")
            
            events_df = st.session_state.events_df
            teams = events_df['team'].dropna().unique()
            
            if len(teams) >= 2:
                team1, team2 = teams[0], teams[1]
                
                if st.button("Generate Goal Post Analysis"):
                    with st.spinner("Creating goal post visualization..."):
                        fig = plot_goal_post_analysis(events_df, team1, team2)
                        st.pyplot(fig)
                        
                        st.markdown("---")
                        st.info("""
                        **Goal Post Analysis:**
                        - Shows actual shot placement in goal
                        - Top goal: Shots faced by home team
                        - Bottom goal: Shots faced by away team
                        - Green dots: Goals scored
                        - Colored circles: Shots saved
                        - Orange circles: Shots hitting post
                        - xG Faced: Total expected goals conceded
                        """)
        
        with tab3:
            st.subheader("📊 Detailed Shot Statistics")
            
            events_df = st.session_state.events_df
            
            # Get teams
            teams = events_df['team'].dropna().unique()
            
            if len(teams) >= 2:
                team1, team2 = teams[0], teams[1]
                
                # Calculate detailed statistics
                shot_stats = {}
                for team in [team1, team2]:
                    team_shots = events_df[(events_df['team'] == team) & (events_df['type'] == 'Shot')].copy()
                    
                    if len(team_shots) > 0:
                        # Goals
                        goals = team_shots[team_shots['shot_outcome'] == 'Goal']
                        
                        # Shots on target
                        on_target = team_shots[team_shots['shot_outcome'].isin(['Goal', 'Saved', 'Saved to Post'])]
                        
                        # Shots off target
                        off_target = team_shots[team_shots['shot_outcome'].isin(['Off T', 'Wayward', 'Blocked'])]
                        
                        # Posts
                        posts = team_shots[team_shots['shot_outcome'].isin(['Post', 'Saved to Post'])]
                        
                        # xG
                        total_xg = team_shots['shot_statsbomb_xg'].sum() if 'shot_statsbomb_xg' in team_shots.columns else 0
                        
                        shot_stats[team] = {
                            'total_shots': len(team_shots),
                            'goals': len(goals),
                            'on_target': len(on_target),
                            'off_target': len(off_target),
                            'posts': len(posts),
                            'total_xg': round(total_xg, 2),
                            'conversion_rate': (len(goals) / len(team_shots) * 100) if len(team_shots) > 0 else 0,
                            'on_target_percentage': (len(on_target) / len(team_shots) * 100) if len(team_shots) > 0 else 0
                        }
                
                # Display statistics in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    if team1 in shot_stats:
                        st.subheader(f"{team1}")
                        stats = shot_stats[team1]
                        st.metric("Total Shots", stats['total_shots'])
                        st.metric("Goals", stats['goals'])
                        st.metric("On Target", f"{stats['on_target']} ({stats['on_target_percentage']:.1f}%)")
                        st.metric("Conversion Rate", f"{stats['conversion_rate']:.1f}%")
                        st.metric("Total xG", stats['total_xg'])
                        if stats['posts'] > 0:
                            st.metric("Hit Post", stats['posts'])
                
                with col2:
                    if team2 in shot_stats:
                        st.subheader(f"{team2}")
                        stats = shot_stats[team2]
                        st.metric("Total Shots", stats['total_shots'])
                        st.metric("Goals", stats['goals'])
                        st.metric("On Target", f"{stats['on_target']} ({stats['on_target_percentage']:.1f}%)")
                        st.metric("Conversion Rate", f"{stats['conversion_rate']:.1f}%")
                        st.metric("Total xG", stats['total_xg'])
                        if stats['posts'] > 0:
                            st.metric("Hit Post", stats['posts'])

# Page 12: NEW - Defensive Analysis
elif page == "🛡️ 12. Defensive Analysis (NEW)":
    if st.session_state.events_df is None:
        st.warning("⚠️ Please first load and analyze a match in step 3.")
    else:
        st.header("🛡️ Defensive Analysis")
        
        tab1, tab2, tab3 = st.tabs([
            "🛡️ Defensive Block Structure",
            "🎨 Elegant Defensive Blocks",
            "👤 Individual Defensive Actions"
        ])
        
        with tab1:
            st.subheader("🛡️ Defensive Block Structure")
            
            events_df = st.session_state.events_df
            teams = events_df['team'].dropna().unique()
            
            if len(teams) >= 2:
                team1, team2 = teams[0], teams[1]
                
                if st.button("Generate Defensive Block Analysis"):
                    with st.spinner("Creating defensive block visualization..."):
                        fig, axes = plt.subplots(1, 2, figsize=(22, 12), facecolor='#0e1111')
                        
                        plot_defensive_block(events_df, team1, axes[0], '#e30613')
                        plot_defensive_block(events_df, team2, axes[1], '#10ac84')
                        
                        plt.suptitle("DEFENSIVE BLOCK ANALYSIS", 
                                    color='white', fontsize=28, fontweight='bold', y=0.98)
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        st.markdown("---")
                        st.info("""
                        **Defensive Block Analysis:**
                        - Heatmap: Density of defensive actions
                        - Large circles: Players with more defensive actions
                        - Circle size: Number of defensive actions
                        - Solid circles: Starters, Square: Substitutes
                        - White line: Average defensive line height (DAH)
                        - Shows team's defensive organization and pressure points
                        """)
        
        with tab2:
            st.subheader("🎨 Elegant Defensive Blocks")
            
            events_df = st.session_state.events_df
            teams = events_df['team'].dropna().unique()
            
            if len(teams) >= 2:
                team1, team2 = teams[0], teams[1]
                
                if st.button("Generate Elegant Defensive Blocks"):
                    with st.spinner("Creating elegant defensive visualization..."):
                        fig, axes = plt.subplots(1, 2, figsize=(22, 12), facecolor='#0e1111')
                        
                        plot_elegant_defensive_block(events_df, team1, axes[0], '#ff002e')
                        plot_elegant_defensive_block(events_df, team2, axes[1], '#00ff87')
                        
                        plt.suptitle("DEFENSIVE STRUCTURE & INTERVENTION DENSITY",
                                    color='white', fontsize=28, fontweight='bold', y=0.98)
                        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                        st.pyplot(fig)
                        
                        st.markdown("---")
                        st.info("""
                        **Elegant Defensive Blocks:**
                        - Enhanced visualization with visible pitch lines
                        - Smooth density gradients
                        - Clear player nodes with abbreviated names
                        - Defensive line height indicator
                        - Professional aesthetic for presentations
                        """)
        
        with tab3:
            st.subheader("👤 Individual Defensive Actions")
            
            events_df = st.session_state.events_df
            
            # Get defensive actions by player
            def_types = ['Ball Recovery', 'Interception', 'Tackle', 'Clearance', 'Block', 'Foul Committed']
            defensive_actions = events_df[events_df['type'].isin(def_types)].copy()
            
            if len(defensive_actions) > 0:
                teams = defensive_actions['team'].dropna().unique()
                
                if len(teams) >= 2:
                    selected_team = st.selectbox("Select team for defensive stats:", teams[:2], key='def_stats')
                    
                    if selected_team:
                        team_def = defensive_actions[defensive_actions['team'] == selected_team]
                        
                        # Count defensive actions by player
                        def_by_player = team_def['player'].value_counts().head(10)
                        
                        # Display as bar chart
                        fig, ax = plt.subplots(figsize=(12, 6))
                        def_by_player.plot(kind='barh', ax=ax, color='#e30613' if selected_team == teams[0] else '#10ac84')
                        ax.set_xlabel('Number of Defensive Actions')
                        ax.set_title(f'Top 10 Defensive Players - {selected_team}')
                        ax.invert_yaxis()  # Highest at top
                        st.pyplot(fig)
                        
                        # Detailed statistics
                        st.markdown("---")
                        st.subheader("📊 Defensive Action Types")
                        
                        # Count by type
                        def_by_type = team_def['type'].value_counts()
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**By Action Type:**")
                            for action_type, count in def_by_type.items():
                                st.write(f"- {action_type}: {count}")
                        
                        with col2:
                            # Success rate for tackles
                            if 'Tackle' in team_def['type'].values:
                                tackles = team_def[team_def['type'] == 'Tackle']
                                successful_tackles = tackles[tackles['tackle_outcome'].isin(['Won', 'Success'])]
                                tackle_success = len(successful_tackles) / len(tackles) * 100 if len(tackles) > 0 else 0
                                st.metric("Tackle Success Rate", f"{tackle_success:.1f}%")
            else:
                st.info("No defensive actions recorded in this match.")

# Page 13: NEW - Danger Zones Analysis
elif page == "🎪 13. Danger Zones Analysis (NEW)":
    if st.session_state.events_df is None:
        st.warning("⚠️ Please first load and analyze a match in step 3.")
    else:
        st.header("🎪 Danger Zones Analysis")
        
        tab1, tab2, tab3 = st.tabs([
            "⚠️ Dangerous Passes",
            "🎯 Pass End Zones",
            "🧩 Convex Hull Analysis"
        ])
        
        with tab1:
            st.subheader("⚠️ Dangerous Passes Analysis")
            
            events_df = st.session_state.events_df
            teams = events_df['team'].dropna().unique()
            
            if len(teams) >= 2:
                team1, team2 = teams[0], teams[1]
                
                if st.button("Generate Dangerous Passes Analysis"):
                    with st.spinner("Creating dangerous passes visualization..."):
                        fig, axes = plt.subplots(1, 2, figsize=(22, 12), facecolor='#0e1111')
                        
                        # Get successful passes for each team
                        hp = get_successful_passes(events_df, team1)
                        ap = get_successful_passes(events_df, team2)
                        
                        draw_danger_passes(axes[0], hp, team1, '#ff002e')
                        draw_danger_passes(axes[1], ap, team2, '#00ff87')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        st.markdown("---")
                        st.info("""
                        **Dangerous Passes Analysis:**
                        - Orange arrows: Passes into Zone 14 (central attacking zone)
                        - Team-colored arrows: Passes into Half-Spaces
                        - Hexagons show counts for each dangerous zone
                        - Shows teams' ability to create danger through key passes
                        - Zone 14 (Z14): Central area just outside box
                        - Half-Spaces (HS): Channels between center and wings
                        """)
        
        with tab2:
            st.subheader("🎯 Pass End Zone Distribution")
            
            events_df = st.session_state.events_df
            teams = events_df['team'].dropna().unique()
            
            if len(teams) >= 2:
                team1, team2 = teams[0], teams[1]
                
                if st.button("Generate Pass End Zones"):
                    with st.spinner("Creating pass end zone visualization..."):
                        fig, axes = plt.subplots(1, 2, figsize=(24, 12), facecolor='#0e1111')
                        
                        df_h = get_pass_end_data(events_df, team1)
                        df_a = get_pass_end_data(events_df, team2)
                        
                        plot_pass_end_zone(axes[0], df_h, team1, '#ff002e')
                        plot_pass_end_zone(axes[1], df_a, team2, '#00ff87')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        st.markdown("---")
                        st.info("""
                        **Pass End Zone Distribution:**
                        - Shows where teams complete their passes
                        - Heatmap intensity: Percentage of passes ending in each zone
                        - White dots: Individual pass endpoints
                        - Percentages show distribution across tactical zones
                        - Reveals teams' attacking preferences and penetration areas
                        """)
        
        with tab3:
            st.subheader("🧩 Convex Hull Analysis")
            
            events_df = st.session_state.events_df
            teams = events_df['team'].dropna().unique()
            
            if len(teams) >= 2:
                tab_convex1, tab_convex2, tab_convex3 = st.tabs([
                    "👤 Individual Player Hulls",
                    "👥 Team Hulls Grid",
                    "📊 Hull Statistics"
                ])
                
                with tab_convex1:
                    st.write("**Individual Player Influence Zones**")
                    
                    selected_team = st.selectbox("Select team:", teams[:2], key='hull_team')
                    
                    if selected_team:
                        # Get top players from the team
                        team_players = events_df[events_df['team'] == selected_team]['player'].value_counts().head(5).index.tolist()
                        
                        selected_player = st.selectbox("Select player:", team_players)
                        
                        if selected_player:
                            if st.button(f"Generate {selected_player} Convex Hull"):
                                with st.spinner("Creating convex hull..."):
                                    fig, ax = plt.subplots(figsize=(12, 8), facecolor='#0e1111')
                                    
                                    color = '#e30613' if selected_team == teams[0] else '#10ac84'
                                    plot_player_hull(events_df, selected_player, selected_team, ax, color)
                                    
                                    plt.tight_layout()
                                    st.pyplot(fig)
                
                with tab_convex2:
                    st.write("**Team Influence Zones Grid**")
                    
                    selected_team = st.selectbox("Select team for grid:", teams[:2], key='hull_grid')
                    
                    if selected_team:
                        if st.button(f"Generate {selected_team} Hull Grid"):
                            with st.spinner("Creating team hull grid..."):
                                color = '#e30613' if selected_team == teams[0] else '#10ac84'
                                fig = plot_team_hulls(events_df, selected_team, color)
                                st.pyplot(fig)
                
                with tab_convex3:
                    st.write("**Convex Hull Statistics**")
                    
                    st.info("""
                    **Convex Hull Analysis Explained:**
                    
                    **What is a Convex Hull?**
                    - The smallest convex polygon that contains all of a player's actions
                    - Shows the player's "zone of influence" on the pitch
                    
                    **What it reveals:**
                    - **Size**: Larger hull = more movement/versatility
                    - **Shape**: Horizontal = wide role, Vertical = box-to-box
                    - **Position**: Shows preferred areas of the pitch
                    - **Density**: Point concentration shows main action zones
                    
                    **Interpretation:**
                    - Compact hull: Specialized role, stays in specific area
                    - Large hull: Free role, covers more ground
                    - Forward-positioned: Attacking focus
                    - Centered: Playmaker/central role
                    - Wide: Winger/full-back role
                    """)

# Sidebar with information
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Current Data")

if st.session_state.selected_match_id:
    st.sidebar.success(f"**Current match:** {st.session_state.selected_match_id}")
if st.session_state.selected_competition_id:
    st.sidebar.info(f"**Competition:** {st.session_state.selected_competition_id}")
if st.session_state.events_df is not None:
    st.sidebar.info(f"**Events loaded:** {len(st.session_state.events_df)}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Instructions")
st.sidebar.info("""
1. **Step 1:** Load and select a competition
2. **Step 2:** Choose a specific match
3. **Step 3:** Run basic analysis
4. **Step 4-13:** Explore advanced visualizations

**⚠️ Note:** Data is preserved between steps.
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>⚽ Complete Football Analysis Platform by Oussama Boutrouft (@Oussama_Boutrouft)</strong></p>
    <p>Using free StatsBomb data • All notebook codes integrated • Advanced features included</p>
</div>
""", unsafe_allow_html=True)
