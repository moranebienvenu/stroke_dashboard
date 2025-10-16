import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table, ALL, MATCH
from flask import request, jsonify, make_response
import json
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import io
import zipfile
import re
import base64
import json
from scipy.stats import ttest_ind, ttest_rel, levene, mannwhitneyu, shapiro, wilcoxon, norm
import colorsys
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Tweedie, Gaussian, Gamma, Poisson
from statsmodels.genmod.families.links import Log, Identity, InversePower, Sqrt
from statsmodels.stats.power import TTestIndPower, TTestPower
from itertools import combinations
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
from io import BytesIO
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from dash.dash_table import FormatTemplate

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server

#---------------------------- Communication with a notebook --------------------------------

# Store global pour maintenir l'état des données
_global_data = {
    'dataset1': None,
    'dataset2': None,
    'master_data': None,
    'current_plots': None
}

# Chargement initial des données master
def load_master_data_global():
    """Charge les données master au démarrage"""
    #path = "/Users/bienvenumorane/Documents/Doctorat/AVC_stats/modules/App/data/data_article.csv"
    path="data/data_article.csv"
    try:
        df = pd.read_csv(path)
        _global_data['master_data'] = df.to_dict('records')
        return True
    except Exception as e:
        print(f"Error loading master data: {e}")
        return False

# Charger les données au démarrage
load_master_data_global()

@server.route('/api/health', methods=['GET'])
def health_check():
    """Endpoint de santé pour vérifier que l'API fonctionne"""
    return jsonify({
        "status": "healthy",
        "available_datasets": {
            "master": _global_data['master_data'] is not None,
            "dataset1": _global_data['dataset1'] is not None,
            "dataset2": _global_data['dataset2'] is not None
        }
    })

@server.route('/api/subjects', methods=['GET'])
def get_available_subjects():
    """Retourne la liste des sujets disponibles"""
    dataset = request.args.get('dataset', 'master')
    
    if dataset == 'master':
        data = _global_data['master_data']
    elif dataset == 'dataset1':
        data = _global_data['dataset1']
    elif dataset == 'dataset2':
        data = _global_data['dataset2']
    else:
        return jsonify({"error": "Invalid dataset"}), 400
    
    if not data:
        return jsonify({"error": "Dataset not available"}), 404
    
    df = pd.DataFrame(data)
    subjects_info = []
    
    for subject in df['subject'].unique():
        group = detect_group(subject)
        session_match = re.search(r"ses-(V[0-9]+)", subject)
        session = session_match.group(1) if session_match else "Unknown"
        sex = df[df['subject'] == subject]['sex'].iloc[0] if 'sex' in df.columns else "Unknown"
        
        subjects_info.append({
            "subject": subject,
            "group": group,
            "session": session,
            "sex": sex
        })
    
    return jsonify({
        "subjects": subjects_info,
        "total_count": len(subjects_info)
    })


# ---------------------------- API Base Graphs Endpoints ----------------------------

@server.route('/api/generate_plots', methods=['POST'])
def api_generate_plots():
    """Génère des graphiques via API"""
    try:
        # Récupérer les paramètres de la requête
        params = request.get_json()
        
        # Paramètres par défaut
        dataset = params.get('dataset', 'master')
        analysis_type = params.get('analysis_type', 'session_sex')
        session = params.get('session', 'V1')
        sex_filter = params.get('sex_filter', 'men')
        selected_groups = params.get('groups', [])
        selected_subject = params.get('subject', None)
        
        # Validation des paramètres
        if dataset not in ['master', 'dataset1', 'dataset2']:
            return jsonify({"error": "Invalid dataset"}), 400
        
        # if analysis_type not in ['single', 'session_sex']:
        #     return jsonify({"error": "Invalid analysis_type"}), 400
        
        # Sélectionner les données
        if dataset == 'master':
            data = _global_data['master_data']
        elif dataset == 'dataset1':
            data = _global_data['dataset1']
        elif dataset == 'dataset2':
            data = _global_data['dataset2']
        
        if not data:
            return jsonify({"error": f"Dataset {dataset} not available"}), 404
        
        df = pd.DataFrame(data)
        
        # Logique de génération des graphiques
        if analysis_type == 'single':
            if not selected_subject:
                return jsonify({"error": "Subject required for single analysis"}), 400
            
            if selected_subject not in df['subject'].values:
                return jsonify({"error": "Subject not found in dataset"}), 404
            
            subjects = [selected_subject]
            title = f"{selected_subject}"
            is_group = False
            
        elif analysis_type == 'session_sex':
            if not session:
                return jsonify({"error": "Session required for session_sex analysis"}), 400
            
            # Filtrer par session
            session_subjects = df[df['subject'].str.contains(f"_ses-{session}")]['subject'].tolist()
            
            # Filtrer par sexe
            if sex_filter != 'all' and 'sex' in df.columns:
                gender = "M" if sex_filter == 'men' else "F"
                session_subjects = df[
                    (df['subject'].isin(session_subjects)) & 
                    (df['sex'] == gender)
                ]['subject'].tolist()
            
            # Filtrer par groupes
            if selected_groups:
                subjects = [s for s in session_subjects if detect_group(s) in selected_groups]
            else:
                subjects = session_subjects
            
            if not subjects:
                return jsonify({"error": "No subjects found for these criteria"}), 404
            
            # Construire le titre
            if 'title' in params and params['title']:
                title = params['title']
            else:
                title = f"Session {session}"
                if sex_filter != 'all':
                    title += f" ({'Men' if sex_filter == 'men' else 'Women'})"
                if selected_groups:
                    title += f" | Groups: {', '.join(selected_groups)}"
            
            is_group = True
        
        # Générer les graphiques
        fig1, fig2, fig3 = create_interactive_plots(df, subjects, title, is_group=is_group)
        
        # Sauvegarder dans le store global
        plots_data = {
            'fig1': fig1.to_dict(),
            'fig2': fig2.to_dict(),
            'fig3': fig3.to_dict(),
            'subjects': subjects,
            'title': title,
            'metadata': {
                'dataset': dataset,
                'analysis_type': analysis_type,
                'session': session,
                'sex_filter': sex_filter,
                'groups': selected_groups,
                'subject_count': len(subjects)
            }
        }
        
        _global_data['current_plots'] = plots_data
        
        return jsonify({
            "status": "success",
            "plots": plots_data,
            #"message": f"Generated plots for {len(subjects)} subjects"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@server.route('/api/get_plots', methods=['GET'])
def api_get_current_plots():
    """Retourne les graphiques actuels"""
    if not _global_data['current_plots']:
        return jsonify({"error": "No plots available. Generate plots first."}), 404
    
    return jsonify({
        "status": "success",
        "plots": _global_data['current_plots']
    })

@server.route('/api/update_plots', methods=['PUT'])
def api_update_plots():
    """Met à jour les graphiques avec de nouveaux paramètres"""
    try:
        params = request.get_json()
        
        # Utiliser les paramètres actuels comme base
        if _global_data['current_plots'] and 'metadata' in _global_data['current_plots']:
            current_metadata = _global_data['current_plots']['metadata']
            
            # Mettre à jour seulement les paramètres fournis
            dataset = params.get('dataset', current_metadata.get('dataset', 'master'))
            analysis_type = params.get('analysis_type', current_metadata.get('analysis_type', 'session_sex'))
            session = params.get('session', current_metadata.get('session', 'V1'))
            sex_filter = params.get('sex_filter', current_metadata.get('sex_filter', 'all'))
            selected_groups = params.get('groups', current_metadata.get('groups', []))
            selected_subject = params.get('subject', None)
        else:
            # Utiliser les valeurs par défaut si aucun plot actuel
            return jsonify({"error": "No current plots to update. Use generate_plots first."}), 400
        
        # Réutiliser la logique de génération
        return api_generate_plots()
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@server.route('/api/export_plots', methods=['GET'])
def api_export_plots():
    """Exporte les graphiques dans différents formats"""
    if not _global_data['current_plots']:
        return jsonify({"error": "No plots available"}), 404
    
    export_format = request.args.get('format', 'json')
    
    if export_format == 'json':
        return jsonify(_global_data['current_plots'])
    
    elif export_format == 'plotly':
        # Retourne les objets Plotly directement utilisables
        plots = _global_data['current_plots']
        return jsonify({
            "fig1": plots['fig1'],
            "fig2": plots['fig2'], 
            "fig3": plots['fig3'],
            "metadata": plots['metadata']
        })
    
    else:
        return jsonify({"error": "Unsupported format. Use 'json' or 'plotly'"}), 400

# ---------------------------- API Overlay Endpoints ----------------------------

@server.route('/api/overlay/generate', methods=['POST'])
def api_generate_overlay():
    """Génère un overlay via API avec état naturel"""
    try:
        params = request.get_json()
        
        # Paramètres par défaut pour l'état naturel
        dataset = params.get('dataset', 'master')
        analysis_type = params.get('analysis_type', 'session_sex')  # État naturel
        session = params.get('session', 'V1')  # État naturel
        sex_filter = params.get('sex_filter', 'women')  # État naturel
        selected_groups = params.get('groups', [])
        selected_subject = params.get('subject', None)
        overlay_title = params.get('title', 'Overlay')
        
        # Validation
        if dataset not in ['master', 'dataset1', 'dataset2']:
            return jsonify({"error": "Invalid dataset"}), 400
        
        # Sélectionner les données
        if dataset == 'master':
            data = _global_data['master_data']
        elif dataset == 'dataset1':
            data = _global_data['dataset1']
        elif dataset == 'dataset2':
            data = _global_data['dataset2']
        
        if not data:
            return jsonify({"error": f"Dataset {dataset} not available"}), 404
        
        df = pd.DataFrame(data)
        
        # Logique de génération des sujets overlay
        if analysis_type == 'single':
            if not selected_subject:
                return jsonify({"error": "Subject required for single analysis"}), 400
            subjects = [selected_subject]
            is_group = False
            
        elif analysis_type == 'session_sex':
            if not session:
                return jsonify({"error": "Session required"}), 400
            
            # Filtrer par session
            session_subjects = df[df['subject'].str.contains(f"_ses-{session}")]['subject'].tolist()
            
            # Filtrer par sexe
            if sex_filter != 'all' and 'sex' in df.columns:
                gender = "M" if sex_filter == 'men' else "F"
                session_subjects = df[
                    (df['subject'].isin(session_subjects)) & 
                    (df['sex'] == gender)
                ]['subject'].tolist()
            
            # Filtrer par groupes
            if selected_groups:
                subjects = [s for s in session_subjects if detect_group(s) in selected_groups]
            else:
                subjects = session_subjects
            
            if not subjects:
                return jsonify({"error": "No subjects found"}), 404
            is_group = True
        
        # Générer les graphiques overlay
        fig1, fig2, fig3 = create_interactive_plots(df, subjects, overlay_title, is_group=is_group, is_overlay=True)
        
        overlay_data = {
            'fig1': fig1.to_dict(),
            'fig2': fig2.to_dict(),
            'fig3': fig3.to_dict(),
            'subjects': subjects,
            'title': overlay_title,
            'metadata': {
                'dataset': dataset,
                'analysis_type': analysis_type,
                'session': session,
                'sex_filter': sex_filter,
                'groups': selected_groups,
                'subject_count': len(subjects)
            }
        }
        
        # Stocker l'overlay
        if 'overlays' not in _global_data:
            _global_data['overlays'] = []
        _global_data['overlays'].append(overlay_data)
        
        return jsonify({
            "status": "success",
            "overlay": overlay_data,
            "overlay_count": len(_global_data['overlays']),
            #"message": f"Overlay generated for {len(subjects)} subjects"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@server.route('/api/overlay/list', methods=['GET'])
def api_list_overlays():
    """Liste tous les overlays actuels"""
    overlays = _global_data.get('overlays', [])
    return jsonify({
        "status": "success",
        "overlays": overlays,
        "count": len(overlays)
    })

@server.route('/api/overlay/clear', methods=['DELETE'])
def api_clear_overlays():
    """Efface tous les overlays"""
    _global_data['overlays'] = []
    return jsonify({
        "status": "success",
        "message": "All overlays cleared"
    })

@server.route('/api/overlay/combine', methods=['GET'])
def api_get_combined_plots():
    """Retourne les graphiques combinés (base + tous les overlays)"""
    base_plots = _global_data.get('current_plots')
    overlays = _global_data.get('overlays', [])
    
    if not base_plots:
        return jsonify({"error": "No base plots available"}), 404
    
    try:
        # Commencer avec les graphiques de base
        fig1_combined = go.Figure(base_plots['fig1'])
        fig2_combined = go.Figure(base_plots['fig2'])
        fig3_combined = go.Figure(base_plots['fig3'])
        
        # Ajouter chaque overlay
        for i, overlay in enumerate(overlays):
            for trace in go.Figure(overlay['fig1']).data:
                fig1_combined.add_trace(trace)
            for trace in go.Figure(overlay['fig2']).data:
                fig2_combined.add_trace(trace)
            for trace in go.Figure(overlay['fig3']).data:
                fig3_combined.add_trace(trace)
        
        # Configurer l'affichage superposé
        fig1_combined.update_layout(barmode='overlay', showlegend=True)
        fig2_combined.update_layout(barmode='overlay', showlegend=True)
        fig3_combined.update_layout(barmode='overlay', showlegend=True)
        
        combined_data = {
            'fig1': fig1_combined.to_dict(),
            'fig2': fig2_combined.to_dict(),
            'fig3': fig3_combined.to_dict(),
            'base_title': base_plots.get('title', 'Base Plot'),
            'overlay_count': len(overlays),
            'overlay_titles': [ov.get('title', f'Overlay {i+1}') for i, ov in enumerate(overlays)]
        }
        
        return jsonify({
            "status": "success",
            "combined_plots": combined_data
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Callback pour synchroniser les plots API avec Dash -- Modification pour communication notebook
@app.callback(
    Output('plots-store', 'data', allow_duplicate=True),
    Input('tabs', 'active_tab'),
    prevent_initial_call=True
)
def sync_api_plots_to_dash(active_tab):
    """Synchronise les plots générés via API avec l'interface Dash"""
    if active_tab == "tab-visualization" and _global_data['current_plots']:
        return _global_data['current_plots']
    raise dash.exceptions.PreventUpdate

# Callback pour synchroniser les overlays avec Dash
@app.callback(
    Output('overlay-store', 'data'),
    Input('overlay-generate-btn', 'n_clicks'),
    prevent_initial_call=True
)
def sync_overlay_to_dash(n_clicks):
    """Synchronise les overlays générés avec l'interface Dash"""
    if n_clicks and _global_data.get('overlays'):
        return _global_data['overlays'][-1]  # Retourne le dernier overlay
    raise dash.exceptions.PreventUpdate


# ---------------------------- API Correlation Endpoints ----------------------------

@server.route('/api/correlation/generate_heatmaps', methods=['POST'])
def api_generate_correlation_heatmaps():
    """Génère des heatmaps de corrélation via API en utilisant les mêmes fonctions que Dash"""
    try:
        params = request.get_json()
        
        # Paramètres pour les heatmaps
        dataset = params.get('dataset', 'master')
        session = params.get('session', 'V1')
        system_type = params.get('system_type', 'Synaptic ratio')
        groups = params.get('groups', ['A'])
        
        # Validation des paramètres
        if dataset not in ['master', 'dataset1', 'dataset2']:
            return jsonify({"error": "Invalid dataset"}), 400
        
        # Sélectionner les données
        if dataset == 'master':
            data = _global_data['master_data']
        elif dataset == 'dataset1':
            data = _global_data['dataset1']
        elif dataset == 'dataset2':
            data = _global_data['dataset2']
        
        if not data:
            return jsonify({"error": f"Dataset {dataset} not available"}), 404
        
        df = pd.DataFrame(data)
        
        # Définir les variables pour les ratios synaptiques 
        # synaptic_vars = [
        #     "pre_A4B2", "pre_M1", "pre_D1", "pre_D2",
        #     "pre_5HT1a", "pre_5HT1b", "pre_5HT2a",
        #     "pre_5HT4", "pre_5HT6",
        #     "post_VAChT", "post_DAT", "post_5HTT"
        # ]
        
        outcome_vars_map = {
            "Synaptic ratio": [
                "pre_A4B2", "pre_M1", "pre_D1", "pre_D2",
                "pre_5HT1a", "pre_5HT1b", "pre_5HT2a",
                "pre_5HT4", "pre_5HT6",
                "post_VAChT", "post_DAT", "post_5HTT"
            ],
            "Neurotransmitter (Loc)": [f"loc_inj_{sys}" for sys in [
                "GABAa", "mGluR5", "MU", "H3", "CB1", "A4B2", "M1", "VAChT",
                "D1", "D2", "DAT", "Nor", "5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT"
            ] if f"loc_inj_{sys}" in df.columns],
            "Neurotransmitter (Tract)": [f"tract_inj_{sys}" for sys in [
                "GABAa", "mGluR5", "MU", "H3", "CB1", "A4B2", "M1", "VAChT",
                "D1", "D2", "DAT", "Nor", "5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT"
            ] if f"tract_inj_{sys}" in df.columns],
            "Clinical Outcomes": [col for col in df.columns 
                        if col not in ['subject', 'Sexe_bin', 'sex', 'lesion_volume']
                        and not col.startswith(('loc_inj_', 'tract_inj_', 'pre_', 'post_'))]
        }
        
        # CORRECTION : Sélectionner les variables selon system_type
        available_vars = [col for col in outcome_vars_map.get(system_type, []) if col in df.columns]
        
        if not available_vars:
            return jsonify({"error": f"No variables found for {system_type}"}), 404
       
        
        # # Filtrer les variables disponibles
        # available_vars = [col for col in synaptic_vars if col in df.columns]
        
        # if not available_vars:
        #     return jsonify({"error": "No synaptic ratio variables found"}), 404
        
        # Générer les heatmaps pour All, Men only, Women only
        heatmaps_data = {}
        
        for sex_filter in ['all', 'men', 'women']:
            # Obtenir les sujets pour ce filtre 
            subjects, _, _, _ = get_subjects_by_criteria(
                df,
                analysis_type="By session and sex",
                session=session,
                sex_filter="Men only" if sex_filter == "men" else "Women only" if sex_filter == "women" else "All",
                groups=groups
            )
            
            if len(subjects) < 3:
                heatmaps_data[sex_filter] = {
                    "status": "insufficient_data",
                    "message": f"Only {len(subjects)} subjects found"
                }
                continue
            
            # Filtrer les données
            df_filtered = df[df['subject'].isin(subjects)].copy()
            df_numeric = df_filtered[available_vars].select_dtypes(include=[np.number])
            
           
            corr_matrix, pval_matrix = get_correlation_matrix(df_numeric, include_sex_bin=False)
            
            # Créer la heatmap avec la même logique que le callback Dash
            fig = create_dash_style_heatmap(corr_matrix, pval_matrix, f"Session {session} - {sex_filter.title()}")
            
            heatmaps_data[sex_filter] = {
                "status": "success",
                "correlation_matrix": corr_matrix.round(4).to_dict(),
                "pvalue_matrix": pval_matrix.round(4).to_dict(),
                "heatmap": fig.to_dict(),
                "subject_count": len(subjects),
                "variables": available_vars,
                "corr_index": corr_matrix.index.tolist(),
                "corr_columns": corr_matrix.columns.tolist()
            }
        
        # Sauvegarder dans le store global
        _global_data['correlation_heatmaps'] = heatmaps_data
        
        return jsonify({
            "status": "success",
            "heatmaps": heatmaps_data,
            "metadata": {
                'dataset': dataset,
                'session': session,
                'system_type': system_type,
                'groups': groups
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def create_dash_style_heatmap(corr_matrix, pval_matrix, title):
    """Crée une heatmap identique à celle du Dashboard"""
    # Préparer les données comme dans le callback Dash
    mask = pval_matrix > 0.05
    corr_display = corr_matrix.where(~mask, None)
    
    # Créer la heatmap avec les mêmes paramètres 
    fig = go.Figure(data=go.Heatmap(
        z=corr_display.values,
        x=corr_display.columns.tolist(),
        y=corr_display.index.tolist(),
        colorscale='RdBu_r',
        zmin=-1,
        zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverinfo="text",
        hovertemplate=(
            "Variable X: %{x}<br>"
            "Variable Y: %{y}<br>"
            "Correlation: %{z:.3f}<br>"
            "<extra></extra>"
        )
    ))
    
    # Mise en page 
    fig.update_layout(
        title=dict(
            text=title,
            x=0.001,
            xanchor='center'
        ),
        xaxis=dict(
            tickangle=45,
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            autorange='reversed',
            tickfont=dict(size=10)
        ),
        height=300,
        width=300,
        #margin=dict(l=40, r=10, t=50, b=60),
        font=dict(size=10)
    )
    
    # Ajouter les formes pour les cases non significatives 
    shapes = []
    for i in range(len(corr_matrix.index)):
        for j in range(len(corr_matrix.columns)):
            if pval_matrix.iloc[i, j] >= 0.05:
                shapes.append(dict(
                    type="rect",
                    xref="x",
                    yref="y",
                    x0=j - 0.5,
                    x1=j + 0.5,
                    y0=i - 0.5,
                    y1=i + 0.5,
                    fillcolor="rgba(200,200,200,0.5)",
                    line_width=0,
                    layer="above"
                ))
    
    if shapes:
        fig.update_layout(shapes=shapes)
    
    return fig

#Tests correlation deux sets 


@server.route('/api/correlation/generate_cross_heatmaps', methods=['POST'])
def api_generate_cross_correlation_heatmaps():
    """Génère des heatmaps de corrélation croisée entre deux sets de sujets"""
    try:
        params = request.get_json()
        
        # Paramètres Set 1
        dataset = params.get('dataset', 'master')
        session1 = params.get('session1', 'V1')
        sex_filter1 = params.get('sex_filter1', 'All')
        outcome1 = params.get('outcome1', 'Synaptic ratio')
        groups1 = params.get('groups1', ['A'])
        
        # Paramètres Set 2
        session2 = params.get('session2', 'V1')
        sex_filter2 = params.get('sex_filter2', 'All')
        outcome2 = params.get('outcome2', 'Synaptic ratio')
        groups2 = params.get('groups2', ['A'])
        
        # Sélectionner les données
        if dataset == 'master':
            data = _global_data['master_data']
        elif dataset == 'dataset1':
            data = _global_data['dataset1']
        elif dataset == 'dataset2':
            data = _global_data['dataset2']
        
        if not data:
            return jsonify({"error": f"Dataset {dataset} not available"}), 404
        
        df = pd.DataFrame(data)
        
        # Définir les variables selon l'outcome
        outcome_vars_map = {
                "Synaptic ratio": [
                    "pre_A4B2", "pre_M1", "pre_D1", "pre_D2",
                    "pre_5HT1a", "pre_5HT1b", "pre_5HT2a",
                    "pre_5HT4", "pre_5HT6",
                    "post_VAChT", "post_DAT", "post_5HTT"
                ],
                "Neurotransmitter (Loc)": [f"loc_inj_{sys}" for sys in [
                    "GABAa", "mGluR5", "MU", "H3", "CB1", "A4B2", "M1", "VAChT",
                    "D1", "D2", "DAT", "Nor", "5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT"
                ] if f"loc_inj_{sys}" in df.columns],
                "Neurotransmitter (Tract)": [f"tract_inj_{sys}" for sys in [
                    "GABAa", "mGluR5", "MU", "H3", "CB1", "A4B2", "M1", "VAChT",
                    "D1", "D2", "DAT", "Nor", "5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT"
                ] if f"tract_inj_{sys}" in df.columns],
                "Clinical Outcomes":
                 [col for col in df.columns 
                            if col not in ['subject', 'Sexe_bin', 'sex', 'lesion_volume']
                            and not col.startswith(('loc_inj_', 'tract_inj_', 'pre_', 'post_'))]
            }
        vars1 = [col for col in outcome_vars_map.get(outcome1, []) if col in df.columns]
        vars2 = [col for col in outcome_vars_map.get(outcome2, []) if col in df.columns]
        
        if not vars1 or not vars2:
            return jsonify({"error": "No variables found for selected outcomes"}), 404
        
        # Obtenir les sujets pour chaque set
        subjects1, _, _, _ = get_subjects_by_criteria(
            df,
            analysis_type="By session and sex",
            session=session1,
            sex_filter=sex_filter1,
            groups=groups1
        )
        
        subjects2, _, _, _ = get_subjects_by_criteria(
            df,
            analysis_type="By session and sex",
            session=session2,
            sex_filter=sex_filter2,
            groups=groups2
        )
        
        if len(subjects1) < 3 or len(subjects2) < 3:
            return jsonify({
                "error": f"Insufficient subjects (Set1: {len(subjects1)}, Set2: {len(subjects2)})"
            }), 400
        
        # Préparer les données de corrélation croisée
        df_corr, suffix1, suffix2 = prepare_correlation_data_two_sets(
            df, subjects1, vars1, subjects2, vars2, session1, session2
        )
        
        # Calculer la matrice de corrélation
        corr_matrix, pval_matrix = get_correlation_matrix(df_corr, include_sex_bin=False)
        
        # Extraire les corrélations croisées
        set1_cols = [col for col in corr_matrix.columns if col.endswith(suffix1)]
        set2_cols = [col for col in corr_matrix.columns if col.endswith(suffix2)]
        
        cross_corr = corr_matrix.loc[set1_cols, set2_cols]
        cross_pvals = pval_matrix.loc[set1_cols, set2_cols]
        
        # Créer la heatmap
        mask = cross_pvals > 0.05
        corr_display = cross_corr.where(~mask, None)
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_display.values,
            x=[col.replace(suffix2, '') for col in cross_corr.columns],
            y=[idx.replace(suffix1, '') for idx in cross_corr.index],
            colorscale='RdBu_r',
            zmin=-1,
            zmax=1,
            text=np.round(cross_corr.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate=(
                "Set2: %{x}<br>"
                "Set1: %{y}<br>"
                "Correlation: %{z:.3f}<br>"
                "<extra></extra>"
            )
        ))
        
        fig.update_layout(
            title=dict(
                text=f"Cross Correlation: {session1} ({sex_filter1}) vs {session2} ({sex_filter2})",
                x=0.001,
                xanchor='center'
            ),
            xaxis=dict(
                title=f"Set 2: {outcome2} ({session2})",
                tickangle=45,
                tickfont=dict(size=10)
            ),
            yaxis=dict(
                title=f"Set 1: {outcome1} ({session1})",
                autorange='reversed',
                tickfont=dict(size=10)
            ),
            height=250,
            width=250,
            #margin=dict(l=120, r=50, t=100, b=120)
        )
        
        # Ajouter les formes pour les cases non significatives
        shapes = []
        for i in range(len(cross_corr.index)):
            for j in range(len(cross_corr.columns)):
                if cross_pvals.iloc[i, j] >= 0.05:
                    shapes.append(dict(
                        type="rect",
                        xref="x",
                        yref="y",
                        x0=j - 0.5,
                        x1=j + 0.5,
                        y0=i - 0.5,
                        y1=i + 0.5,
                        fillcolor="rgba(200,200,200,0.5)",
                        line_width=0,
                        layer="above"
                    ))
        
        if shapes:
            fig.update_layout(shapes=shapes)
        
        result = {
            "status": "success",
            "correlation_matrix": cross_corr.round(4).to_dict(),
            "pvalue_matrix": cross_pvals.round(4).to_dict(),
            "heatmap": fig.to_dict(),
            "subject_count_set1": len(subjects1),
            "subject_count_set2": len(subjects2),
            "common_subjects": len(df_corr),
            "variables_set1": vars1,
            "variables_set2": vars2
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
# ---------------------------- Used functions --------------------------------

def process_zip_to_dataframe(contents, filename):
    """Process uploaded ZIP file and return combined DataFrame"""
    if contents is None:
        return pd.DataFrame()
        
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    data_les_dis = {}
    data_pre_post = {}
    df_clinical = None
    missing_clinical_subjects = []
    systems_list = []

    with zipfile.ZipFile(io.BytesIO(decoded)) as z:
        for filename in z.namelist():
            # Skip system files
            if (
                "__MACOSX" in filename 
                or filename.endswith(".DS_Store") 
                or "/._" in filename 
                or (not "output_les_dis" in filename and not "output_pre_post_synaptic_ratio" in filename and "clinical_data" not in filename)
            ):
                continue

            if filename.endswith(".csv") or filename.endswith(".xlsx"):
                # Clinical data file
                if "clinical_data" in filename:
                    with z.open(filename) as f:
                        try:
                            if filename.endswith(".csv"):
                                content = f.read().decode("utf-8")
                                if "," in content and " " in content:
                                    df_clinical = pd.read_csv(io.StringIO(content), sep=" ", decimal=",")
                                else:
                                    df_clinical = pd.read_csv(io.StringIO(content))
                            else:
                                df_clinical = pd.read_excel(f, engine="openpyxl")
                        except Exception as e:
                            print(f"Error loading clinical file: {e}")
                    continue  

            # Extract subject ID
            if "output_les_dis" in filename or "output_pre_post_synaptic_ratio" in filename:
                match = re.search(r"sub-[A-Za-z0-9]+_ses-V[0-9]+", filename)
                if not match:
                    continue

            subject = match.group(0)

            with z.open(filename) as f:
                try:
                    df = pd.read_csv(f, sep=' ', index_col=0)
                except Exception as e:
                    continue

            # Classify by type
            if "output_les_dis" in filename:
                row = {"subject": subject}
                if not systems_list and len(df.columns) > 0:
                    systems_list = list(df.columns)
                for idx in df.index:
                    prefix = str(idx).split('_sub-')[0]
                    for system in df.columns:
                        colname = f"{prefix}_{system}"
                        row[colname] = df.at[idx, system]
                data_les_dis[subject] = pd.DataFrame([row])  

            elif "output_pre_post_synaptic_ratio" in filename:
                row = {"subject": subject}
                for col in df.columns:
                    m = re.match(r"(.+?)\s+(presynaptic|postsynaptic)", col)
                    if m:
                        system_name = m.group(1).strip()
                        syn_type = m.group(2).strip()
                        prefix = "pre" if syn_type == "presynaptic" else "post"
                        new_colname = f"{prefix}_{system_name}"
                        val = df[col].iloc[0] if not df.empty else None
                        row[new_colname] = val
                data_pre_post[subject] = pd.DataFrame([row])

    # Merge subject by subject
    combined_rows = []
    all_subjects = set(data_les_dis.keys()) | set(data_pre_post.keys()) 

    for subject in sorted(all_subjects):
        row = {"subject": subject}

        if subject in data_les_dis:
            try:
                row.update(data_les_dis[subject].iloc[0].to_dict())
            except Exception as e:
                pass

        if subject in data_pre_post:
            try:
                row.update(data_pre_post[subject].iloc[0].to_dict())
            except Exception as e:
                pass

        # Add clinical data if available
        if df_clinical is not None:
            match_row = df_clinical[df_clinical['subject'] == subject]
            if not match_row.empty:
                row.update(match_row.iloc[0].to_dict())
            else:
                missing_clinical_subjects.append(subject)

        combined_rows.append(row)

    # Create final DataFrame
    final_df = pd.DataFrame(combined_rows)
    
    # Reorganize columns for logical order
    if systems_list:
        ordered_columns = ['subject']
        
        for system in systems_list:
            for measure in ['loc_inj', 'loc_inj_perc', 'tract_inj', 'tract_inj_perc']:
                colname = f"{measure}_{system}"
                if colname in final_df.columns:
                    ordered_columns.append(colname)
        
        pre_post_columns = [col for col in final_df.columns if col.startswith(('pre_', 'post_')) and col not in ordered_columns]
        ordered_columns.extend(sorted(pre_post_columns))
        
        other_columns = [col for col in final_df.columns if col not in ordered_columns and col != 'subject']
        ordered_columns.extend(other_columns)
        
        final_df = final_df[ordered_columns]

    return final_df

def detect_group(subject_id):
    """Detect subject group from ID"""
    if "_sub-NA" in subject_id or "-NA" in subject_id:
        return "NA"
    elif "_sub-AN" in subject_id or "-AN" in subject_id: 
        return "AN"
    elif "_sub-A" in subject_id or "-A" in subject_id:
        return "A"
    elif "_sub-C" in subject_id or "-C" in subject_id: 
        return "C"
    elif "_sub-B" in subject_id or "-B" in subject_id: 
        return "B"
    elif "_sub-W" in subject_id or "-W" in subject_id: 
        return "W"
    elif "_sub-G" in subject_id or "-G" in subject_id: 
        return "G"
    elif "_sub-TCM" in subject_id or "-TCM" in subject_id: 
        return "TCM"
    elif "_sub-TCS" in subject_id or "-TCS" in subject_id: 
        return "TCS"
    elif "_sub-TCMix" in subject_id or "-TCMix" in subject_id: 
        return "TCMix"
    else:
        return "Unknown"

def create_interactive_plots(df, subjects, title_suffix="", is_group=False, is_overlay=False):
    # Filtrer les données et calculer les moyennes que si groupe sinon mettre les données individuelles pour chaque sujet base/overlay
    plot_data = df[df['subject'].isin(subjects)]
    
    # Détection dynamique des systèmes
    systems = [col.replace('loc_inj_', '') for col in df.columns 
               if col.startswith('loc_inj_') and not col.startswith('loc_inj_perc')]
    
    # Colonnes d'intérêt pour la moyenne
    loc_cols = [f'loc_inj_perc_{sys}' for sys in systems if f'loc_inj_perc_{sys}' in df.columns]
    tract_cols = [f'tract_inj_perc_{sys}' for sys in systems if f'tract_inj_perc_{sys}' in df.columns]
    pre_systems = ['A4B2', 'M1', 'D1', 'D2', '5HT1a', '5HT1b', '5HT2a', '5HT4', '5HT6']
    pre_cols = [f'pre_{sys}' for sys in pre_systems if f'pre_{sys}' in df.columns]
    post_systems = ['VAChT', 'DAT', '5HTT']
    post_cols = [f'post_{sys}' for sys in post_systems if f'post_{sys}' in df.columns]
    
    if is_group or len(subjects) > 1:
        # Cas groupe : on calcule la moyenne
        mean_values = {}
        for col in loc_cols:
            mean_values[col] = plot_data[col].mean()
        for col in tract_cols:
            mean_values[col] = plot_data[col].mean()
        for col in pre_cols:
            mean_values[col] = plot_data[col].mean()
        for col in post_cols:
            mean_values[col] = plot_data[col].mean()
        data_to_plot = pd.Series(mean_values)
    else:
        # Cas sujet unique : on prend les données brutes
        data_to_plot = plot_data.iloc[0]  

        
    # 1. Préparation des données pour les graphiques 1 et 2
    loc_inj_perc = [data_to_plot[f'loc_inj_perc_{sys}'] for sys in systems]
    tract_inj_perc = [data_to_plot[f'tract_inj_perc_{sys}'] for sys in systems]

    # 2. Préparation des données pour le graphiques 3
    pre_cols_used = [data_to_plot[f'pre_{sys}'] for sys in pre_systems ]
    post_cols_used = [data_to_plot[f'post_{sys}'] for sys in post_systems]
    
    # Gestion des cas où certaines colonnes pourraient manquer
    radii3_log = pre_cols_used + post_cols_used

    # Configuration des couleurs
    if is_overlay:
        # Pour les overlays, utiliser une couleur différente
        overlay_color = f"hsla({hash(title_suffix) % 360}, 80%, 50%, 0.5)"
        colors1 = [overlay_color] * len(systems)
        colors3 = [overlay_color if np.exp(val) > 1 else overlay_color.replace("0.5", "0.2") for val in radii3_log]
        #colors3 = [overlay_color] * len(radii3_log)
    else:
        # Couleurs par défaut
        fixed_color_strong = 'lightskyblue'
        fixed_color_light = 'rgba(135, 206, 250, 0.3)'
        colors1 = [fixed_color_strong] * len(systems)
        #colors3 = [fixed_color_strong] * len(radii3_log)
        colors3 = [fixed_color_strong if np.exp(val) > 1 else fixed_color_light for val in radii3_log]

    # Configuration commune
    config = {
        'height': 400,
        'width': 500,
        'margin': dict(l=40, r=40, t=40, b=40),
        'legend': dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        ),
        'title_x': 0.5,
        'title_font_size': 12,
        'polar': {
            'angularaxis': {
                'direction': 'clockwise',
                'rotation': 90,
            },
            'bargap': 0.1  
        }
    }

    base_val = 0 if is_overlay else None
    
    # Graphique 1: Lésions
    fig1 = go.Figure()
    fig1.add_trace(go.Barpolar(
        r=loc_inj_perc,
        theta=systems,
        marker_color=colors1[:len(systems)],
        name=title_suffix,
        hovertemplate='<b>%{theta}</b><br>%{r:.2f}%<extra></extra>',
        width=np.pi/4,
        base=base_val
    ))
    #{title_suffix}
    fig1.update_layout(
        title_text=f'<b>Receptor/transporter lesion</b>', 
        polar_radialaxis_ticksuffix='%',
        #height=500,
        showlegend=True,
        #margin=dict(l=60, r=20, t=80, b=40),
        #font=dict(size=12),
        **config
    )

    # Graphique 2: Déconnexions
    fig2 = go.Figure()
    fig2.add_trace(go.Barpolar(
        r=tract_inj_perc,
        theta=systems,
        marker_color=colors1[:len(systems)],
        name=title_suffix,
        hovertemplate='<b>%{theta}</b><br>%{r:.2f}%<extra></extra>',
        width=np.pi/4,
        base=base_val 
    ))
    fig2.update_layout(
        title_text=f'<b>Receptor/transporter disconnection </b>',
        polar_radialaxis_ticksuffix='%',
        #height=400,
        showlegend=True,
        #margin=dict(l=50, r=0, t=50, b=10),
        #font=dict(size=12),
        **config
    )
 
    # Graphique 3: Ratios
    fig3 = go.Figure()
    fig3.add_trace(go.Barpolar(
        r=radii3_log,
        theta=[f"pre {sys}" for sys in pre_systems ] + 
              [f"post {sys}" for sys in post_systems ],
        marker_color=colors3,
        name=title_suffix,
        hovertemplate='<b>%{theta}</b><br>%{r:.2f}<extra></extra>',
        width=np.pi/4,
        base=base_val
    ))
    fig3.update_layout(
        title_text=f'<b>Pre/post synaptic ratios </b>',
        #polar_radialaxis_range=[min(radii3_log) - 0.1, max(radii3_log) + 0.1],
        polar_radialaxis_range=[-1, 1],
        #height=400,  
        showlegend=True,
        #font=dict(size=12),
        **config
    )

    return fig1, fig2, fig3

def get_subjects_by_criteria(df, analysis_type, session=None, sex_filter="All", groups=None, existing_subjects=None):
    """Get subjects based on analysis criteria"""
    if existing_subjects is None:
        existing_subjects = []
    
    if analysis_type == "Single subject":
        available_subjects = [s for s in df['subject'].unique() if s not in existing_subjects]
        return available_subjects, available_subjects[0] if available_subjects else None, None, None
    
    elif analysis_type == "By session and sex":
        if not session:
            return [], None, None, None
            
        # Filter by session
        session_subjects = df[df['subject'].str.contains(f"_ses-{session}")]['subject'].tolist()
        
        # Filter by sex
        if sex_filter != "All":
            gender = "M" if sex_filter == "Men only" else "F"
            if 'sex' in df.columns:
                session_subjects = df[
                    (df['subject'].isin(session_subjects)) & 
                    (df['sex'] == gender)
                ]['subject'].tolist()
        
        # Filter by groups
        if groups:
            subject_groups = {subj: detect_group(subj) for subj in session_subjects}
            session_subjects = [s for s in session_subjects if subject_groups[s] in groups]

        
        # Exclude existing subjects
        subjects = [s for s in session_subjects if s not in existing_subjects]
        
        #Create title
        title = ""
            
        return subjects, title,  sex_filter, session 

def get_dataset(dataset_sel, data1, data2, master_store, data_source):
    if dataset_sel == 'dataset1':
        return data1
    elif dataset_sel == 'dataset2':
        return data2
    elif data_source == 'master': # or dataset_sel == 'master':
        return master_store

    return None

#a implémenter pour la gestion base/overlay plot
# def select_subjects(df, analysis_type, selected_subject, session, sex_filter, selected_groups):
#     """
#     Retourne la liste de sujets selon le type d'analyse et les filtres
#     """
#     if analysis_type == 'single':
#         if not selected_subject:
#             return [], "Error: Please select a subject"
#         return [selected_subject], f"Subject: {selected_subject}"

#     elif analysis_type == 'session_sex':
#         if not session:
#             return [], "Error: Please select a session"

#         # Filtrer par session
#         session_subjects = df[df['subject'].str.contains(f"_ses-{session}")]['subject'].tolist()

#         # Filtrer par sexe
#         if sex_filter != 'all' and 'sex' in df.columns:
#             gender = "M" if sex_filter == 'men' else "F"
#             session_subjects = df[
#                 (df['subject'].isin(session_subjects)) &
#                 (df['sex'] == gender)
#             ]['subject'].tolist()

#         # Filtrer par groupes
#         if selected_groups:
#             subjects = [s for s in session_subjects if detect_group(s) in selected_groups]
#         else:
#             subjects = session_subjects

#         if not subjects:
#             return [], "Error: No subjects found for these criteria"

#         # Construire le titre
#         title = f"Session {session}"
#         if sex_filter != 'all':
#             title += f" ({'Men' if sex_filter == 'men' else 'Women'})"
#         if selected_groups:
#             title += f" | Groups: {', '.join(selected_groups)}"

#         return subjects, title

#     else:
#         return [], "Error: Unknown analysis type"


#nouvelle fonction pour les statistiques
def get_data_for_analysis(dataset_sel, data1, data2, master_store):
    """Récupère les données appropriées pour l'analyse avec support master"""
    if dataset_sel == 'master-store' and master_store:
        return pd.DataFrame(master_store)
    elif dataset_sel == 'dataset1' and data1:
        return pd.DataFrame(data1)
    elif dataset_sel == 'dataset2' and data2:
        return pd.DataFrame(data2)
    elif dataset_sel == 'both' and data1 and data2:
        df1 = pd.DataFrame(data1)
        df2 = pd.DataFrame(data2)
        df = pd.concat([df1, df2], ignore_index=True)
        df['dataset'] = ['Dataset 1'] * len(df1) + ['Dataset 2'] * len(df2)
        return df
    return None

def get_subjects_for_analysis(df, analysis_type, session, sex, glm_subjects, 
                            group1_subjects, group2_subjects, corr_group1, 
                            corr_group2, method):
    """Filtre les sujets selon le type d'analyse"""
    if analysis_type == 'session_sex':
        # Implémenter la logique de filtrage par session et sexe
        subjects = df['subject'].unique()
        if session != 'all':
            subjects = [s for s in subjects if f"ses-{session}" in s]
        if sex != 'all':
            # Adapter selon la structure de vos données
            pass
        return subjects
    
    elif analysis_type == 'personalized':
        if method == 'glm':
            return glm_subjects or []
        elif method == 'ttest':
            return list(set((group1_subjects or []) + (group2_subjects or [])))
        elif method == 'correlation':
            return list(set((corr_group1 or []) + (corr_group2 or [])))
    
    return df['subject'].unique()

#---------fonction pour GLM ---------------------
def get_family_and_link(dist_name, link_name, var_power=None):
    link_map = {
        "log": Log(),
        "identity": Identity(),
        "inverse": InversePower(),
        "sqrt": Sqrt()
    }
    link_func = link_map.get(link_name, Log())

    if dist_name == "Gaussian":
        family = Gaussian(link=link_func)
    elif dist_name == "Gamma":
        family = Gamma(link=link_func)
    elif dist_name == "Poisson":
        family = Poisson(link=link_func)
    elif dist_name == "Tweedie":
        power = var_power if var_power is not None else 1.5
        family = Tweedie(var_power=power, link=link_func)
    else:
        family = Gaussian(link=link_func)

    return family

def check_model_assumptions(df, outcome, predictors, family):
    """Vérifie les hypothèses du modèle avec messages détaillés"""
    messages = []
    is_valid = True

    # Vérifier données manquantes dans le DataFrame
    missing = df[[outcome] + predictors].isna().sum()
    if missing.any():
        missing_vars = missing[missing > 0].to_dict()
        messages.append(f"❌ Missing data: {missing_vars}")
        is_valid = False

    # Vérifications spécifiques à la famille choisie
    outcome_values = df[outcome].dropna()
    if isinstance(family, Poisson):
        # Vérifier valeurs non-négatives
        if (outcome_values < 0).any():
            messages.append(f"❌ '{outcome}' contains negative values - incompatible with Poisson")
            is_valid = False
        
        # Vérifier valeurs entières
        if not all(outcome_values % 1 == 0):
            messages.append(f"❌ '{outcome}' contains non-integer values - Poisson requires integer counts")
            is_valid = False
        
        # Vérifier surdispersion (avertissement seulement)
        if len(outcome_values) > 1 and outcome_values.mean() > 0:
            variance_ratio = outcome_values.var() / outcome_values.mean()
            if variance_ratio > 1.5:
                messages.append(f"⚠️ Overdispersion in '{outcome}' (variance/mean = {variance_ratio:.2f}) - consider Tweedie")
    
    elif isinstance(family, Gamma):
        # Vérifier valeurs strictement positives 
        if (outcome_values <= 0).any():
            messages.append(f"❌ '{outcome}' contains values ≤0 - Gamma requires values > 0")
            is_valid = False
    
    elif isinstance(family, Tweedie):
        # Vérifier valeurs non-négatives
        if (outcome_values < 0).any():
            messages.append(f"❌ '{outcome}' contains negative values - Tweedie requires values ≥ 0")
            is_valid = False
    
    # Afficher tous les messages
    for message in messages:
        print(message)
    
    return is_valid

def extract_pseudo_r2_cs_from_summary(model):
    try:
        return 1 - (model.deviance / model.null_deviance)
    except:
        return None

def clean_predictor_name(predictor):
    """Nettoie le nom du prédicteur pour l'affichage """
    if not isinstance(predictor, str):
        return str(predictor)
    
    # Supprimer Q(' et ') si présent
    if predictor.startswith("Q('") and predictor.endswith("')"):
        predictor = predictor[3:-2]
    
    # Supprimer les interactions
    if ':' in predictor:
        predictor = predictor.split(':')[0]
    
    # Supprimer les préfixes pour avoir le nom de base
    prefixes = ['pre_', 'post_', 'loc_inj_perc_', 'tract_inj_perc_', 'loc_inj_', 'tract_inj_']
    for prefix in prefixes:
        if predictor.startswith(prefix):
            return predictor.replace(prefix, '')
    
    return predictor

def run_glm_analysis(
    df_predictors,
    df_outcomes,
    outcomes,
    systems,
    covariate=[],
    visit_name="",
    family=None,
    interaction_var=None
):
    results = []
    error_messages = []
    for outcome in outcomes:
        outcome_var = f"Q('{outcome}')" if any(c.isdigit() for c in outcome) else outcome

        for system, predictors in systems.items():
            for predictor in predictors:
                formula_terms = []
                term = f"Q('{predictor}')" if any(c.isdigit() for c in predictor) else predictor
                formula_terms.append(term)

                if interaction_var and interaction_var in df_predictors.columns:
                    formula_terms.append(f"{term}:{interaction_var}")

                for cov in covariate:
                    if cov != interaction_var:
                        cov_term = f"Q('{cov}')" if any(c.isdigit() for c in cov) else cov
                        formula_terms.append(cov_term)

                formula = f"{outcome_var} ~ {' + '.join(formula_terms)}"

                try:
                    df_predictors_temp = df_predictors.reset_index() if 'subject' not in df_predictors.columns else df_predictors.copy()
                    df_outcomes_temp = df_outcomes.reset_index() if 'subject' not in df_outcomes.columns else df_outcomes.copy()

                    needed_cols = list(set(['subject', predictor] + covariate))
                    if interaction_var and interaction_var not in needed_cols:
                        needed_cols.append(interaction_var)

                    if outcome not in df_outcomes_temp.columns:
                        print(f"{outcome} non trouvé dans les outcomes.")
                        continue

                    df_merged = df_outcomes_temp[['subject', outcome]].merge(
                        df_predictors_temp[needed_cols],
                        on='subject',
                        how='inner'
                    )

                    if df_merged.empty:
                        error_messages.append(f"Aucune donnée après merge pour {outcome} ~ {predictor}")
                        continue

                    drop_cols = list(set([outcome, predictor] + covariate + ([interaction_var] if interaction_var else [])))
                    df_clean = df_merged.dropna(subset=drop_cols)

                    if df_clean.empty or len(df_clean) < 3:
                        error_messages.append(f"Données insuffisantes pour {outcome} ~ {predictor} (n={len(df_clean)})")
                        continue

                    non_numeric = df_clean[drop_cols].select_dtypes(exclude=['number']).columns.tolist()
                    if non_numeric:
                        error_messages.append(f"Variables non numériques pour {outcome} ~ {predictor}: {non_numeric}")
                        continue

                    if not check_model_assumptions(df_clean, outcome, [predictor] + covariate, family):
                        error_messages.append(f"Hypothèses non respectées pour {outcome} ~ {predictor}")
                        continue

                    try:
                        model = smf.glm(formula, data=df_clean, family=family).fit()
                    except Exception as e:
                        error_messages.append(f"Erreur modèle pour {outcome} ~ {predictor}: {str(e)}")
                        continue

                    n_obs = int(model.nobs)
                    df_resid = int(model.df_resid)
                    df_model = int(model.df_model)
                    log_likelihood = model.llf
                    deviance = model.deviance
                    pearson_chi2 = model.pearson_chi2
                    pseudo_r2 = extract_pseudo_r2_cs_from_summary(model)
                    scale = model.scale

                    for param in model.params.index:
                        coef = model.params[param]
                        pval = model.pvalues[param]
                        is_interaction = ':' in param
                        base_pred = param.split(':')[0].replace("Q('", "").replace("')", "")

                        results.append({
                            'Visit': visit_name,
                            'Outcome': outcome,
                            'System': system,
                            'Predictor': param,
                            'Base_Predictor': base_pred,
                            'Coefficient': coef,
                            'Effect_Type': 'Interaction' if is_interaction else 'Main',
                            'P-value': pval,
                            'Significant': pval < 0.05,
                            'N_obs': n_obs,
                            'Df_resid': df_resid,
                            'Df_model': df_model,
                            'Log-likelihood': log_likelihood,
                            'Deviance': deviance,
                            'Pearson_chi2': pearson_chi2,
                            'Pseudo_R2_CS': pseudo_r2,
                            'Scale': scale,
                        })

                except Exception as e:
                    error_messages.append(f"Erreur avec {outcome} ~ {predictor}: {e}")
                    continue
    return pd.DataFrame(results), error_messages

def create_glm_results_visualizations(results_df):
    """Crée les visualisations pour les résultats GLM (comme dans Streamlit)"""
    
    if results_df.empty:
        return html.Div("No results to visualize")
    
    # Couleurs pour les systèmes (identique à Streamlit)
    base_colors = {
        # Systèmes acétycholinergiques
        'A4B2': '#76b7b2', 'M1': '#59a14f', 'VAChT': '#edc948',
        #Système dopaminergiques
        'D1': '#b07aa1', 'D2': '#ff9da7','DAT': '#9c755f',
        #Système Noradrénaline
        'Nor': '#79706e',
        #Systèmes sérotoninergiques
        '5HT1a': '#86bcb6', '5HT1b': '#d95f02', '5HT2a': '#e7298a',
        '5HT4': '#66a61e', '5HT6': '#e6ab02','5HTT': '#a6761d',
        # Systèmes GABA/Glutamate
        'GABAa': '#76b7b2', 'mGluR5': '#59a14f',
        # Systèmes opioïdes/Cannabinoïdes/Histamine
        'MU': '#edc948', 'CB1': '#b07aa1', 'H3': '#ff9da7',
    }
    
    # Étendre les couleurs aux préfixes
    neuro_colors = base_colors.copy()
    prefixes_pre = ['pre_', 'loc_inj_', 'tract_inj_', 'loc_inj_perc_', 'tract_inj_perc_']
    prefixes_post = ['post_', 'loc_inj_', 'tract_inj_','loc_inj_perc_', 'tract_inj_perc_']
    
    keys_pre = ['A4B2', 'M1', 'D1', 'D2', 'Nor', '5HT1a', '5HT1b', '5HT2a', '5HT4', '5HT6']
    keys_post = ['VAChT', 'DAT', '5HTT']

    
    for key in keys_pre:
        for prefix in prefixes_pre:
            neuro_colors[prefix + key] = base_colors[key]
    
    for key in keys_post:
        for prefix in prefixes_post:
            neuro_colors[prefix + key] = base_colors[key]
    
    # Créer un graphique par outcome
    viz_components = []
    
    for outcome in results_df['Outcome'].unique():
        outcome_data = results_df[results_df['Outcome'] == outcome].copy()
        outcome_data['Clean_Predictor'] = outcome_data['Predictor'].apply(clean_predictor_name)
        
        # Filtrer seulement les prédicteurs avec des couleurs définies
        outcome_data = outcome_data[outcome_data['Clean_Predictor'].isin(neuro_colors.keys())]
        
        if outcome_data.empty:
            continue
        
        # Créer le graphique
        fig = go.Figure()
        
        for _, row in outcome_data.iterrows():
            predictor = row['Clean_Predictor']
            color = neuro_colors.get(predictor, '#1f77b4')
            
            fig.add_trace(go.Bar(
                x=[predictor],
                y=[row['Coefficient']],
                name=predictor,
                marker_color=color,
                text=[f"p={row['P-value']:.3f}"],
                textposition='auto',
                hoverinfo='text',
                hovertext=(
                    f"Predictor: {row['Predictor']}<br>"
                    f"Clean: {predictor}<br>"
                    #f"Predictor: {predictor}<br>"
                    f"Coefficient: {row['Coefficient']:.3f}<br>"
                    f"p-value: {row['P-value']:.3f}<br>"
                    f"Significant: {'Yes' if row['Significant'] else 'No'}"
                )
            ))
            
            # Ajouter astérisque pour significativité
            if row['Significant']:
                fig.add_annotation(
                    x=predictor,
                    y=row['Coefficient'],
                    text="*",
                    showarrow=False,
                    font=dict(size=20, color='black'),
                    yshift=10
                )
        
        fig.update_layout(
            title=f"GLM Results for {outcome}",
            xaxis_title="Predictors",
            yaxis_title="Coefficient",
            barmode='group',
            showlegend=False,
            hovermode='closest',
            template='plotly_white',
            height=400
        )
        
        viz_components.append(
            dbc.Card([
                dbc.CardHeader(f"Outcome: {outcome}"),
                dbc.CardBody([
                    dcc.Graph(figure=fig)
                ])
            ], className="mb-4")
        )

        if not viz_components:
            return html.Div(
                dbc.Alert("No visualizable results found. Check console for details.", color="warning")
            )
    
    return html.Div(viz_components)

#---------fonction pour T-Test ---------------------
def perform_group_comparison(group1_data, group2_data, paired=False):
    """
    Effectue une comparaison statistique entre deux groupes avec vérifications préalables
    
    Args:
        group1_data (pd.Series): Données du groupe 1
        group2_data (pd.Series): Données du groupe 2
        paired (bool): Si True, utilise un test apparié quand V1 ; V2 ; V3 comparaison
        
    Returns:
        dict: Dictionnaire contenant tous les résultats statistiques
    """
    # Nettoyage des données
    vals1 = group1_data.dropna()
    vals2 = group2_data.dropna()
    
    # Vérification des effectifs
    if len(vals1) < 3 or len(vals2) < 3:
        return None
    
    if paired and len(vals1) != len(vals2):
        raise ValueError("For paired tests, group sizes must be equal")

    results = {
        'n_group1': len(vals1),
        'n_group2': len(vals2),
        'mean_group1': vals1.mean(),
        'mean_group2': vals2.mean(),
        'std_group1': vals1.std(),
        'std_group2': vals2.std()
    }
    
    # 1. Test de normalité (Shapiro-Wilk)
    shapiro1 = shapiro(vals1)
    shapiro2 = shapiro(vals2)
    results.update({
        'shapiro_p1': shapiro1.pvalue,
        'shapiro_p2': shapiro2.pvalue,
        'normal_dist': (shapiro1.pvalue > 0.05) and (shapiro2.pvalue > 0.05)
    })
    
    # 2. Test d'homogénéité des variances (Levene) - seulement si non apparié
    if not paired:
        levene_test = levene(vals1, vals2)
        results.update({
            'levene_p': levene_test.pvalue,
            'equal_var': levene_test.pvalue > 0.05
        })
    
    # Choix du test statistique
    if paired:
        # Tests pour données appariées
        if results['normal_dist']:
            test_result = ttest_rel(vals1, vals2)
            results.update({
                'test_type': 'Paired t-test',
                'statistic': test_result.statistic,
                'p_value': test_result.pvalue,
                'effect_size': (vals1.mean() - vals2.mean()) / np.sqrt((vals1.std()**2 + vals2.std()**2)/2)
            })
        else:
            test_result = wilcoxon(vals1, vals2)
            results.update({
                'test_type': 'Wilcoxon signed-rank',
                'statistic': test_result.statistic,
                'p_value': test_result.pvalue,
                'effect_size': test_result.statistic / np.sqrt(len(vals1))
            })
    else:
        # Tests pour groupes indépendants
        if results['normal_dist']:
            if results.get('equal_var', True):
                test_result = ttest_ind(vals1, vals2, equal_var=True)
                test_type = "Student-t (var égales)"
            else:
                test_result = ttest_ind(vals1, vals2, equal_var=False)
                test_type = "Welch-t (var inégales)"
            
            effect_size = (vals1.mean() - vals2.mean()) / np.sqrt((vals1.std()**2 + vals2.std()**2)/2)
            
            try:
                analysis = TTestIndPower()
                power = analysis.power(
                    effect_size=effect_size, 
                    nobs1=len(vals1), 
                    alpha=0.05,
                    ratio=len(vals2)/len(vals1), 
                    alternative='two-sided'
                )
            except:
                power = np.nan
                
            results.update({
                'test_type': test_type,
                'statistic': test_result.statistic,
                'p_value': test_result.pvalue,
                'effect_size': effect_size,
                'power': power
            })
        else:
            test_result = mannwhitneyu(vals1, vals2, alternative='two-sided')
            n1, n2 = len(vals1), len(vals2)
            U = test_result.statistic
            Z = (U - n1*n2/2) / np.sqrt(n1*n2*(n1+n2+1)/12)  # Conversion U → Z
            p_value_from_z = 2 * (1 - norm.cdf(abs(Z)))
            effect_size = Z / np.sqrt(n1 + n2)
            
            results.update({
                'test_type': "Mann-Whitney U",
                'statistic': test_result.statistic,
                'p_value': test_result.pvalue,
                'statistic_z': Z,
                'p_value_from_z': p_value_from_z,
                'effect_size': effect_size,
                #'power': np.nan
            })
    
    results['significant'] = results['p_value'] < 0.05
    return results

def clean_groups_for_variable(df1, df2, var, paired):
    """Supprime les sujets ayant des valeurs manquantes pour une variable sélectionné par l'utilsateur.
       Si paired, conserve uniquement les paires valides."""
    df1_valid = df1[df1[var].notna()]
    df2_valid = df2[df2[var].notna()]

    if paired:
        # On garde les sujets présents et valides dans les deux groupes
        base_ids_1 = df1_valid['subject'].apply(lambda x: x.split('-V')[0])
        base_ids_2 = df2_valid['subject'].apply(lambda x: x.split('-V')[0])
        common_bases = set(base_ids_1).intersection(set(base_ids_2))

        df1_clean = df1_valid[df1_valid['subject'].apply(lambda x: x.split('-V')[0]).isin(common_bases)]
        df2_clean = df2_valid[df2_valid['subject'].apply(lambda x: x.split('-V')[0]).isin(common_bases)]

        return df1_clean, df2_clean, len(common_bases)
    else:
        return df1_valid, df2_valid, None

#---------fonction pour Corrélation ---------------------
def get_correlation_matrix(df, include_sex_bin=True):
    """
    Calculate correlation matrix with automatic test selection (Pearson/Spearman)
    and FDR correction for multiple comparisons.
    
    Parameters:
    - df: DataFrame containing the variables to correlate
    - include_sex_bin: Whether to include 'Sexe_bin' in the matrix (False for single-sex analysis)
    
    Returns:
    - corr_matrix: DataFrame of correlation coefficients
    - pval_matrix: DataFrame of FDR-corrected p-values
    """
    # Select and clean numeric data
    df_num = df.select_dtypes(include=['float64', 'int64','bool']).dropna(axis=1, thresh=int(0.5 * len(df)))
    
    # Convert bool to int if needed
    df_num = df_num.apply(lambda x: x.astype(int) if x.dtype == bool else x)

    # Exclude Sexe_bin if requested
    if not include_sex_bin and 'Sexe_bin' in df_num.columns:
        df_num = df_num.drop(columns=['Sexe_bin'])
        
    cols = df_num.columns
    corr_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
    pval_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
    
    pvals_list = []
    index_pairs = []

    for col1, col2 in combinations(cols, 2):
        x, y = df_num[col1].dropna(), df_num[col2].dropna()
        common_index = x.index.intersection(y.index)
        x, y = x.loc[common_index], y.loc[common_index]

        if len(x) < 3:
            continue  # Skip pairs with less than 3 observations

        # Normality test with error handling
        try:
            norm_x = shapiro(x)[1] > 0.05
            norm_y = shapiro(y)[1] > 0.05
        except:
            norm_x = norm_y = False

        # Choose appropriate correlation test
        if norm_x and norm_y:
            corr, pval = pearsonr(x, y)
        else:
            corr, pval = spearmanr(x, y)

        # Store results
        corr_matrix.loc[col1, col2] = corr
        corr_matrix.loc[col2, col1] = corr
        pval_matrix.loc[col1, col2] = pval
        pval_matrix.loc[col2, col1] = pval
        
        # Prepare for FDR correction
        pvals_list.append(pval)
        index_pairs.append((col1, col2))

    # Fill diagonal
    np.fill_diagonal(corr_matrix.values, 1.0)
    for col in cols:
        pval_matrix.loc[col, col] = 0.0  # p-value for diagonal is 0

    # Apply FDR correction (Benjamini-Hochberg)
    if pvals_list:
        _, pvals_corrected, _, _ = multipletests(pvals_list, alpha=0.05, method='fdr_bh')
        for (col1, col2), p_corr in zip(index_pairs, pvals_corrected):
            pval_matrix.loc[col1, col2] = p_corr
            pval_matrix.loc[col2, col1] = p_corr
    
    return corr_matrix, pval_matrix

def extract_subject_id(subject_name):
                match = re.match(r"(sub-[A-Za-z0-9]+)_ses-V[0-9]+", subject_name)
                return match.group(1) if match else None

def clean_predictor_name(name):
                    # Enlève Q(' et ') si présent
                    if isinstance(name, str) and name.startswith("Q('") and name.endswith("')"):
                        return name[3:-2]
                    return name

def style_corr_with_pval(corr_df, pval_df, p_thresh):
    def highlight(val, pval):
        if pval >= p_thresh:
            # Griser les cases en conservanr la valeur
            return 'color: lightgray;'
        else:
            return ''

def prepare_correlation_data_two_sets(df, subjects1, vars1, subjects2, vars2, session1, session2):
    """Prépare les données pour l'analyse de corrélation avec deux sets indépendants - VERSION SIMPLIFIÉE"""
    
    # # Filtrer le DataFrame avec tous les sujets des deux sets
    all_subjects = list(set(subjects1 + subjects2))
    df_filtered = df[df['subject'].isin(all_subjects)].copy()
    
    # Extraire l'ID de base du sujet
    df_filtered["subject_base"] = df_filtered["subject"].apply(extract_subject_id)

    # Préparer les DataFrames pour chaque set avec les sessions spécifiques
    df1 = df_filtered[
        (df_filtered["subject"].isin(subjects1)) &
        (df_filtered["subject"].str.contains(f"_ses-{session1}"))
    ][["subject_base"] + vars1].drop_duplicates()

    df2 = df_filtered[
        (df_filtered["subject"].isin(subjects2)) &
        (df_filtered["subject"].str.contains(f"_ses-{session2}"))
    ][["subject_base"] + vars2].drop_duplicates()

    # Gérer les suffixes basés sur les sessions
    suffix1 = f"_{session1}_1" if session1 == session2 else f"_{session1}"
    suffix2 = f"_{session2}_2" if session1 == session2 else f"_{session2}"

    df1_renamed = df1.rename(columns={col: col + suffix1 for col in vars1})
    df2_renamed = df2.rename(columns={col: col + suffix2 for col in vars2})
    
    # Vérification des sujets communs
    common_ids = sorted(set(df1_renamed["subject_base"]) & set(df2_renamed["subject_base"]))
    if len(common_ids) < 3:
        raise ValueError(f"Only {len(common_ids)} common subjects found (minimum 3 required)")
    
    # Fusion finale
    df_corr = df1_renamed[df1_renamed["subject_base"].isin(common_ids)].merge(
        df2_renamed[df2_renamed["subject_base"].isin(common_ids)],
        on="subject_base"
    ).drop(columns=['subject_base'])

    return df_corr, suffix1, suffix2

def create_correlation_results(corr_matrix, pval_matrix, vars1, vars2, session1, session2, suffix1, suffix2):
    """Crée les composants d'affichage des résultats pour deux sessions"""
    
    # Extraire les corrélations croisées entre set1 et set2
    set1_cols = [col for col in corr_matrix.columns if col.endswith(suffix1)]
    set2_cols = [col for col in corr_matrix.columns if col.endswith(suffix2)]
    set1_cols_p = [col for col in pval_matrix.columns if col.endswith(suffix1)]
    set2_cols_p = [col for col in pval_matrix.columns if col.endswith(suffix2)]
    
    cross_corr = corr_matrix.loc[set1_cols, set2_cols]
    cross_pvals = pval_matrix.loc[set1_cols_p, set2_cols_p]

    # Onglets pour différents types de visualisation
    tabs = dcc.Tabs([
        # Onglet Matrice complète
        dcc.Tab(label='Full Matrix', children=[
            html.Div([
                html.H4("Full Correlation Matrix"),
                dash_table.DataTable(
                    id='full-correlation-table',
                    columns=[{"name": col, "id": col} for col in corr_matrix.columns],
                    data=corr_matrix.round(3).to_dict('records'),
                    style_cell={'textAlign': 'center', 'padding': '5px'},
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                    page_size=10
                ),
                html.H4("P-Value Matrix"),
                dash_table.DataTable(
                    id='pvalue-table',
                    columns=[{"name": col, "id": col} for col in pval_matrix.columns],
                    data=pval_matrix.round(4).to_dict('records'),
                    style_cell={'textAlign': 'center', 'padding': '5px'},
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                    page_size=10
                )
            ])
        ]),
        
        # Onglet Heatmap
        dcc.Tab(label='Visualization', children=[
            html.Div([
                html.H4(f"Correlation Heatmap (Session {session1} vs {session2})"),
                dcc.Graph(
                    id='correlation-heatmap',
                    config={'displayModeBar': True}
                ),
                html.Div([
                    dcc.Checklist(
                        id='show-all-correlations',
                        options=[{'label': 'Show all correlations (incl. non-significant)', 'value': 'show_all'}],
                        value=[],
                        inline=True
                    ),
                    html.Label("p-value threshold:"),
                    dcc.Slider(
                        id='corr-p-threshold',
                        min=0.001,
                        max=0.1,
                        value=0.05,
                        step=0.001,
                        marks={i/100: str(i/100) for i in range(1, 11, 2)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], className='mb-3 p-3 border rounded')
            ])
        ]),
        
        # Onglet Corrélations croisées
        dcc.Tab(label='Cross Correlations', children=[
            html.Div([
                html.H4(f"Cross Correlation Matrix (Session {session1} vs {session2})"),
                dash_table.DataTable(
                    id='cross-correlation-table',
                    columns=[{"name": col, "id": col} for col in cross_corr.columns],
                    data=cross_corr.round(3).to_dict('records'),
                    style_cell={'textAlign': 'center', 'padding': '5px'},
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                    page_size=10
                ),
                html.H4("Cross P-Values Matrix"),
                dash_table.DataTable(
                    id='cross-pvalues-table',
                    columns=[{"name": col, "id": col} for col in cross_pvals.columns],
                    data=cross_pvals.round(4).to_dict('records'),
                    style_cell={'textAlign': 'center', 'padding': '5px'},
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                    page_size=10
                )
            ])
        ]),
        
        # Onglet Corrélations significatives
        dcc.Tab(label='Significant Correlations', children=[
            html.Div([
                html.H4("Significant Correlations"),
                html.Div(id='significant-correlations-table')
            ])
        ])
    ])
    
    return html.Div([
        html.H3("🔗 Correlation Analysis Results"),
        html.P(f"Session 1: {session1}, {len(vars1)} variables | Session 2: {session2}, {len(vars2)} variables"),
        html.P(f"Common subjects: {cross_corr.shape[0]}"),
        tabs,
        html.Hr(),
        html.Button("📥 Download Results", id="download-correlation-btn", className="btn btn-success"),
        dcc.Download(id="download-correlation"),
        
        # Stockage des données pour les callbacks interactifs
        dcc.Store(id='correlation-cross-data', data={
            'cross_corr': cross_corr.to_dict('records'),
            'cross_pvals': cross_pvals.to_dict('records'),
            'corr_columns': cross_corr.columns.tolist(),
            'corr_index': cross_corr.index.tolist(),
            'session1': session1,
            'session2': session2,
            'full_corr': corr_matrix.to_dict('records'),
            'full_pvals': pval_matrix.to_dict('records'),
            'full_columns': corr_matrix.columns.tolist(),
            'suffix1': suffix1,
            'suffix2': suffix2
        })
    ])

# ---------------------------- Layout setting -------------------------------

app.layout = dbc.Container([
    # ==================== STORES Main ==================================
    dcc.Store(id='dataset1-store'),
    dcc.Store(id='dataset2-store'),
    dcc.Store(id='plots-store'), 
    dcc.Store(id='overlay-store', data={'overlays': []}),
    dcc.Store(id='session-state', data={
        'base_plots': None,
        'overlay_plots': None,
        'show_overlay': False,
        'overlay_subjects': [],
        'overlay_color_map': {},
        'subject_selections': {} ,
        'current_subjects': [],
        'current_title': '',
        'current_session': None,
        'current_sex_filter': 'All',
        'current_groups': []
    }),
    dcc.Store(id='master-store'),
    #dcc.Store(id='ttest-cleaned-data-store', data=None),

    # ==================== SOURCE DE DONNÉES ===========================
    dcc.RadioItems(
        id='data-source',
        options=[
            {'label': 'Master (Article)', 'value': 'master'},
            {'label': 'Upload', 'value': 'upload'}
        ],
        value='master',
        inline=True,
        style={'display': 'none'}
    ),

    # ==================== HEADER ======================================
    html.Div([
        # dbc.Row([
        #     dbc.Col([
        #         html.H1("🧠 Neurotransmitter Balance & Outcomes", 
        #             className="text-center mb-3",
        #             style={'color': '#e6f8ec', 'font-weight': '600'}),
        #         html.H4("Explore neurotransmitter ratios and their clinical relevance", 
        #             className="text-center",
        #             style={'color': '#c8e6c9', 'font-weight': '400'})
        #     ], width=12)
        # ], className="p-4 mb-4 rounded", 
        # style={'background-color': '#9caf88', 'box-shadow': '0 4px 12px rgba(0, 0, 0, 0.08)'})
        html.Div([
            dbc.Col([
                html.H1("🧠 Neurotransmitter Balance & Outcomes", 
                    className="custom-title text-center mb-3",
                    style={'color': '#e6f8ec', 'font-weight': '600'}),
                html.H4("Explore neurotransmitter ratios and their clinical relevance", 
                    className="custom-subtitle text-center",
                    style={'color': '#c8e6c9', 'font-weight': '400'})
            ], className="p-4 mb-4 rounded", 
           style={'background-color': '#9caf88', 'box-shadow': '0 4px 12px rgba(0, 0, 0, 0.08)'})
        ])
    ]),

    # ==================== SECTION UPLOAD =============================
    html.Hr(),
    html.H3("📦 Data Upload"),
    dbc.Row([
        dbc.Col([
            html.Label("Upload Dataset 1", className="form-label"),
            dcc.Upload(
                id='upload-data1',
                children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
                style={
                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                    'borderWidth': '1px', 'borderStyle': 'dashed',
                    'borderRadius': '5px', 'textAlign': 'center',
                    'margin': '10px'
                },
                multiple=False,
                accept='.zip'
            )
        ], width=6),
        dbc.Col([
            html.Label("Upload Dataset 2 (Optional)", className="form-label"),
            dcc.Upload(
                id='upload-data2',
                children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
                style={
                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                    'borderWidth': '1px', 'borderStyle': 'dashed',
                    'borderRadius': '5px', 'textAlign': 'center',
                    'margin': '10px'
                },
                multiple=False,
                accept='.zip'
            )
        ], width=6)
    ]),

    # Statut upload
    html.Div(id='upload-status'),

    # ==================== TABS POUR DIFFÉRENTES VUES ====================
    dbc.Tabs([
        dbc.Tab(label="Visualization", tab_id="tab-visualization"),
        dbc.Tab(label="Statistical Analysis", tab_id="tab-stats"),
        dbc.Tab(label="Data Explorer", tab_id="tab-data"),
    ], id="tabs", active_tab="tab-visualization"),
    
    # Contenu dynamique des tabs
    html.Div(id="tab-content", style={'min-height': '400px'}),

    # ==================== SECTION PRINCIPALE ============================
    html.Div(id="main-configuration-container", children=[ 
    
        html.Hr(),
        html.H3("📊 Profile Configuration"),
        
        # Sélecteurs principaux (dataset et type d'analyse)
        dbc.Row([
            dbc.Col([
                html.Label("Dataset Selection:"),
                dcc.RadioItems(
                    id='dataset-selector',
                    options=[
                        {'label': 'Dataset 1', 'value': 'dataset1'},
                        {'label': 'Dataset 2', 'value': 'dataset2'},
                        #{'label': 'Master', 'value': 'master-store'},
                    ],
                    value='dataset1',
                    inline=True
                )
            ], width=6),
            dbc.Col([
                html.Label("Analysis Type:"),
                dcc.RadioItems(
                    id='analysis-type',
                    options=[
                        {'label': 'Single subject', 'value': 'single'},
                        {'label': 'By session and sex', 'value': 'session_sex'}
                    ],
                    value='session_sex',
                    inline=True
                )
            ], width=6),
        ]),
        
        html.Br(),
        
        # ==================== CONTENEURS DE FILTRES (affichage dynamique) ====================
        
        # Conteneur pour sélection de sujet unique
        html.Div(id='subject-selection-container', children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Select Subject:"),
                    dcc.Dropdown(
                        id='subject-dropdown',
                        options=[],
                        value=None,
                        placeholder="Select a subject"
                    )
                ], width=12)
            ])
        ], style={'display': 'none'}),
        
        # Conteneur pour sélection de session
        html.Div(id='session-filter-container', children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Select Session:"),
                    dcc.Dropdown(
                        id='session-dropdown',
                        options=[],
                        value=None,
                        placeholder="Select session"
                    )
                ], width=4)
            ])
        ], style={'display': 'none'}),
        
        # Conteneur pour filtre de sexe
        html.Div(id='sex-filter-container', children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Select Sex:"),
                    dcc.Dropdown(
                        id='sex-filter',
                        options=[],
                        value=None,
                        placeholder="Select sex filter"
                    )
                ], width=4)
            ])
        ], style={'display': 'none'}),
        
        # Conteneur pour sélection de groupes
        html.Div(id='group-selection-container', children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Filter by Subject Group:"),
                    dcc.Checklist(
                        id='group-checklist',
                        options=[],
                        value=[],
                        inline=True
                    )
                ], width=12)
            ])
        ], style={'display': 'none'}),
        
        html.Br(),
        
        # ==================== AFFICHAGE DU STATUT ====================
        dbc.Row([
            dbc.Col([
                html.Label("Subjects Selected:"),
                html.H4(id='subject-count-display', 
                    children="0 subjects selected", 
                    style={'font-weight': 'bold', 'color': '#2c3e50'})
            ], width=6),
            dbc.Col([
                html.Label("Profile Status:"),
                html.H4(id='profile-title', 
                    children="No profile generated", 
                    style={'font-weight': 'bold', 'color': '#7f8c8d'}
                ),
                html.H5(
                id='profile-title-overlay', 
                children="No overlay generated", 
                style={'font-weight': 'bold', 'color': '#7f8c8d', 'margin-top': '10px'})       
            ], width=6)
        ]),
        
        html.Br(),
        
        # ==================== BOUTON GÉNÉRATION ====================
        dbc.Row([
            dbc.Col([
                dbc.Button("Generate Profile", 
                        id='generate-profile-btn', 
                        color="primary", 
                        size="lg",
                        className="w-100")
            ], width=12)
        ]),
        
        html.Hr(),
    ], style={'display': 'none'}),  
    # ==================== SECTION OVERLAY ====================

    html.Div(id='enable-overlay-container', children=[ #modifier
        #html.Div([
            html.Label("Enable Overlay:"),
            dcc.RadioItems(
                id='enable-overlay',
                options=[
                    {'label': 'Yes', 'value': 'yes'},
                    {'label': 'No', 'value': 'no'}
                ],
                value='no',
                inline=True
            )
        ], style={'display': 'none'}),
            
    html.Div(id='overlay-options-container', children=[
        html.Hr(),
        html.H4("Overlay Configuration"),
        dbc.Row([
            dbc.Col([
                html.Label("Overlay Dataset:"),
                dcc.RadioItems(
                    id='overlay-dataset-selector',
                    options=[
                        {'label': 'Dataset 1', 'value': 'dataset1'},
                        {'label': 'Dataset 2', 'value': 'dataset2'}
                    ],
                    value='dataset1',
                    inline=True
                )
            ], width=6),
            dbc.Col([
                html.Label("Overlay Analysis Type:"),
                dcc.RadioItems(
                    id='overlay-analysis-type',
                    options=[
                        {'label': 'Single subject', 'value': 'single'},
                        {'label': 'By session and sex', 'value': 'session_sex'}
                    ],
                    value='single',
                    inline=True
                )
            ], width=6),
        ]), 
        
        html.Br(),


        # Conteneurs d'overlay (mêmes que les principaux)
        html.Div(id='overlay-subject-container', children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Select Overlay Subject:"),
                    dcc.Dropdown(
                        id='overlay-subject-dropdown',
                        options=[],
                        value=None,
                        placeholder="Select overlay subject"
                    )
                ], width=12)
            ])
        ], style={'display': 'none'}),

        html.Div(id='overlay-session-container', children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Overlay Session:"),
                    dcc.Dropdown(
                        id='overlay-session-dropdown',
                        options=[],
                        value=None,
                        placeholder="Select overlay session"
                    )
                ], width=4)
            ])
        ], style={'display': 'none'}),
        
        html.Div(id='overlay-sex-container', children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Overlay Sex:"),
                    dcc.Dropdown(
                        id='overlay-sex-filter',
                        options=[],
                        value=None,
                        placeholder="Select overlay sex filter"
                    )
                ], width=4)
            ])
        ], style={'display': 'none'}),
        
        html.Div(id='overlay-groups-container', children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Overlay Groups:"),
                    dcc.Checklist(
                        id='overlay-group-checklist',
                        options=[],
                        value=[],
                        inline=True
                    )
                ], width=12)
            ])
        ], style={'display': 'none'}),
        
        html.Br(),
        
        dbc.Row([
            dbc.Col([
                html.Label("Overlay Status:"),
                html.Div(id='overlay-subject-count', 
                        children="0 overlay subjects", 
                        style={'font-weight': 'bold', 'color': '#e74c3c'})
            ], width=6),
            dbc.Col([
                dbc.Button("Add Overlay", 
                        id='add-overlay-btn', 
                        color="warning", 
                        size="md"),
                dbc.Button("Clear Overlays", 
                        id='clear-overlay-btn', 
                        color="danger", 
                        size="md",
                        className="ms-2")
            ], width=6)
        ])
    ], style={'display': 'none'}), 

    # ==================== ÉLÉMENTS STATISTIQUES (cachés) ====================
    html.Div([
        html.Div(id='stats-method-container', children=[
            dbc.Label("Choose Analysis Method:", className="fw-bold mb-0"),
            #html.Label("Choose Analysis Method:", className="fw-bold"),
            dcc.RadioItems(
                id='stats-method',
                options=[
                    {'label': 'GLM', 'value': 'glm'},
                    {'label': 'T-Test', 'value': 'ttest'},
                    {'label': 'Correlation', 'value': 'correlation'}
                ],
                value='glm',
                inline=True,
            )
        ], style={'display': 'none'}),  # ← Conteneur caché au début
        
        html.Div(id='stats-dataset-container', children=[
            dbc.Label("Select Dataset for Analysis:", className="fw-bold mb-0"),
            #html.Label("Select Dataset for Analysis:", className="fw-bold"),
            dcc.RadioItems(
                id='stats-dataset',
                options=[
                    {'label': 'Dataset 1', 'value': 'dataset1'},
                    {'label': 'Dataset 2', 'value': 'dataset2'},
                    {'label': 'Both Datasets', 'value': 'both'}
                ],
                value='dataset1',
                inline=True,
            )
        ], style={'display': 'none'}),
        
        html.Div(id='stats-analysis-type-container', children=[
            dbc.Label("Analysis Type:", className="fw-bold mb-0"), 
            #html.Label("Analysis Type:", className="fw-bold"),
            dcc.RadioItems(
                id='stats-analysis-type',
                options=[
                    {'label': 'By session and sex', 'value': 'session_sex'},
                    {'label': 'Personalized subject list', 'value': 'personalized'}
                ],
                value='session_sex',
                inline=True, 
            )
        ], style={'display': 'none'}) 
    ]),
    # ==================== SECTION GLM (CACHÉE - AFFICHÉE DYNAMIQUEMENT) ====================
    html.Div(id='glm-analysis-container', children=[
        html.H3(" GLM Analysis", className="mb-4"),
        
        # Sélection des sujets
        dbc.Card([
            dbc.CardHeader("Subject Selection"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Session"),
                        dcc.Dropdown(id='glm-session', options=[], value='V1')
                    ], width=4),
                    dbc.Col([
                        dbc.Label("Sex Filter"),
                        dcc.Dropdown(id='glm-sex', options=[], value='all')
                    ], width=4),
                    dbc.Col([
                        dbc.Label("Subject Groups"),
                        dcc.Checklist(id='glm-groups', options=[], value=[], inline=True)
                    ], width=4)
                ]),
                html.Br(),
                html.Div(id='glm-subject-count', className='text-muted')
            ])
        ], className="mb-4"),
        # Configuration de base
        dbc.Card([
            dbc.CardHeader("Analysis Configuration"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Distribution Family"),
                        dcc.Dropdown(
                            id='glm-distribution',
                            options=[
                                {'label': 'Gaussian', 'value': 'Gaussian'},
                                {'label': 'Gamma', 'value': 'Gamma'},
                                {'label': 'Poisson', 'value': 'Poisson'},
                                {'label': 'Tweedie', 'value': 'Tweedie'}
                            ],
                            value='Gaussian'
                        )
                    ], width=4),
                    dbc.Col([
                        dbc.Label("Link Function"),
                        dcc.Dropdown(
                            id='glm-link',
                            options=[
                                {'label': 'Identity', 'value': 'identity'},
                                {'label': 'Log', 'value': 'log'},
                                {'label': 'Inverse', 'value': 'inverse'},
                                {'label': 'Square Root', 'value': 'sqrt'}
                            ],
                            value='identity'
                        )
                    ], width=4),
                    dbc.Col([
                        dbc.Label("Tweedie Power (if applicable)"),
                        dcc.Input(
                            id='glm-tweedie-power',
                            type='number',
                            value=1.5,
                            step=0.1,
                            min=0,
                            max=3,
                            disabled=True
                        )
                    ], width=4)
                ], className="mb-3"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Checklist(
                            id='glm-interaction',
                            options=[{"label": " Include Interaction", "value": True}],
                            value=[]
                        )
                    ], width=6),
                    dbc.Col([
                        dcc.Dropdown(
                            id='glm-interaction-var',
                            options=[],
                            placeholder="Select interaction variable",
                            disabled=True
                        )
                    ], width=6)
                ], className="mb-3")
            ])
        ], className="mb-4"),
        
        # Sélection des variables
        dbc.Card([
            dbc.CardHeader("Variable Selection"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Outcome Variables"),
                        dcc.Dropdown(
                            id='glm-outcomes',
                            options=[],
                            multi=True,
                            placeholder="Select dependent variables"
                        )
                    ], width=6),
                    dbc.Col([
                        dbc.Label("Covariates"),
                        dcc.Dropdown(
                            id='glm-covariates',
                            options=[],
                            multi=True,
                            placeholder="Select covariates"
                        )
                    ], width=6)
                ], className="mb-3"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Predictor System"),
                        dbc.RadioItems(
                            id='glm-predictor-system',
                            options=[
                                {"label": "Synaptic ratio", "value": "synaptic"},
                                {"label": "Neurotransmitter (Loc)", "value": "nt_loc"},
                                {"label": "Neurotransmitter (Tract)", "value": "nt_tract"}
                            ],
                            value="synaptic",
                            inline=True
                        )
                    ], width=12)
                ])
            ])
        ], className="mb-4"),
        
        # Visualisation des données
        dbc.Card([
            dbc.CardHeader("Data Visualization"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Select Variable to Visualize"),
                        dcc.Dropdown(id='glm-viz-variable', options=[])
                    ], width=4),
                    dbc.Col([
                        dbc.Label("Color By"),
                        dcc.Dropdown(
                            id='glm-viz-color',
                            options=[{'label': 'None', 'value': 'none'}],
                            value='none'
                        )
                    ], width=4),
                    dbc.Col([
                        dbc.Checklist(
                            id='glm-viz-points',
                            options=[{"label": " Show Individual Points", "value": True}],
                            value=[True]
                        )
                    ], width=4)
                ]),
                dcc.Graph(id='glm-viz-plot')
            ])
        ], className="mb-4"),
        
        # Bouton d'exécution
        dbc.Button(" Run GLM Analysis", id='run-glm-analysis', color="primary", className="mb-4"),
        
        # Résultats
        html.Div(id='glm-results-container')
    ], style={'display': 'none'}),  # Caché par défaut

     # ==================== SECTION T-TEST (CACHÉE - AFFICHÉE DYNAMIQUEMENT) ====================
    html.Div(id='ttest-analysis-container', children=[
        html.H3(" T-Test Analysis", className="mb-4"),
        
        # Configuration de base
        dbc.Card([
            dbc.CardHeader("T-Test Configuration"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Test Type"),
                        dcc.RadioItems(
                            id='ttest-type',
                            options=[
                                {'label': 'Independent samples', 'value': 'independent'},
                                {'label': 'Paired samples', 'value': 'paired'}
                            ],
                            value='independent',
                            inline=True
                        )
                    ], width=6),
                    dbc.Col([
                        dbc.Label("Dataset for Analysis"),
                        dcc.RadioItems(
                            id='ttest-dataset',
                            options=[
                                {'label': 'Dataset 1', 'value': 'dataset1'},
                                {'label': 'Dataset 2', 'value': 'dataset2'},
                                {'label': 'Master', 'value': 'master-store'}
                            ],
                            value='dataset1',
                            inline=True
                        )
                    ], width=6)
                ])
            ])
        ], className="mb-4"),
        
        # Sélection des groupes
        dbc.Card([
            dbc.CardHeader("Group Selection"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Group 1 Configuration"),
                        html.Div([
                            dbc.Label("Session"),
                            dcc.Dropdown(id='ttest-group1-session', options=[], value='V1'),
                            html.Br(),
                            dbc.Label("Sex Filter"),
                            dcc.Dropdown(id='ttest-group1-sex', options=[], value='all'),
                            html.Br(),
                            dbc.Label("Subject Groups"),
                            dcc.Checklist(id='ttest-group1-groups', options=[], value=[], inline=True),
                            html.Br(),
                            html.Div(id='ttest-group1-count', className='text-muted')
                        ])
                    ], width=6),
                    dbc.Col([
                        dbc.Label("Group 2 Configuration"),
                        html.Div([
                            dbc.Label("Session"),
                            dcc.Dropdown(id='ttest-group2-session', options=[], value='V1'),
                            html.Br(),
                            dbc.Label("Sex Filter"),
                            dcc.Dropdown(id='ttest-group2-sex', options=[], value='all'),
                            html.Br(),
                            dbc.Label("Subject Groups"),
                            dcc.Checklist(id='ttest-group2-groups', options=[], value=[], inline=True),
                            html.Br(),
                            html.Div(id='ttest-group2-count', className='text-muted')
                        ])
                    ], width=6)
                ])
            ])
        ], className="mb-4"),
        
        # Sélection des variables
        dbc.Card([
            dbc.CardHeader("Variable Selection"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Variables to Compare"),
                        dcc.Dropdown(
                            id='ttest-variables',
                            options=[],
                            multi=True,
                            placeholder="Select variables for comparison"
                        )
                    ], width=12)
                ])
            ])
        ], className="mb-4"),
        
        # Bouton d'exécution
        dbc.Button(" Run T-Test Analysis", id='run-ttest-analysis', color="success", className="mb-4"),
        
        # Résultats
        html.Div(id='ttest-results-container'),
        # Store pour les données (ajoutez cette ligne)
        dcc.Store(id='ttest-cleaned-data-store')
    ], style={'display': 'none'}),

    # ==================== SECTION CORRELATION (CACHÉE - AFFICHÉE DYNAMIQUEMENT) ====================
    html.Div(id='correlation-analysis-container', children=[
    html.H3(" Correlation Analysis", className="mb-4"),
    
    # Sélection des sujets PAR SET
    dbc.Card([
        dbc.CardHeader("Subject Selection by Set"),
        dbc.CardBody([
            dbc.Row([
                # SET 1
                dbc.Col([
                    html.H5("Set 1 Subjects", className="text-primary"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Session"),
                            dcc.Dropdown(
                                id='corr-session1', 
                                options=[], 
                                value='V1',
                                placeholder="Select session for Set 1"
                            )
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Sex Filter"),
                            dcc.Dropdown(
                                id='corr-sex1', 
                                options=[], 
                                value='all',
                                placeholder="Select sex for Set 1"
                            )
                        ], width=6)
                    ]),
                    dbc.Label("Subject Groups"),
                    dcc.Checklist(id='corr-groups1', options=[], value=[], inline=True),
                    html.Br(),
                    html.Div(id='corr-subject-count1', className='text-muted small')
                ], width=6),
                
                # SET 2
                dbc.Col([
                    html.H5("Set 2 Subjects", className="text-success"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Session"),
                            dcc.Dropdown(
                                id='corr-session2', 
                                options=[], 
                                value='V1',
                                placeholder="Select session for Set 2"
                            )
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Sex Filter"),
                            dcc.Dropdown(
                                id='corr-sex2', 
                                options=[], 
                                value='all',
                                placeholder="Select sex for Set 2"
                            )
                        ], width=6)
                    ]),
                    dbc.Label("Subject Groups"),
                    dcc.Checklist(id='corr-groups2', options=[], value=[], inline=True),
                    html.Br(),
                    html.Div(id='corr-subject-count2', className='text-muted small')
                ], width=6)
            ])
        ])
    ], className="mb-4"),
    
    # Sélection des variables par sets
    dbc.Card([
        dbc.CardHeader("Variable Selection by Sets"),
        dbc.CardBody([
            dbc.Row([
                # SET 1
                dbc.Col([
                    html.H5("Set 1 Variables", className="text-primary"),
                    dbc.Label("Variable Type for Set 1"),
                    dcc.RadioItems(
                        id='corr-system-type1',
                        options=[
                            {'label': 'Synaptic Ratios', 'value': 'Synaptic ratio'},
                            {'label': 'Neurotransmitter (Lesion)', 'value': 'Neurotransmitter (Loc)'},
                            {'label': 'Neurotransmitter (Tract)', 'value': 'Neurotransmitter (Tract)'},
                            {'label': 'Clinical Outcomes', 'value': 'Clinical Outcomes'}
                        ],
                        value='Synaptic ratio',
                        inline=False
                    )
                ], width=6),
                
                # SET 2
                dbc.Col([
                    html.H5("Set 2 Variables", className="text-success"),
                    dbc.Label("Variable Type for Set 2"),
                    dcc.RadioItems(
                        id='corr-system-type2',
                        options=[
                            {'label': 'Synaptic Ratios', 'value': 'Synaptic ratio'},
                            {'label': 'Neurotransmitter (Lesion)', 'value': 'Neurotransmitter (Loc)'},
                            {'label': 'Neurotransmitter (Tract)', 'value': 'Neurotransmitter (Tract)'},
                            {'label': 'Clinical Outcomes', 'value': 'Clinical Outcomes'}
                        ],
                        value='Clinical Outcomes',
                        inline=False
                    )
                ], width=6)
            ])
        ])
    ], className="mb-4"),
        
        # Bouton d'exécution
        dbc.Button(" Run Correlation Analysis", id='run-corr-analysis', color="info", className="mb-4"),
        
        # Résultats
        html.Div(id='corr-results-container')
    ], style={'display': 'none'}),

], fluid=True, style={'background-color': '#f8f8f0'})

# # ==================== CALLBACKS ====================
@app.callback(
    Output('master-store', 'data'),
    Input('data-source', 'value')
)
#changer le path avant exportation github
def load_master_data(source):
    if source != 'master':
        raise dash.exceptions.PreventUpdate
    
    #changer le path avant exportation github
    path = "data/data_article.csv"
    try:
        df = pd.read_csv(path)
        data = df.to_dict('records')
        # Synchroniser avec le store global de l'API
        _global_data['master_data'] = data #modifier pour la communication
        return data
    except Exception as e:
        return None
    
# Callback for file upload
@app.callback(
    [Output('dataset1-store', 'data'),
     Output('dataset2-store', 'data'),
     Output('upload-status', 'children')],
    [Input('upload-data1', 'contents'),
     Input('upload-data2', 'contents')],
    [State('upload-data1', 'filename'),
     State('upload-data2', 'filename')]
)
def process_uploads(contents1, contents2, filename1, filename2):
    df1_data = None
    df2_data = None
    status_messages = []
    
    if contents1:
        df1 = process_zip_to_dataframe(contents1, filename1)
        if not df1.empty:
            df1_data = df1.to_dict('records')
            # Synchroniser avec l'API
            _global_data['dataset1'] = df1_data # modification avec notebook
            status_messages.append(
                dbc.Alert(f"✅ Dataset 1 processed successfully! ({len(df1)} subjects)", color="success")
            )
        else:
            status_messages.append(
                dbc.Alert("⚠️ Dataset 1 processing failed", color="warning")
            )
    
    if contents2:
        df2 = process_zip_to_dataframe(contents2, filename2)
        if not df2.empty:
            df2_data = df2.to_dict('records')
            # Synchroniser avec l'API
            _global_data['dataset2'] = df2_data #modification avec notebook
            status_messages.append(
                dbc.Alert(f"✅ Dataset 2 processed successfully! ({len(df2)} subjects)", color="success")
            )
        else:
            status_messages.append(
                dbc.Alert("⚠️ Dataset 2 processing failed", color="warning")
            )
    
    return df1_data, df2_data, html.Div(status_messages)

def create_visualization_tab(df1, df2):
    """Onglet Visualisation - SEULEMENT les graphiques et options d'affichage"""
    return html.Div([
        html.H3(" Visualization Results"),
        
        # Graphiques (les mêmes IDs que dans votre layout principal)
        dbc.Row([
            dbc.Col(dcc.Graph(id='graph1', 
                             figure={}, 
                             config={'displayModeBar': True}), width=4),
            dbc.Col(dcc.Graph(id='graph2', 
                             figure={}, 
                             config={'displayModeBar': True}), width=4),
            dbc.Col(dcc.Graph(id='graph3', 
                             figure={}, 
                             config={'displayModeBar': True}), width=4)
        ]),
        
        html.Hr(),
        html.H4("Display Options"),
        
        # Options d'affichage spécifiques à la visualisation
        dbc.Row([
            dbc.Col([
                html.Label("Show Data Table:"),
                dcc.RadioItems(
                    id='show-data',
                    options=[
                        {'label': 'Yes', 'value': 'yes'},
                        {'label': 'No', 'value': 'no'}
                    ],
                    value='no',
                    inline=True
                )
            ], width=4),
         ]),
         html.Div(id='data-table-container') #modifier
      ])

def create_statistics_tab(df1, df2):
    """Onglet Statistiques avec sélection des groupes de sujets"""
    return html.Div([

        html.H3(" Statistical Analysis"),

        # Conteneur qui recevra les éléments statistiques
        html.Div(id='stats-method-container', className="mb-3"),
        html.Div(id='stats-dataset-container',className="mb-3"),
        html.Div(id='stats-analysis-type-container',className="mb-3"),

        # Contrôles dynamiques pour la sélection des sujets
        html.Div(id='stats-subject-selection'),# className="mb-3"),
        
        # Contrôles spécifiques à la méthode
        html.Div(id='stats-method-controls'), #, className="mb-3"),
        
        # Résultats
        html.Div(id='stats-results') #, className="mt-3")
    ])

def create_data_explorer_tab(df1, df2):
    """Onglet Explorateur de données"""
    return html.Div([
        html.H3("📋 Data Explorer"),
        
        html.Label("Select Dataset to Explore:"),
        dcc.RadioItems(
            id='data-explorer-dataset',
            options=[
                {'label': f'Dataset 1 ({len(df1)} subjects)' if df1 is not None else 'Dataset 1 (N/A)', 
                 'value': 'dataset1', 'disabled': df1 is None},
                {'label': f'Dataset 2 ({len(df2)} subjects)' if df2 is not None else 'Dataset 2 (N/A)', 
                 'value': 'dataset2', 'disabled': df2 is None}
            ],
            value='dataset1' if df1 is not None else 'dataset2',
            inline=True
        ),
        html.Div(id='data-explorer-content')
    ])

# Callback pour gérer le contenu des tabs
@app.callback(
    Output('tab-content', 'children'),
    [Input('tabs', 'active_tab'),
     Input('dataset1-store', 'data'),
     Input('dataset2-store', 'data')]
)
def render_tab_content(active_tab, data1, data2):
    """Affiche le contenu selon l'onglet actif"""
    if not data1 and not data2:
        return html.Div([
            dbc.Alert("Please upload data first to access this functionality.", 
                     color="info", className="text-center")
        ])
    
    df1 = pd.DataFrame(data1) if data1 else None
    df2 = pd.DataFrame(data2) if data2 else None
    
    if active_tab == "tab-visualization":
        return create_visualization_tab(df1, df2)
    elif active_tab == "tab-stats":
        return create_statistics_tab(df1, df2)
    elif active_tab == "tab-data":
        return create_data_explorer_tab(df1, df2)
    
    return html.Div("Select a tab to view content")

# Callback maître pour contrôler l'affichage des sections
@app.callback(
    [Output('main-configuration-container', 'style'),
     Output('enable-overlay-container', 'style'),
     Output('overlay-options-container', 'style')],
    [Input('tabs', 'active_tab'),
     Input('enable-overlay', 'value')]
)
def master_display_control(active_tab, overlay_enabled):
    """Affiche/masque les sections selon l'onglet actif"""
    
    if active_tab == 'tab-visualization':
        # Afficher la configuration principale
        main_style = {'display': 'block'}
        
        # Afficher enable-overlay
        overlay_container_style = {'display': 'block', 'margin-top': '20px'}
        
        # Afficher/masquer les options overlay
        if overlay_enabled == 'yes':
            overlay_options_style = {'display': 'block', 'background-color': '#f8f9fa', 'padding': '20px'}
        else:
            overlay_options_style = {'display': 'none'}
            
        return main_style, overlay_container_style, overlay_options_style
    
    else:
        # Masquer tout pour les autres onglets
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}


# Callback pour afficher/masquer les conteneurs 
@app.callback(
    [Output('subject-selection-container', 'style'),
     Output('session-filter-container', 'style'), 
     Output('sex-filter-container', 'style'),
     Output('group-selection-container', 'style')],
    [Input('analysis-type', 'value')]
)
def toggle_selection_containers(analysis_type):
    """Affiche/masque les conteneurs selon le type d'analyse"""
    if analysis_type == 'single':
        return (
            {'display': 'block'},  # subject-selection visible
            {'display': 'none'},   # session masqué
            {'display': 'none'},   # sex masqué  
            {'display': 'none'}    # groups masqué
        )
    elif analysis_type == 'session_sex':
        return (
            {'display': 'none'},   # subject masqué
            {'display': 'block'},  # session visible
            {'display': 'block'},  # sex visible
            {'display': 'block'}   # groups visible
        )
    else:
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}

# Callback pour mettre à jour les OPTIONS des dropdowns (pas recréer les composants)
@app.callback(
    [Output('subject-dropdown', 'options'),
     Output('subject-dropdown', 'value'),
     Output('session-dropdown', 'options'), 
     Output('session-dropdown', 'value'),
     Output('sex-filter', 'options'),
     Output('sex-filter', 'value'),
     Output('group-checklist', 'options'),
     Output('group-checklist', 'value')],
    [Input('dataset-selector', 'value'),
     Input('analysis-type', 'value')],
    [State('dataset1-store', 'data'),
     State('dataset2-store', 'data'),
     State('data-source', 'value'),
     State('master-store', 'data')]
)
def update_all_options(dataset_sel, analysis_type, data1, data2, data_source, master_store):
    """Met à jour toutes les options des composants selon le dataset sélectionné"""
    
    # Sélectionner le dataset approprié
    data = get_dataset(dataset_sel, data1, data2, master_store, data_source)
    # if data_source == 'master':
    #     data = master_store
    # else:
    #     data = data1 if dataset_sel == 'dataset1' else data2
    
    if not data:
        return [], None, [], None, [], None, [], []
    
    df = pd.DataFrame(data)
    
    # --- OPTIONS SUJETS (pour mode single) ---
    subjects = sorted(df['subject'].unique())
    subject_options = [{'label': s, 'value': s} for s in subjects]
    subject_value = subjects[0] if subjects else None
    
    # --- OPTIONS SESSIONS ---
    all_sessions = sorted({re.search(r'_ses-V(\d+)', subj).group(1) 
                          for subj in df['subject'] if re.search(r'_ses-V(\d+)', subj)})
    session_options = [{'label': f"V{ses}", 'value': f"V{ses}"} for ses in all_sessions]
    session_value = session_options[0]['value'] if session_options else None
    
    # --- OPTIONS SEX ---
    sex_options = [
        {'label': 'All', 'value': 'all'},
        {'label': 'Men only', 'value': 'men'}, 
        {'label': 'Women only', 'value': 'women'}
    ]
    sex_value = 'all'
    
    # --- OPTIONS GROUPES (tous disponibles dans le dataset) ---
    all_subjects = df['subject'].unique()
    all_groups = sorted({detect_group(subj) for subj in all_subjects})
    group_options = [{'label': group, 'value': group} for group in all_groups if group != 'Unknown']
    group_values = [group['value'] for group in group_options]  # Tous sélectionnés par défaut
    
    return (
        subject_options, subject_value,  # subject dropdown
        session_options, session_value,  # session dropdown  
        sex_options, sex_value,          # sex filter
        group_options, group_values      # group checklist
    )

# Callback pour filtrer les groupes selon session/sex ET compter les sujets
@app.callback(
    [Output('group-checklist', 'options', allow_duplicate=True),
     Output('subject-count-display', 'children')],
    [Input('session-dropdown', 'value'),
     Input('sex-filter', 'value'),
     Input('group-checklist', 'value'),
     Input('subject-dropdown', 'value'),
     Input('analysis-type', 'value')],
    [State('dataset-selector', 'value'),
     State('dataset1-store', 'data'),
     State('dataset2-store', 'data'),
     State('data-source', 'value'),
     State('master-store', 'data')],
    prevent_initial_call=True
)
def update_groups_and_count(session, sex_filter, selected_groups, selected_subject, 
                           analysis_type, dataset_sel, data1, data2, data_source, master_store):
    """Filtre les options de groupes selon session/sex ET compte les sujets"""
    
    # Sélectionner le dataset approprié
    data = get_dataset(dataset_sel, data1, data2, master_store, data_source)
    # if data_source == 'master':
    #     data = master_store
    # else:
    #     data = data1 if dataset_sel == 'dataset1' else data2
    
    if not data:
        return [], "0 subjects selected"
    
    df = pd.DataFrame(data)
    
    # Mode single subject
    if analysis_type == 'single':
        count = "1 subject selected" if selected_subject else "0 subjects selected"
        # Retourner les options de groupes existantes (pas de changement nécessaire)
        all_subjects = df['subject'].unique()
        all_groups = sorted({detect_group(subj) for subj in all_subjects})
        group_options = [{'label': group, 'value': group} for group in all_groups if group != 'Unknown']
        return group_options, count
    
    # Mode session_sex
    elif analysis_type == 'session_sex':
        if not session:
            return [], "Please select a session"
        
        # 1. Filtrer par session
        session_subjects = df[df['subject'].str.contains(f"_ses-{session}")]['subject'].tolist()
        
        # 2. Filtrer par sexe
        if sex_filter != 'all' and 'sex' in df.columns:
            gender = "M" if sex_filter == 'men' else "F"
            session_subjects = df[
                (df['subject'].isin(session_subjects)) & 
                (df['sex'] == gender)
            ]['subject'].tolist()
        
        # 3. Obtenir les groupes disponibles pour cette sélection
        available_groups = sorted({detect_group(subj) for subj in session_subjects})
        group_options = [{'label': group, 'value': group} for group in available_groups if group != 'Unknown']
        
        # 4. Compter les sujets avec les filtres de groupes appliqués
        if selected_groups:
            filtered_subjects = [s for s in session_subjects if detect_group(s) in selected_groups]
        else:
            filtered_subjects = session_subjects
        
        count = f"{len(filtered_subjects)} subjects selected"
        
        return group_options, count
    
    return [], "0 subjects selected"

#Callback principal pour générer les graphiques de base
@app.callback(
    [Output('plots-store', 'data'),
     Output('profile-title', 'children')],
    [Input('generate-profile-btn', 'n_clicks')],
    [State('analysis-type', 'value'),
     State('dataset-selector', 'value'),
     State('dataset1-store', 'data'),
     State('dataset2-store', 'data'),
     State('subject-dropdown', 'value'),
     State('session-dropdown', 'value'),
     State('sex-filter', 'value'),
     State('group-checklist', 'value'),
     State('data-source', 'value'),
     State('master-store', 'data')],
    prevent_initial_call=True
)
def generate_plots(n_clicks, analysis_type, dataset_sel, data1, data2,
                   selected_subject, session, sex_filter, selected_groups, 
                   data_source, master_store):
    """Génère les graphiques selon les paramètres sélectionnés"""
    
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    # Sélectionner le dataset approprié
    data = get_dataset(dataset_sel, data1, data2, master_store, data_source)
    # if data_source == 'master':
    #     data = master_store
    # else:
    #     data = data1 if dataset_sel == 'dataset1' else data2
    
    if not data:
        return None, "Error: No data available"
    
    df = pd.DataFrame(data)
    
    try:
        if analysis_type == 'single':
            # Mode sujet unique
            if not selected_subject:
                return None, "Error: Please select a subject"
            
            subjects = [selected_subject]
            title = f"{selected_subject}"
            is_group = False
            
        elif analysis_type == 'session_sex':
            # Mode session & sexe
            if not session:
                return None, "Error: Please select a session"
            
            # Filtrer par session
            session_subjects = df[df['subject'].str.contains(f"_ses-{session}")]['subject'].tolist()
            
            # Filtrer par sexe
            if sex_filter != 'all' and 'sex' in df.columns:
                gender = "M" if sex_filter == 'men' else "F"
                session_subjects = df[
                    (df['subject'].isin(session_subjects)) & 
                    (df['sex'] == gender)
                ]['subject'].tolist()
            
            # Filtrer par groupes
            if selected_groups:
                subjects = [s for s in session_subjects if detect_group(s) in selected_groups]
            else:
                subjects = session_subjects
            
            if not subjects:
                return None, "Error: No subjects found for these criteria"
            
            # Construire le titre
            title = f"Session {session}"
            if sex_filter != 'all':
                title += f" ({'Men' if sex_filter == 'men' else 'Women'})"
            if selected_groups:
                title += f" | Groups: {', '.join(selected_groups)}"
            
            is_group = True
        
        else:
            return None, "Error: Unknown analysis type"
        
        # Générer les graphiques
        fig1, fig2, fig3 = create_interactive_plots(df, subjects, title, is_group=is_group)
        
        # Préparer les données pour le store
        plots_data = {
            'fig1': fig1.to_dict(),
            'fig2': fig2.to_dict(),
            'fig3': fig3.to_dict(),
            'subjects': subjects,
            'title': title
        }
        
        return plots_data, title
        
    except Exception as e:
        print(f"Error generating plots: {e}")
        return None, f"Error: {str(e)}"


# Callback to update graphs from store
@app.callback(
    [Output('graph1', 'figure'),
     Output('graph2', 'figure'),
     Output('graph3', 'figure')],
    [Input('plots-store', 'data')],
    prevent_initial_call=True
)
def update_graphs_from_store(plots_data):
    if not plots_data:
        raise dash.exceptions.PreventUpdate

    return (
        go.Figure(plots_data['fig1']),
        go.Figure(plots_data['fig2']),
        go.Figure(plots_data['fig3'])
    )


# ====================== Overlay ======================================== 
    
#Callback pour afficher/masquer les options d'overlay
@app.callback(
    [#Output('overlay-options-container', 'style'),
     Output('overlay-dataset-selector', 'style'),
     Output('overlay-analysis-type', 'style')],
    Input('enable-overlay', 'value')
)
def toggle_overlay_container(enable_overlay):
    """Affiche/masque tout le contenu overlay"""
    if enable_overlay == 'yes':
        return (
            #{'display': 'block', 'background-color': '#f8f9fa', 'padding': '20px', 'border-radius': '10px', 'margin-top': '20px'},
            {'display': 'block'},  # Afficher dataset selector
            {'display': 'block'}   # Afficher analysis type
        )
    return (
        #{'display': 'none'},  # Masquer container principal
        {'display': 'none'},  # Masquer dataset selector  
        {'display': 'none'}   # Masquer analysis type
    )

# Callback pour afficher/masquer les conteneurs d'overlay selon le type
@app.callback(
    [Output('overlay-subject-container', 'style'),
     Output('overlay-session-container', 'style'),
     Output('overlay-sex-container', 'style'),
     Output('overlay-groups-container', 'style')],
    [Input('overlay-analysis-type', 'value'),
     Input('enable-overlay', 'value')]
)
def toggle_overlay_selection_containers(analysis_type, enable_overlay):
    """Affiche/masque les conteneurs d'overlay selon le type d'analyse ET l'état enable"""
    
    # Si overlay n'est pas activé, tout masquer
    if enable_overlay != 'yes':
        return (
            {'display': 'none'}, 
            {'display': 'none'}, 
            {'display': 'none'}, 
            {'display': 'none'}
        )
    
    # Sinon, afficher selon le type d'analyse
    if analysis_type == 'single':
        return (
            {'display': 'block'},  # subject visible
            {'display': 'none'},   # session masqué
            {'display': 'none'},   # sex masqué
            {'display': 'none'}    # groups masqué
        )
    elif analysis_type == 'session_sex':
        return (
            {'display': 'none'},   # subject masqué
            {'display': 'block'},  # session visible
            {'display': 'block'},  # sex visible
            {'display': 'block'}   # groups visible
        )
    else:
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}

# Callback pour mettre à jour les options des composants d'overlay
@app.callback(
    [Output('overlay-subject-dropdown', 'options'),
     Output('overlay-subject-dropdown', 'value'),
     Output('overlay-session-dropdown', 'options'),
     Output('overlay-session-dropdown', 'value'),
     Output('overlay-sex-filter', 'options'),
     Output('overlay-sex-filter', 'value'),
     Output('overlay-group-checklist', 'options'),
     Output('overlay-group-checklist', 'value')],
    [Input('overlay-dataset-selector', 'value'),
     Input('overlay-analysis-type', 'value'),
     Input('enable-overlay', 'value')],
    [State('dataset1-store', 'data'),
     State('dataset2-store', 'data'),
     State('data-source', 'value'),
     State('master-store', 'data')]
)
def update_overlay_options(dataset_sel, analysis_type, enable_overlay, data1, data2, data_source, master_store):
    """Met à jour les options des composants d'overlay"""
    
    # Si overlay n'est pas activé, retourner des valeurs vides
    if enable_overlay != 'yes':
        return [], None, [], None, [], None, [], []
    
    # Sélectionner le dataset approprié
    data = get_dataset(dataset_sel, data1, data2, master_store, data_source)
    # if data_source == 'master':
    #     data = master_store
    # else:
    #     data = data1 if dataset_sel == 'dataset1' else data2
    
    if not data:
        return [], None, [], None, [], None, [], []
    
    df = pd.DataFrame(data)
    
    # OPTIONS SUJETS
    subjects = sorted(df['subject'].unique())
    subject_options = [{'label': s, 'value': s} for s in subjects]
    subject_value = subjects[0] if subjects else None
    
    # OPTIONS SESSIONS
    all_sessions = sorted({re.search(r'_ses-V(\d+)', subj).group(1) 
                          for subj in df['subject'] if re.search(r'_ses-V(\d+)', subj)})
    session_options = [{'label': f"V{ses}", 'value': f"V{ses}"} for ses in all_sessions]
    session_value = session_options[0]['value'] if session_options else None
    
    # OPTIONS SEX
    sex_options = [
        {'label': 'All', 'value': 'all'},
        {'label': 'Men only', 'value': 'men'}, 
        {'label': 'Women only', 'value': 'women'}
    ]
    sex_value = 'all'
    
    # OPTIONS GROUPES
    all_subjects = df['subject'].unique()
    all_groups = sorted({detect_group(subj) for subj in all_subjects})
    group_options = [{'label': group, 'value': group} for group in all_groups if group != 'Unknown']
    group_values = [group['value'] for group in group_options][:3]  # Seulement quelques-uns par défaut
    
    return (
        subject_options, subject_value,
        session_options, session_value,
        sex_options, sex_value,
        group_options, group_values
    )

# Callback pour compter les sujets d'overlay
@app.callback(
    Output('overlay-subject-count', 'children'),
    [Input('overlay-session-dropdown', 'value'),
     Input('overlay-sex-filter', 'value'),
     Input('overlay-group-checklist', 'value'),
     Input('overlay-subject-dropdown', 'value'),
     Input('overlay-analysis-type', 'value')],
    [State('overlay-dataset-selector', 'value'),
     State('dataset1-store', 'data'),
     State('dataset2-store', 'data'),
     State('data-source', 'value'),
     State('master-store', 'data')]
)
def update_overlay_count(session, sex_filter, selected_groups, selected_subject,
                        analysis_type, dataset_sel, data1, data2, data_source, master_store):
    """Compte les sujets d'overlay"""
    
    # Sélectionner le dataset approprié
    data = get_dataset(dataset_sel, data1, data2, master_store, data_source)
    # if data_source == 'master':
    #     data = master_store
    # else:
    #     data = data1 if dataset_sel == 'dataset1' else data2
    
    if not data:
        return "0 overlay subjects"
    
    df = pd.DataFrame(data)
    
    if analysis_type == 'single':
        count = "1 overlay subject" if selected_subject else "0 overlay subjects"
        return count
    
    elif analysis_type == 'session_sex':
        if not session:
            return "Select session for overlay"
        
        # Filtrer par session
        session_subjects = df[df['subject'].str.contains(f"_ses-{session}")]['subject'].tolist()
        
        # Filtrer par sexe
        if sex_filter != 'all' and 'sex' in df.columns:
            gender = "M" if sex_filter == 'men' else "F"
            session_subjects = df[
                (df['subject'].isin(session_subjects)) & 
                (df['sex'] == gender)
            ]['subject'].tolist()
        
        # Filtrer par groupes
        if selected_groups:
            filtered_subjects = [s for s in session_subjects if detect_group(s) in selected_groups]
        else:
            filtered_subjects = session_subjects
        
        return f"{len(filtered_subjects)} overlay subjects"
    
    return "0 overlay subjects"

# Callback principal pour ajouter l'overlay
@app.callback(
    [Output('graph1', 'figure', allow_duplicate=True),
     Output('graph2', 'figure', allow_duplicate=True),
     Output('graph3', 'figure', allow_duplicate=True),
     Output('profile-title-overlay', 'children')],
    Input('add-overlay-btn', 'n_clicks'),
    [State('overlay-analysis-type', 'value'),
     State('overlay-dataset-selector', 'value'),
     State('overlay-subject-dropdown', 'value'),
     State('overlay-session-dropdown', 'value'),
     State('overlay-sex-filter', 'value'),
     State('overlay-group-checklist', 'value'),
     State('dataset1-store', 'data'),
     State('dataset2-store', 'data'),
     State('data-source', 'value'),
     State('master-store', 'data'),
     State('plots-store', 'data')],  # Graphiques de base
    prevent_initial_call=True
)
def add_overlay_to_plots(n_clicks, overlay_analysis_type, overlay_dataset_sel,
                        overlay_subject, overlay_session, overlay_sex_filter, overlay_groups,
                        data1, data2, data_source, master_store, base_plots_data):
    """Ajoute l'overlay aux graphiques existants"""
    
    if not n_clicks or not base_plots_data:
        raise dash.exceptions.PreventUpdate
    
    # Sélectionner le dataset d'overlay
    overlay_data = get_dataset(overlay_dataset_sel, data1, data2, master_store, data_source)
    # if data_source != 'master':
    #     overlay_data = master_store
    # else:
    #     overlay_data = data1 if overlay_dataset_sel == 'dataset1' else data2
    
    if not overlay_data:
        raise dash.exceptions.PreventUpdate
    
    df_overlay = pd.DataFrame(overlay_data)
    
    try:
        # Déterminer les sujets d'overlay selon le type
        if overlay_analysis_type == 'single':
            if not overlay_subject:
                raise dash.exceptions.PreventUpdate
            overlay_subjects = [overlay_subject]
            overlay_title = f"{overlay_subject}"
            is_group = False
            
        elif overlay_analysis_type == 'session_sex':
            if not overlay_session:
                raise dash.exceptions.PreventUpdate
            
            # Filtrer par session
            session_subjects = df_overlay[df_overlay['subject'].str.contains(f"_ses-{overlay_session}")]['subject'].tolist()
            
            # Filtrer par sexe
            if overlay_sex_filter != 'all' and 'sex' in df_overlay.columns:
                gender = "M" if overlay_sex_filter == 'men' else "F"
                session_subjects = df_overlay[
                    (df_overlay['subject'].isin(session_subjects)) & 
                    (df_overlay['sex'] == gender)
                ]['subject'].tolist()
            
            # Filtrer par groupes
            if overlay_groups:
                overlay_subjects = [s for s in session_subjects if detect_group(s) in overlay_groups]
            else:
                overlay_subjects = session_subjects
            
            if not overlay_subjects:
                raise dash.exceptions.PreventUpdate
            
            # Titre d'overlay
            overlay_title = f"Session {overlay_session}"
            if overlay_sex_filter != 'all':
                overlay_title += f" ({'Men' if overlay_sex_filter == 'men' else 'Women'})"
            if overlay_groups:
                overlay_title += f" | Groups: {', '.join(overlay_groups)}"
            
            is_group = True
        
        # Générer les graphiques d'overlay
        fig1_overlay, fig2_overlay, fig3_overlay = create_interactive_plots(
            df_overlay, overlay_subjects, overlay_title, 
            is_group=is_group, is_overlay=True
        )
        
        # Récupérer les graphiques de base
        fig1_base = go.Figure(base_plots_data['fig1'])
        fig2_base = go.Figure(base_plots_data['fig2'])
        fig3_base = go.Figure(base_plots_data['fig3'])
        
        # Ajouter les traces d'overlay aux graphiques de base
        for trace in fig1_overlay.data:
            fig1_base.add_trace(trace)
        for trace in fig2_overlay.data:
            fig2_base.add_trace(trace)
        for trace in fig3_overlay.data:
            fig3_base.add_trace(trace)
        
        # Configurer pour l'affichage superposé
        fig1_base.update_layout(barmode='overlay')
        fig2_base.update_layout(barmode='overlay')
        fig3_base.update_layout(barmode='overlay')
        
        return fig1_base, fig2_base, fig3_base, overlay_title
        
    except Exception as e:
        print(f"Overlay error: {e}")
        # Retourner les graphiques de base en cas d'erreur
        return (
            go.Figure(base_plots_data['fig1']),
            go.Figure(base_plots_data['fig2']),
            go.Figure(base_plots_data['fig3']),
            "Overlay error"
        )

# Callback pour effacer les overlays
@app.callback(
    [Output('graph1', 'figure', allow_duplicate=True),
     Output('graph2', 'figure', allow_duplicate=True),
     Output('graph3', 'figure', allow_duplicate=True),
     Output('profile-title-overlay', 'children', allow_duplicate=True)],
    Input('clear-overlay-btn', 'n_clicks'),
    State('plots-store', 'data'),
    prevent_initial_call=True
)
def clear_overlays(n_clicks, base_plots_data):
    """Efface tous les overlays et retourne aux graphiques de base"""
    
    if not n_clicks or not base_plots_data:
        raise dash.exceptions.PreventUpdate
    
    return (
        go.Figure(base_plots_data['fig1']),
        go.Figure(base_plots_data['fig2']),
        go.Figure(base_plots_data['fig3']),
        "No overlay"
    )


# #====================== Show selected data to generate graphs ========================================
# Callback to show data table
@app.callback(
    Output('data-table-container', 'children'),
    [Input('show-data', 'value'),
     Input('plots-store', 'data')],
    [State('dataset1-store', 'data'),
     State('dataset2-store', 'data')]
)
def show_data_table(show_data, plots_data, data1, data2):
    if show_data != 'yes' or not plots_data or 'subjects' not in plots_data:
        return html.Div()
    
    # Find which dataset contains the subjects
    subjects = plots_data['subjects']
    df1 = pd.DataFrame(data1) if data1 else pd.DataFrame()
    df2 = pd.DataFrame(data2) if data2 else pd.DataFrame()
    
    # Check which dataset has the subjects
    if not df1.empty and any(subj in df1['subject'].values for subj in subjects):
        df = df1[df1['subject'].isin(subjects)]
    elif not df2.empty and any(subj in df2['subject'].values for subj in subjects):
        df = df2[df2['subject'].isin(subjects)]
    else:
        return html.Div("No data available for selected subjects")
    
    return html.Div([
        html.H5("Selected Subjects Data"),
        dash.dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': col, 'id': col} for col in df.columns],
            page_size=10,
            style_table={'overflowX': 'auto'}
        )
    ])



#================================= Statitics tab =========================================================
# Callback pour afficher les éléments statistiques dans l'onglet Statistics
@app.callback(
    [Output('stats-method-container', 'style'),
     Output('stats-dataset-container', 'style'),
     Output('stats-analysis-type-container', 'style')],
    Input('tabs', 'active_tab')
)
def toggle_stats_visibility(active_tab):
    """Contrôle la visibilité des éléments statistiques"""
    # if active_tab != "tab-stats":
    #     raise PreventUpdate
    # else : 
    #    return {'display': 'block'}, {'display': 'block'}, {'display': 'block'}
    if active_tab == "tab-stats":
        # Afficher dans l'onglet Statistics
        return {'display': 'block'}, {'display': 'block'}, {'display': 'block'}
    else:
        # Cacher dans les autres onglets
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}

# Callback principal pour afficher/masquer les sections selon la méthode sélectionnée
@app.callback(
    [Output('glm-analysis-container', 'style'),
     Output('ttest-analysis-container', 'style'), 
     Output('correlation-analysis-container', 'style')],
    [Input('tabs', 'active_tab'),
     Input('stats-method', 'value')]
)
def toggle_stats_containers(active_tab, method):
    """Affiche/masque les conteneurs selon l'onglet ET la méthode statistique"""
    
    # Masquer tout si on n'est pas dans l'onglet stats
    if active_tab != "tab-stats":
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
    
    # Afficher seulement la méthode sélectionnée dans l'onglet stats
    glm_style = {'display': 'block'} if method == 'glm' else {'display': 'none'}
    ttest_style = {'display': 'block'} if method == 'ttest' else {'display': 'none'}
    corr_style = {'display': 'block'} if method == 'correlation' else {'display': 'none'}
    
    return glm_style, ttest_style, corr_style

# Callback pour mettre à jour les options de base (sessions, sex, groupes) pour toutes les méthodes
@app.callback(
    [# GLM outputs 
     Output('glm-session', 'options'),
     Output('glm-sex', 'options'),
     Output('glm-groups', 'options'),
     Output('glm-interaction-var', 'options'),
     Output('glm-outcomes', 'options'),
     Output('glm-covariates', 'options'),
     # T-Test outputs
     Output('ttest-group1-session', 'options'),
     Output('ttest-group1-sex', 'options'),
     Output('ttest-group1-groups', 'options'),
     Output('ttest-group2-session', 'options'),
     Output('ttest-group2-sex', 'options'),
     Output('ttest-group2-groups', 'options'),
     Output('ttest-variables', 'options'),
     # Correlation outputs
     Output('corr-session1', 'options'),
     Output('corr-sex1', 'options'),
     Output('corr-groups1', 'options'),
     Output('corr-session2', 'options'),
     Output('corr-sex2', 'options'),
     Output('corr-groups2', 'options')],
    [Input('stats-method', 'value'),
     Input('tabs', 'active_tab')],
    [State('dataset1-store', 'data'),
     State('dataset2-store', 'data'),
     State('master-store', 'data')]
)
def update_stats_options(method, active_tab, data1, data2, master_store):
    """Met à jour les options communes pour toutes les méthodes statistiques"""
    
    # Si on n'est pas dans l'onglet stats, retourner des options vides
    if active_tab != "tab-stats":
        empty_options = []
        return [empty_options] * 19
    
    # Utiliser les données disponibles (priorité à master si disponible)
    data = master_store if master_store else (data1 if data1 else data2)
    
    if not data:
        empty_options = []
        return [empty_options] * 19  # 19 outputs
    
    df = pd.DataFrame(data)
    
    # Options communes
    session_options = [
        {'label': 'V1', 'value': 'V1'},
        {'label': 'V2', 'value': 'V2'},
        {'label': 'V3', 'value': 'V3'}
    ]
    
    sex_options = [
        {'label': 'All', 'value': 'all'},
        {'label': 'Men only', 'value': 'men'},
        {'label': 'Women only', 'value': 'women'}
    ]
    
    # Groupes disponibles
    all_subjects = df['subject'].unique()
    all_groups = sorted({detect_group(subj) for subj in all_subjects})
    group_options = [{'label': group, 'value': group} for group in all_groups if group != 'Unknown']
    
    # Variables pour GLM et T-Test
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Filtrer pour garder seulement les variables pertinentes
    relevant_vars = [col for col in numeric_cols if any(x in col.lower() for x in 
                    ['pre_', 'post_', 'loc_inj', 'tract_inj', 'score', 'volume'])]
    
    var_options = [{'label': col, 'value': col} for col in relevant_vars]
    
    # Variables catégorielles pour interactions GLM
    categorical_vars = ['sex'] + [col for col in df.columns if df[col].dtype == 'object' and col != 'subject']
    interaction_options = [{'label': col, 'value': col} for col in categorical_vars if col in df.columns]
    
    return (
        # GLM
        session_options,      # glm-session
        sex_options,          # glm-sex
        group_options,        # glm-groups
        interaction_options,  # glm-interaction-var options
        var_options,          # glm-outcomes options  
        var_options,          # glm-covariates options
        # T-Test Group 1
        session_options,      # ttest-group1-session
        sex_options,          # ttest-group1-sex
        group_options,        # ttest-group1-groups
        # T-Test Group 2
        session_options,      # ttest-group2-session
        sex_options,          # ttest-group2-sex
        group_options,        # ttest-group2-groups
        var_options,          # ttest-variables
        # Correlation SET 1
        session_options,      # corr-session1
        sex_options,          # corr-sex1
        group_options,        # corr-groups1
        # Correlation SET 2  
        session_options,      # corr-session2
        sex_options,          # corr-sex2
        group_options         # corr-groups2
    )

#Callback pour compter les sujets GLM
@app.callback(
    Output('glm-subject-count', 'children'),
    [Input('glm-session', 'value'),
     Input('glm-sex', 'value'),
     Input('glm-groups', 'value'),
     Input('stats-dataset', 'value')],
    [State('dataset1-store', 'data'),
     State('dataset2-store', 'data'),
     State('master-store', 'data')]
)
def update_glm_subject_count(session, sex_filter, groups, dataset, data1, data2, master_store):
    """Compte les sujets pour GLM"""
    
    # Sélectionner les données
    if dataset == 'master-store':
        data = master_store
    elif dataset == 'dataset1':
        data = data1
    else:
        data = data2
    
    if not data or not session:
        return "0 subjects selected"
    
    df = pd.DataFrame(data)
    
    # Filtrer par session
    subjects = df[df['subject'].str.contains(f"_ses-{session}")]['subject'].tolist()
    
    # Filtrer par sexe
    if sex_filter != 'all' and 'sex' in df.columns:
        gender = "M" if sex_filter == 'men' else "F"
        subjects = df[
            (df['subject'].isin(subjects)) & 
            (df['sex'] == gender)
        ]['subject'].tolist()
    
    # Filtrer par groupes
    if groups:
        subjects = [s for s in subjects if detect_group(s) in groups]
    
    count = len(subjects)
    color = "text-success" if count >= 3 else "text-danger"
    
    return html.Span(f"{count} subjects selected", className=color)


# Callback pour compter les sujets dans les groupes T-Test
@app.callback(
    [Output('ttest-group1-count', 'children'),
     Output('ttest-group2-count', 'children')],
    [Input('ttest-group1-session', 'value'),
     Input('ttest-group1-sex', 'value'),
     Input('ttest-group1-groups', 'value'),
     Input('ttest-group2-session', 'value'),
     Input('ttest-group2-sex', 'value'),
     Input('ttest-group2-groups', 'value'),
     Input('ttest-dataset', 'value')],
    [State('dataset1-store', 'data'),
     State('dataset2-store', 'data'),
     State('master-store', 'data')]
)
def update_ttest_group_counts(g1_session, g1_sex, g1_groups, g2_session, g2_sex, g2_groups, 
                             dataset, data1, data2, master_store):
    """Compte les sujets dans chaque groupe pour T-Test"""
    
    def count_subjects(session, sex_filter, groups, data):
        if not data or not session:
            return "0 subjects"
        
        df = pd.DataFrame(data)
        
        # Filtrer par session
        subjects = df[df['subject'].str.contains(f"_ses-{session}")]['subject'].tolist()
        
        # Filtrer par sexe
        if sex_filter != 'all' and 'sex' in df.columns:
            gender = "M" if sex_filter == 'men' else "F"
            subjects = df[
                (df['subject'].isin(subjects)) & 
                (df['sex'] == gender)
            ]['subject'].tolist()
        
        # Filtrer par groupes
        if groups:
            subjects = [s for s in subjects if detect_group(s) in groups]
        
        return f"{len(subjects)} subjects"
    
    # Sélectionner les données
    if dataset == 'master-store':
        data = master_store
    elif dataset == 'dataset1':
        data = data1
    else:
        data = data2
    
    g1_count = count_subjects(g1_session, g1_sex, g1_groups, data)
    g2_count = count_subjects(g2_session, g2_sex, g2_groups, data)
    
    return g1_count, g2_count


# Callback pour activer/désactiver le champ Tweedie power
@app.callback(
    Output('glm-tweedie-power', 'disabled'),
    Input('glm-distribution', 'value')
)
def toggle_tweedie_power(distribution):
    """Active/désactive le champ Tweedie power selon la distribution"""
    return distribution != 'Tweedie'

# Callback pour activer/désactiver la variable d'interaction
@app.callback(
    Output('glm-interaction-var', 'disabled'),
    Input('glm-interaction', 'value')
)
def toggle_interaction_var(interaction_enabled):
    """Active/désactive la sélection de variable d'interaction"""
    return not bool(interaction_enabled)


#================================= GLM analyses =================================

# Callback pour exécuter l'analyse GLM
@app.callback(
    Output('glm-results-container', 'children'),
    Input('run-glm-analysis', 'n_clicks'),
    [State('glm-session', 'value'),           
     State('glm-sex', 'value'),               
     State('glm-groups', 'value'),            
     State('stats-dataset', 'value'),        
     State('glm-distribution', 'value'),
     State('glm-link', 'value'),
     State('glm-tweedie-power', 'value'),
     State('glm-interaction', 'value'),
     State('glm-interaction-var', 'value'),
     State('glm-outcomes', 'value'),
     State('glm-covariates', 'value'),
     State('glm-predictor-system', 'value'),
     State('dataset1-store', 'data'),
     State('dataset2-store', 'data'),
     State('master-store', 'data')],
    prevent_initial_call=True
)
def run_glm_analysis_callback(n_clicks, session, sex_filter, groups, dataset,
                             distribution, link, tweedie_power, interaction, 
                             interaction_var, outcomes, covariates, predictor_system,
                             data1, data2, master_store):
    """Exécute l'analyse GLM avec filtrage des sujets"""
    
    if not n_clicks or not outcomes:
        return html.Div("Select outcomes and click Run GLM Analysis")
    
    # Sélectionner les données selon le dataset choisi
    if dataset == 'master-store':
        data = master_store
    elif dataset == 'dataset1':
        data = data1
    else:
        data = data2
    
    if not data:
        return dbc.Alert("No data available for analysis", color="danger")
    
    df = pd.DataFrame(data)
    
    # ========== FILTRER LES SUJETS ==========
    if not session:
        return dbc.Alert("Please select a session", color="warning")
    
    # Filtrer par session
    selected_subjects = df[df['subject'].str.contains(f"_ses-{session}")]['subject'].tolist()
    
    # Filtrer par sexe
    if sex_filter != 'all' and 'sex' in df.columns:
        gender = "M" if sex_filter == 'men' else "F"
        selected_subjects = df[
            (df['subject'].isin(selected_subjects)) & 
            (df['sex'] == gender)
        ]['subject'].tolist()
    
    # Filtrer par groupes
    if groups:
        selected_subjects = [s for s in selected_subjects if detect_group(s) in groups]
    
    # Vérifier le minimum
    if len(selected_subjects) < 3:
        return dbc.Alert(
            f"Insufficient subjects: {len(selected_subjects)} found (minimum 3 required)", 
            color="danger"
        )
    
    # Filtrer le DataFrame
    df = df[df['subject'].isin(selected_subjects)]
    
    try:
        # Configurer la famille et le lien
        family = get_family_and_link(distribution, link, tweedie_power)
        
        # Définir les systèmes de prédicteurs selon le choix
        if predictor_system == 'synaptic':
            pre_systems = ['A4B2', 'M1', 'D1', 'D2', '5HT1a', '5HT1b', '5HT2a', '5HT4', '5HT6']
            post_systems = ['VAChT', 'DAT', '5HTT']
            predictors = [f'pre_{sys}' for sys in pre_systems] + [f'post_{sys}' for sys in post_systems]
            systems = {'Synaptic': predictors}
        elif predictor_system == 'nt_loc':
            loc_systems = [col.replace('loc_inj_perc_', '') for col in df.columns 
                          if col.startswith('loc_inj_perc_')]
            predictors = [f'loc_inj_perc_{sys}' for sys in loc_systems]
            systems = {'Neurotransmitter_Loc': predictors}
        else:  # nt_tract
            tract_systems = [col.replace('tract_inj_perc_', '') for col in df.columns 
                            if col.startswith('tract_inj_perc_')]
            predictors = [f'tract_inj_perc_{sys}' for sys in tract_systems]
            systems = {'Neurotransmitter_Tract': predictors}
        
        # Préparer les DataFrames
        df_predictors = df[['subject'] + [p for p in predictors if p in df.columns]]
        df_outcomes = df[['subject'] + [o for o in outcomes if o in df.columns]]
        
        # Vérifier que les outcomes existent
        missing_outcomes = [o for o in outcomes if o not in df.columns]
        if missing_outcomes:
            return dbc.Alert(
                f"Missing outcomes in data: {', '.join(missing_outcomes)}", 
                color="warning"
            )
        
        # Ajouter les covariates si spécifiées
        if covariates:
            available_covariates = [c for c in covariates if c in df.columns]
            if available_covariates:
                df_predictors = df_predictors.merge(
                    df[['subject'] + available_covariates], 
                    on='subject', 
                    how='left'
                )
        else:
            available_covariates = []
        
        # Variable d'interaction
        interaction_var_to_use = interaction_var if interaction and interaction_var else None
        
        # Exécuter l'analyse GLM
        results_df, error_messages = run_glm_analysis(
            df_predictors=df_predictors,
            df_outcomes=df_outcomes,
            outcomes=outcomes,
            systems=systems,
            covariate=available_covariates,
            visit_name=f"{session} ({len(selected_subjects)} subjects)",
            family=family,
            interaction_var=interaction_var_to_use
        )
        
        # Créer l'affichage des erreurs
        error_display = html.Div()
        if error_messages:
            error_display = html.Div([
                dbc.Alert([
                    html.H5("⚠️ Error/Warning Messages"),
                    html.P(f"{len(error_messages)} problem(s) encountered during analysis"),
                    html.Details([
                        html.Summary("See details below"),
                        html.Ul([html.Li(msg) for msg in error_messages])
                    ])
                ], color="warning")
            ], className="mb-4")

        if results_df.empty:
            return dbc.Alert(
                "No valid results from GLM analysis. Check console for details.", 
                color="warning"
            )
        
        # Formater les résultats
        results_display = results_df.round(4)

        # Créer les visualisations des résultats
        results_viz = create_glm_results_visualizations(results_df)
        
        return html.Div([
            dbc.Alert(
                f"✓ GLM Analysis completed! {len(selected_subjects)} subjects, {len(results_df)} results found.", 
                color="success"
            ),
            # Résumé
            dbc.Card([
                dbc.CardHeader("Analysis Summary"),
                dbc.CardBody([
                    html.P(f"Session: {session} | Sex filter: {sex_filter} | Groups: {groups if groups else 'All'}"),
                    html.P(f"Distribution: {distribution}, Link: {link}"),
                    html.P(f"Significant results: {sum(results_df['Significant'])} / {len(results_df)}"),
                    html.P(f"Subjects analyzed: {len(selected_subjects)}")
                ])
            ], className="mb-4"),

            # Onglets pour résultats et visualisations
            dbc.Tabs([
                dbc.Tab(
                    label="📋 Detailed Results",
                    children=[
                        dash_table.DataTable(
                            data=results_display.to_dict('records'),
                            columns=[
                                {'name': col, 'id': col, 'type': 'numeric', 'format': {'specifier': '.4f'}} 
                                if col in ['Coefficient', 'P-value', 'Pseudo_R2_CS'] 
                                else {'name': col, 'id': col} 
                                for col in results_display.columns
                            ],
                            sort_action="native",
                            filter_action="native",
                            page_size=15,
                            style_table={'overflowX': 'auto'},
                            style_cell={'textAlign': 'left', 'padding': '10px'},
                            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                            style_data_conditional=[
                                {
                                    'if': {'filter_query': '{Significant} = true'},
                                    'backgroundColor': '#d4edda',
                                    'color': 'black',
                                }
                            ]
                        )
                    ]
                ),
                
                dbc.Tab(
                    label="📊 Results Visualization", 
                    children=results_viz
                )
            ])
        ])
            
        #     html.H5("Detailed Results"),
        #     dash_table.DataTable(
        #         data=results_display.to_dict('records'),
        #         columns=[
        #             {'name': col, 'id': col, 'type': 'numeric', 'format': {'specifier': '.4f'}} 
        #             if col in ['Coefficient', 'P-value', 'Pseudo_R2_CS'] 
        #             else {'name': col, 'id': col} 
        #             for col in results_display.columns
        #         ],
        #         sort_action="native",
        #         filter_action="native",
        #         page_size=20,
        #         style_table={'overflowX': 'auto'},
        #         style_cell={'textAlign': 'left', 'padding': '10px'},
        #         style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
        #         style_data_conditional=[
        #             {
        #                 'if': {'filter_query': '{Significant} = true'},
        #                 'backgroundColor': '#d4edda',
        #                 'color': 'black',
        #             }
        #         ]
        #     )
        # ])
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(error_details)  # Pour le debug
        return dbc.Alert(
            [
                html.H5("Error in GLM analysis"),
                html.P(str(e)),
                html.Pre(error_details, style={'fontSize': '10px', 'maxHeight': '200px', 'overflow': 'auto'})
            ], 
            color="danger"
        )

# Callback pour mettre à jour les options de visualisation GLM
@app.callback(
    [Output('glm-viz-variable', 'options'),
     Output('glm-viz-color', 'options')],
    [Input('glm-outcomes', 'value'),
     Input('glm-covariates', 'value'),
     Input('glm-predictor-system', 'value')],
    [State('dataset1-store', 'data'),
     State('dataset2-store', 'data'),
     State('master-store', 'data')]
)
def update_glm_viz_options(outcomes, covariates, predictor_system, data1, data2, master_store):
    """Met à jour les options de visualisation GLM"""
    
    # Utiliser les données disponibles
    data = master_store if master_store else (data1 if data1 else data2)
    
    if not data or not outcomes:
        return [], []
    
    df = pd.DataFrame(data)
    
    if outcomes:
        # Vérifier que les outcomes sélectionnés existent dans les données
        available_outcomes = [outcome for outcome in outcomes if outcome in df.columns]
        var_options = [{'label': outcome, 'value': outcome} for outcome in available_outcomes]
    else:
        # Si aucun outcome sélectionné, retourner liste vide
        var_options = []
    
    # Options de couleur
    color_options = [{'label': 'None', 'value': 'none'}]
    if 'sex' in df.columns:
        color_options.append({'label': 'Sex', 'value': 'sex'})
    if 'group' in df.columns:
        color_options.append({'label': 'Group', 'value': 'group'})
    
    return var_options, color_options

# Callback pour générer le graphique de visualisation
@app.callback(
    Output('glm-viz-plot', 'figure'),
    [Input('glm-viz-variable', 'value'),
     Input('glm-viz-color', 'value'),
     Input('glm-viz-points', 'value'),
     Input('glm-session', 'value'),
     Input('glm-sex', 'value'),
     Input('glm-groups', 'value'),
     Input('stats-dataset', 'value')],
    [State('dataset1-store', 'data'),
     State('dataset2-store', 'data'),
     State('master-store', 'data')]
)
def update_glm_viz_plot(selected_var, color_by, show_points, session, sex_filter, groups, 
                       dataset, data1, data2, master_store):
    """Génère le graphique de distribution pour GLM"""
    
    if not selected_var:
        return go.Figure()
    
    # Sélectionner les données
    if dataset == 'master-store':
        data = master_store
    elif dataset == 'dataset1':
        data = data1
    else:
        data = data2
    
    if not data:
        return go.Figure()
    
    df = pd.DataFrame(data)
    
    # Filtrer les sujets
    if session:
        selected_subjects = df[df['subject'].str.contains(f"_ses-{session}")]['subject'].tolist()
        
        if sex_filter != 'all' and 'sex' in df.columns:
            gender = "M" if sex_filter == 'men' else "F"
            selected_subjects = df[
                (df['subject'].isin(selected_subjects)) & 
                (df['sex'] == gender)
            ]['subject'].tolist()
        
        if groups:
            selected_subjects = [s for s in selected_subjects if detect_group(s) in groups]
        
        df = df[df['subject'].isin(selected_subjects)]
    
    if selected_var not in df.columns:
        return go.Figure()
    
    # Créer le graphique
    fig = go.Figure()
    
    if color_by != 'none' and color_by in df.columns:
        # Boxplot coloré par variable catégorielle
        for category in df[color_by].unique():
            category_data = df[df[color_by] == category][selected_var].dropna()
            
            fig.add_trace(go.Box(
                y=category_data,
                name=str(category),
                boxpoints='all' if show_points else False,
                jitter=0.3 if show_points else 0,
                pointpos=-1.8 if show_points else 0,
                marker_color='#3498db' if color_by == 'sex' and category in ['M', 'male'] else '#e74c3c',
                line_color='#3498db' if color_by == 'sex' and category in ['M', 'male'] else '#e74c3c'
            ))
        
        fig.update_layout(
            title=f"Distribution of {selected_var} by {color_by}",
            yaxis_title=selected_var,
            xaxis_title=color_by,
            showlegend=True
        )
    else:
        # Boxplot simple
        fig.add_trace(go.Box(
            y=df[selected_var].dropna(),
            name=selected_var,
            boxpoints='all' if show_points else False,
            jitter=0.3 if show_points else 0,
            pointpos=-1.8 if show_points else 0,
            marker_color='#3498db',
            line_color='#3498db'
        ))
        
        fig.update_layout(
            title=f"Distribution of {selected_var}",
            yaxis_title=selected_var,
            showlegend=False
        )
    
    return fig

#================================= T-test analyses =================================

# Callback pour exécuter l'analyse T-Test et afficher les résultats
@app.callback(
    [Output('ttest-results-container', 'children'),
    Output('ttest-cleaned-data-store', 'data')],
    [Input('run-ttest-analysis', 'n_clicks'),
     Input('ttest-type', 'value'),
     Input('ttest-dataset', 'value')],
    [State('ttest-group1-session', 'value'),
     State('ttest-group1-sex', 'value'),
     State('ttest-group1-groups', 'value'),
     State('ttest-group2-session', 'value'),
     State('ttest-group2-sex', 'value'),
     State('ttest-group2-groups', 'value'),
     State('ttest-variables', 'value'),
     State('dataset1-store', 'data'),
     State('dataset2-store', 'data'),
     State('master-store', 'data')]
)
def run_ttest_analysis(n_clicks, test_type, dataset, 
                      g1_session, g1_sex, g1_groups, 
                      g2_session, g2_sex, g2_groups, 
                      variables, data1, data2, master_store):
    """Exécute l'analyse T-Test et retourne les résultats"""
    
    if n_clicks is None or n_clicks == 0:
        return html.Div("Configurez les paramètres et cliquez sur 'Run T-Test Analysis'"), None
    
    # Validation des paramètres
    if not all([g1_session, g2_session]):
        return dbc.Alert("Veuillez sélectionner une session pour les deux groupes", color="warning"), None
    
    if not variables:
        return dbc.Alert("Veuillez sélectionner au moins une variable à comparer", color="warning"), None
    
    # Sélectionner les données
    if dataset == 'master-store':
        data = master_store
    elif dataset == 'dataset1':
        data = data1
    else:
        data = data2
    
    if not data:
        return dbc.Alert("Aucune donnée disponible pour l'analyse", color="danger"), None
    
    df = pd.DataFrame(data)
    
    # Fonction pour obtenir les sujets d'un groupe
    def get_group_subjects(session, sex_filter, groups):
        # Filtrer par session
        subjects = df[df['subject'].str.contains(f"_ses-{session}")]['subject'].tolist()
        
        # Filtrer par sexe
        if sex_filter != 'all' and 'sex' in df.columns:
            gender = "M" if sex_filter == 'men' else "F"
            subjects = df[
                (df['subject'].isin(subjects)) & 
                (df['sex'] == gender)
            ]['subject'].tolist()
        
        # Filtrer par groupes
        if groups:
            subjects = [s for s in subjects if detect_group(s) in groups]
        return subjects
    
    # Obtenir les sujets pour chaque groupe
    g1_subjects = get_group_subjects(g1_session, g1_sex, g1_groups)
    g2_subjects = get_group_subjects(g2_session, g2_sex, g2_groups)
    
    if len(g1_subjects) < 3 or len(g2_subjects) < 3:
        return dbc.Alert(f"Groupes trop petits: Groupe 1={len(g1_subjects)}, Groupe 2={len(g2_subjects)} (minimum 3 requis)", color="warning"), None
    
    # Vérification pour test apparié
    paired = (test_type == 'paired')
    if paired:
        # Extraire les identifiants de base
        g1_base_ids = {subj.split('-V')[0] for subj in g1_subjects}
        g2_base_ids = {subj.split('-V')[0] for subj in g2_subjects}
        common_base_ids = g1_base_ids & g2_base_ids
        
        if not common_base_ids:
            return dbc.Alert("Aucun sujet commun trouvé pour le test apparié", color="warning"), None
        
        # Filtrer pour ne garder que les paires valides
        g1_subjects = [subj for subj in g1_subjects if subj.split('-V')[0] in common_base_ids]
        g2_subjects = [subj for subj in g2_subjects if subj.split('-V')[0] in common_base_ids]
        
        if len(common_base_ids) < 3:
            return dbc.Alert(f"Seulement {len(common_base_ids)} paires valides trouvées (minimum 3 requis)", color="warning"), None
    
    # Préparer les données
    df_g1 = df[df['subject'].isin(g1_subjects)]
    df_g2 = df[df['subject'].isin(g2_subjects)]
    
    # Exécuter les analyses pour chaque variable
    results = []
    # cleaned_data = {}
    cleaned_data_dict = {}
    valid_variables = []
    
    for var in variables:
        if var in df.columns:
            # Nettoyage des données pour cette variable
            df1_clean, df2_clean, n_pairs = clean_groups_for_variable(df_g1, df_g2, var, paired)
            
            if paired:
                if n_pairs is None or n_pairs < 3:
                    continue
            else:
                if len(df1_clean) < 3 or len(df2_clean) < 3:
                    continue
            
            # Exécuter le test statistique
            test_results = perform_group_comparison(
                df1_clean[var],
                df2_clean[var],
                paired=paired
            )
            
            if test_results:
                test_results['variable'] = var
                results.append(test_results)

                # Stocker les données nettoyées pour les graphiques
                cleaned_data_dict[var] = {
                    'group1': df1_clean[var].tolist(),
                    'group2': df2_clean[var].tolist(),
                    'group1_subjects': df1_clean['subject'].tolist(),
                    'group2_subjects': df2_clean['subject'].tolist()
                }
                # cleaned_data[var] = (df1_clean, df2_clean)
                # valid_variables.append(var)
    
    if not results:
        return dbc.Alert("Aucune analyse valide n'a pu être effectuée", color="warning"), None
    
    # Créer le DataFrame des résultats
    results_df = pd.DataFrame(results)
    
    # Interface avec onglets
    tabs = dbc.Tabs([
        # Onglet 1: Résultats statistiques
        dbc.Tab(
            dbc.Card([
                dbc.CardHeader("Résultats Statistiques"),
                dbc.CardBody([
                    html.Div([
                        dash_table.DataTable(
                            id='ttest-results-table',
                            columns=[
                                {"name": "Variable", "id": "variable"},
                                {"name": "Test", "id": "test_type"},
                                {"name": "p-value", "id": "p_value", "type": "numeric", "format": {"specifier": ".4f"}},
                                {"name": "Significatif", "id": "significant"},
                                {"name": "Moyenne G1", "id": "mean_group1", "type": "numeric", "format": {"specifier": ".3f"}},
                                {"name": "Moyenne G2", "id": "mean_group2", "type": "numeric", "format": {"specifier": ".3f"}},
                                {"name": "Effect Size", "id": "effect_size", "type": "numeric", "format": {"specifier": ".3f"}},
                                {"name": "N G1", "id": "n_group1"},
                                {"name": "N G2", "id": "n_group2"}
                            ],
                            data=results_df.to_dict('records'),
                            style_cell={'textAlign': 'left'},
                            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                            style_data_conditional=[
                                {
                                    'if': {
                                        'filter_query': '{p_value} < 0.05',
                                        'column_id': 'p_value'
                                    },
                                    'backgroundColor': '#FFE4E1',
                                    'color': 'black'
                                }
                            ]
                        )
                    ]),
                    html.Hr(),
                    html.Div([
                        dbc.Button("Télécharger les résultats Excel", 
                                 id="download-ttest-results", 
                                 color="success", 
                                 className="me-2"),
                        dcc.Download(id="download-ttest")
                    ])
                ])
            ]),
            label="Résultats Statistiques"
        ),
        
        # Onglet 2: Visualisation
        dbc.Tab(
            dbc.Card([
                dbc.CardHeader("Visualisation des Résultats"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Type de graphique"),
                            dbc.RadioItems(
                                id='ttest-plot-type',
                                options=[
                                    {"label": "Violin Plot", "value": "violin"},
                                    {"label": "Box Plot", "value": "box"}
                                ],
                                value="violin",
                                inline=True
                            )
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Variable à visualiser"),
                            dcc.Dropdown(
                                id='ttest-plot-variable',
                                options=[{'label': v, 'value': v} for v in valid_variables],
                                value=valid_variables[0] if valid_variables else None
                            )
                        ], width=6)
                    ]),
                    html.Div(id='ttest-plot-container')
                ])
            ]),
            label="Visualisation"
        )
    ])
    
    return tabs, cleaned_data_dict

# Callback pour mettre à jour le dropdown des variables
@app.callback(
    [Output('ttest-plot-variable', 'options'),
     Output('ttest-plot-variable', 'value')],
    [Input('ttest-cleaned-data-store', 'data')]
)
def update_plot_variables_dropdown(cleaned_data_store):
    """Met à jour les options de variables disponibles pour les graphiques"""
    
    if not cleaned_data_store:
        return [], None
    
    valid_variables = list(cleaned_data_store.keys())
    options = [{'label': var, 'value': var} for var in valid_variables]
    value = valid_variables[0] if valid_variables else None
    
    return options, value

# Callback pour générer les graphiques
@app.callback(
    Output('ttest-plot-container', 'children'),
    [Input('ttest-plot-type', 'value'),
     Input('ttest-plot-variable', 'value')],
    [State('ttest-cleaned-data-store', 'data')]  
)
def update_ttest_plots(plot_type, variable, cleaned_data_store):
    """Met à jour les graphiques pour T-Test"""
    
    if not variable or not cleaned_data_store or variable not in cleaned_data_store:
        return html.Div("Sélectionnez une variable valide")
    
    # Récupérer les vraies données
    var_data = cleaned_data_store[variable]
    group1_data = var_data['group1']
    group2_data = var_data['group2']
    
    # Création du graphique
    if plot_type == "violin":
        fig = go.Figure()
        
        # Ajouter les violons pour chaque groupe
        fig.add_trace(go.Violin(
            y=group1_data,  
            name="Groupe 1",
            box_visible=True,
            meanline_visible=True,
            points="all",
            line_color='blue',
            fillcolor='lightblue'
        ))
        
        fig.add_trace(go.Violin(
            y=group2_data,  # Remplacer par df2_clean[variable]
            name="Groupe 2", 
            box_visible=True,
            meanline_visible=True,
            points="all",
            line_color='orange',
            fillcolor='lightcoral'
        ))
        
    else:  # box plot
        fig = go.Figure()
        
        fig.add_trace(go.Box(
            y=group1_data,  # Remplacer par df1_clean[variable]
            name="Groupe 1",
            boxpoints='all',  # Montrer tous les points
            jitter=0.3,       # Éviter la superposition
            pointpos=-1.8,
            marker_color='blue'
        ))
        
        fig.add_trace(go.Box(
            y=group2_data,  # Remplacer par df2_clean[variable]
            name="Groupe 2",
            boxpoints='all',  # Montrer tous les points
            jitter=0.3,       # Éviter la superposition
            pointpos=-1.8,
            marker_color='orange'
        ))
    
    fig.update_layout(
        title=f"Comparaison des groupes - {variable}",
        yaxis_title=variable,
        xaxis_title="Groupes",
        height=500,
        showlegend=True
    )
    
    return dcc.Graph(figure=fig)

# Callback pour le téléchargement Excel
@app.callback(
    Output("download-ttest", "data"),
    Input("download-ttest-results", "n_clicks"),
    prevent_initial_call=True
)
def download_ttest_results(n_clicks):
    """Télécharge les résultats T-Test en Excel"""
    # Récupérer les résultats depuis le store
    # results_df = ...
    
    # Pour l'instant, retourner un fichier vide
    return dcc.send_data_frame(pd.DataFrame().to_excel, "ttest_results.xlsx")

#================================= Correlation analyses =================================

# Callbacks pour compter les sujets pour chaque set
@app.callback(
    Output('corr-subject-count1', 'children'),
    [Input('corr-session1', 'value'),
     Input('corr-sex1', 'value'),
     Input('corr-groups1', 'value'),
     Input('stats-dataset', 'value')],
    [State('dataset1-store', 'data'),
     State('dataset2-store', 'data'),
     State('master-store', 'data')]
)
def update_corr_subject_count1(session, sex_filter, groups, dataset, data1, data2, master_store):
    """Compte les sujets pour le Set 1"""
    return get_corr_subject_count(session, sex_filter, groups, dataset, data1, data2, master_store)

@app.callback(
    Output('corr-subject-count2', 'children'),
    [Input('corr-session2', 'value'),
     Input('corr-sex2', 'value'),
     Input('corr-groups2', 'value'),
     Input('stats-dataset', 'value')],
    [State('dataset1-store', 'data'),
     State('dataset2-store', 'data'),
     State('master-store', 'data')]
)
def update_corr_subject_count2(session, sex_filter, groups, dataset, data1, data2, master_store):
    """Compte les sujets pour le Set 2"""
    return get_corr_subject_count(session, sex_filter, groups, dataset, data1, data2, master_store)

def get_corr_subject_count(session, sex_filter, groups, dataset, data1, data2, master_store):
    """Fonction utilitaire pour compter les sujets"""
    # Sélectionner les données
    if dataset == 'master-store':
        data = master_store
    elif dataset == 'dataset1':
        data = data1
    elif dataset == 'dataset2':
        data = data2
    else:  # 'both'
        return "Please select a single dataset"
    
    if not data or not session:
        return "0 subjects selected"
    
    df = pd.DataFrame(data)
    
    # Filtrer par session
    subjects = df[df['subject'].str.contains(f"_ses-{session}")]['subject'].tolist()
    
    # Filtrer par sexe
    if sex_filter != 'all' and 'sex' in df.columns:
        gender = "M" if sex_filter == 'men' else "F"
        subjects = df[
            (df['subject'].isin(subjects)) & 
            (df['sex'] == gender)
        ]['subject'].tolist()
    
    # Filtrer par groupes
    if groups:
        subjects = [s for s in subjects if detect_group(s) in groups]
    
    return f"{len(subjects)} subjects selected"

# Callback principal mis à jour pour gérer deux sets
@app.callback(
    Output('corr-results-container', 'children'),
    [Input('run-corr-analysis', 'n_clicks')],
    [State('corr-system-type1', 'value'),
     State('corr-system-type2', 'value'),
     State('corr-session1', 'value'),
     State('corr-sex1', 'value'),
     State('corr-groups1', 'value'),
     State('corr-session2', 'value'),
     State('corr-sex2', 'value'),
     State('corr-groups2', 'value'),
     State('stats-dataset', 'value'),
     State('dataset1-store', 'data'),
     State('dataset2-store', 'data'),
     State('master-store', 'data')]
)
def run_correlation_analysis(n_clicks, system_type1, system_type2, 
                           session1, sex1, groups1, session2, sex2, groups2,
                           dataset, data1, data2, master_store):
    """Exécute l'analyse de corrélation avec deux sets indépendants"""
    
    if n_clicks is None or n_clicks == 0:
        return html.Div()
    
    # Sélectionner les données
    if dataset == 'master-store':
        data = master_store
    elif dataset == 'dataset1':
        data = data1
    elif dataset == 'dataset2':
        data = data2
    else:  # 'both'
        return dbc.Alert("Please select a single dataset for correlation analysis", color="danger")
    
    if not data:
        return dbc.Alert("No data available", color="danger")
    
    df = pd.DataFrame(data)
    
    # Obtenir les sujets pour chaque set
    subjects1, _, _, _ = get_subjects_by_criteria(
        df, 
        analysis_type="By session and sex", 
        session=session1, 
        sex_filter="Men only" if sex1 == "men" else "Women only" if sex1 == "women" else "All",
        groups=groups1
    )
    
    subjects2, _, _, _ = get_subjects_by_criteria(
        df, 
        analysis_type="By session and sex", 
        session=session2, 
        sex_filter="Men only" if sex2 == "men" else "Women only" if sex2 == "women" else "All",
        groups=groups2
    )
    
    if len(subjects1) < 3 or len(subjects2) < 3:
        return dbc.Alert(f"Not enough subjects (Set 1: {len(subjects1)}, Set 2: {len(subjects2)} - minimum 3 each)", color="danger")
    
    # Définir les options de variables
    system_options = {
        "Synaptic ratio": [
            "pre_A4B2", "pre_M1", "pre_D1", "pre_D2",
            "pre_5HT1a", "pre_5HT1b", "pre_5HT2a",
            "pre_5HT4", "pre_5HT6",
            "post_VAChT", "post_DAT", "post_5HTT"
        ],
        "Neurotransmitter (Loc)": [f"loc_inj_{sys}" for sys in [
            "GABAa", "mGluR5", "MU", "H3", "CB1", "A4B2", "M1", "VAChT",
            "D1", "D2", "DAT", "Nor", "5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT"
        ] if f"loc_inj_{sys}" in df.columns],
        "Neurotransmitter (Tract)": [f"tract_inj_{sys}" for sys in [
            "GABAa", "mGluR5", "MU", "H3", "CB1", "A4B2", "M1", "VAChT",
            "D1", "D2", "DAT", "Nor", "5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT"
        ] if f"tract_inj_{sys}" in df.columns],
        "Clinical Outcomes": [col for col in df.columns 
                    if col not in ['subject', 'Sexe_bin', 'sex', 'lesion_volume']
                    and not col.startswith(('loc_inj_', 'tract_inj_', 'pre_', 'post_'))]
    }
    
    # Sélectionner les variables
    vars1 = [col for col in system_options.get(system_type1, []) if col in df.columns]
    vars2 = [col for col in system_options.get(system_type2, []) if col in df.columns]
    
    if not vars1 or not vars2:
        return dbc.Alert("Please select variables for both sets", color="danger")
    
    try:
        # Préparer les données pour la corrélation - ADAPTÉ POUR DEUX SETS
        df_corr, suffix1, suffix2 = prepare_correlation_data_two_sets(df, subjects1, vars1, subjects2, vars2, session1, session2)
        
        if df_corr.shape[0] < 3:
            return dbc.Alert("Not enough common subjects after merging datasets", color="danger")
        
        # Calculer les corrélations (méthode automatique)
        corr_matrix, pval_matrix = get_correlation_matrix(df_corr)
        
        # # Extraire les corrélations croisées entre set1 et set2
        # set1_cols = [col for col in corr_matrix.columns if col.endswith('_set1')]
        # set2_cols = [col for col in corr_matrix.columns if col.endswith('_set2')]
        
        # cross_corr = corr_matrix.loc[set1_cols, set2_cols]
        # cross_pvals = pval_matrix.loc[set1_cols, set2_cols]
        
        # # Créer l'interface des résultats
        # return create_correlation_results(cross_corr, cross_pvals, vars1, vars2, session1, session2)
        return create_correlation_results(corr_matrix, pval_matrix, vars1, vars2, session1, session2, suffix1, suffix2)
        
    except Exception as e:
        return dbc.Alert(f"Error during correlation calculation: {str(e)}", color="danger")

# Callback pour la heatmap interactive
@app.callback(
    Output('correlation-heatmap', 'figure'),
    [Input('correlation-cross-data', 'data'),
     Input('show-all-correlations', 'value'),
     Input('corr-p-threshold', 'value')]
)
def update_correlation_heatmap(correlation_data, show_all, p_thresh):
    """Met à jour la heatmap des corrélations"""
    
    if not correlation_data:
        return go.Figure()
    
    # Récupérer les données
    cross_corr = pd.DataFrame(
        correlation_data['cross_corr'],
        index=correlation_data['corr_index'],
        columns=correlation_data['corr_columns']
    )
    cross_pvals = pd.DataFrame(
        correlation_data['cross_pvals'],
        index=correlation_data['corr_index'],
        columns=correlation_data['corr_columns']
    )
    
    # Préparer les données pour le plot
    show_all_bool = 'show_all' in show_all if show_all else False
    
    # Créer la heatmap
    fig = go.Figure(go.Heatmap(
        z=cross_corr.values,
        x=cross_corr.columns.tolist(),
        y=cross_corr.index.tolist(),
        colorscale='RdBu_r',
        zmin=-1,
        zmax=1,
        text=np.round(cross_corr.values, 2),
        texttemplate="%{text}",
        hoverongaps=False,
        colorbar=dict(title="Correlation")
    ))
    
    # Ajouter des formes pour les cases non significatives
    if not show_all_bool:
        shapes = []
        for i in range(len(cross_corr.index)):
            for j in range(len(cross_corr.columns)):
                if cross_pvals.iloc[i, j] >= p_thresh:
                    shapes.append(dict(
                        type="rect",
                        xref="x",
                        yref="y",
                        x0=j - 0.5,
                        x1=j + 0.5,
                        y0=i - 0.5,
                        y1=i + 0.5,
                        fillcolor="rgba(200,200,200,0.5)",
                        line_width=0,
                        layer="above"
                    ))
        fig.update_layout(shapes=shapes)
    
    fig.update_layout(
        title=f"Correlation Heatmap - Session {correlation_data.get('session', '')}",
        xaxis=dict(tickangle=45),
        yaxis=dict(autorange='reversed'),
        height=600,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

# Callback pour les corrélations significatives
@app.callback(
    Output('significant-correlations-table', 'children'),
    [Input('correlation-cross-data', 'data'),
     Input('corr-p-threshold', 'value')]
)
def update_significant_correlations(correlation_data, p_thresh):
    """Affiche les corrélations significatives"""
    
    if not correlation_data:
        return html.P("No data available")
    
    cross_corr = pd.DataFrame(
        correlation_data['cross_corr'],
        index=correlation_data['corr_index'],
        columns=correlation_data['corr_columns']
    )
    cross_pvals = pd.DataFrame(
        correlation_data['cross_pvals'],
        index=correlation_data['corr_index'],
        columns=correlation_data['corr_columns']
    )
    
    # Filtrer les corrélations significatives
    mask = cross_pvals < p_thresh
    sig_corrs = cross_corr.where(mask)
    
    # Créer un tableau des corrélations significatives
    significant_pairs = []
    for i in range(len(cross_corr.index)):
        for j in range(len(cross_corr.columns)):
            if mask.iloc[i, j] and not np.isnan(sig_corrs.iloc[i, j]):
                significant_pairs.append({
                    'Variable Set 1': cross_corr.index[i],
                    'Variable Set 2': cross_corr.columns[j],
                    'Correlation': round(sig_corrs.iloc[i, j], 3),
                    'p-value': round(cross_pvals.iloc[i, j], 4)
                })
    
    if not significant_pairs:
        return html.P("No significant correlations found")
    
    df_sig = pd.DataFrame(significant_pairs)
    
    return dash_table.DataTable(
        columns=[{"name": col, "id": col} for col in df_sig.columns],
        data=df_sig.to_dict('records'),
        style_cell={'textAlign': 'center', 'padding': '5px'},
        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
        page_size=10,
        sort_action='native'
    )

# Callback pour le téléchargement
@app.callback(
    Output("download-correlation", "data"),
    [Input("download-correlation-btn", "n_clicks")],
    [State('correlation-cross-data', 'data')],
    prevent_initial_call=True
)
def download_correlation_results(n_clicks, correlation_data):
    """Télécharge les résultats de corrélation"""
    
    if not correlation_data:
        return None
    
    cross_corr = pd.DataFrame(
        correlation_data['cross_corr'],
        index=correlation_data['corr_index'],
        columns=correlation_data['corr_columns']
    )
    cross_pvals = pd.DataFrame(
        correlation_data['cross_pvals'],
        index=correlation_data['corr_index'],
        columns=correlation_data['corr_columns']
    )
    
    # Créer un fichier Excel en mémoire
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        cross_corr.to_excel(writer, sheet_name='Cross_Correlation_Matrix')
        cross_pvals.to_excel(writer, sheet_name='Cross_PValue_Matrix')
    
    output.seek(0)
    
    return dcc.send_bytes(
        output.getvalue(),
        filename=f"correlation_analysis_{correlation_data.get('session', '')}.xlsx"
    )

#================================= Data Explorer Tab =================================

# Callback for data explorer content
@app.callback(
    Output('data-explorer-content', 'children'),
    [Input('data-explorer-dataset', 'value')],
    [State('dataset1-store', 'data'),
     State('dataset2-store', 'data')]
)
def update_data_explorer(dataset_sel, data1, data2):
    data = data1 if dataset_sel == 'dataset1' else data2
    if not data:
        return html.Div("No data available")
    
    df = pd.DataFrame(data)
    
    return html.Div([
        html.H5(f"Dataset Preview ({dataset_sel})"),
        html.P(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns"),
        dash_table.DataTable(
            data=df.head(10).to_dict('records'),
            columns=[{'name': col, 'id': col} for col in df.columns],
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '5px'},
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
        ),
        html.Div([
        ]), # className="mb-2"),
       
        # html.Br(),
        # html.H5("Column Summary"),
        # html.Div([
        #     html.Div([
        #         html.H6("Numeric Columns"),
        #         html.Ul([html.Li(col) for col in df.select_dtypes(include=[np.number]).columns])
        #     ], className="col-md-6"),
        #     html.Div([
        #         html.H6("Categorical Columns"),
        #         html.Ul([html.Li(col) for col in df.select_dtypes(include=['object']).columns])
        #     ], className="col-md-6")
        # ], className="row")
    ])

if __name__ == '__main__':
    app.run(debug=True)



