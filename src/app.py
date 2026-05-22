import base64
import os
import sys
import subprocess
import tempfile
import uuid

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
from dash import dcc, html, Input, Output, State, no_update

from bionic_core import BionicClassifier
from ml_core import MLClassifier

bionic_model = BionicClassifier()
try:
    ml_model = MLClassifier(model_type='rf')
except Exception as e:
    ml_model = None
    print(f"Warning: ML Model not loaded: {e}")

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY, dbc.icons.FONT_AWESOME], suppress_callback_exceptions=True)
app.title = "Гібридна класифікація 'Людина-ШІ'"


def get_dataset_files():
    files = []
    for category in ['human', 'ai']:
        path = os.path.join('dataset', category)
        if os.path.exists(path):
            for f in os.listdir(path):
                if f.endswith('.wav'):
                    files.append({'label': f"[{category.upper()}] {f}", 'value': os.path.join(path, f)})
    return files


navbar = dbc.NavbarSimple(
    brand=html.Div(
        [html.I(className="fa-solid fa-microphone-lines me-3", style={"color": "#4facfe"}), "Voice I/O Lab"]),
    brand_style={"fontSize": "22px", "fontWeight": "600", "letterSpacing": "1px"},
    color="transparent",
    dark=True,
    fluid=True,
    className="gradient-navbar mb-4",
)

upload_card = dbc.Card([
    dbc.CardHeader([html.I(className="fa-solid fa-cloud-upload-alt me-2"), "Вхідні дані"], className="glass-header"),
    dbc.CardBody([
        dbc.Tabs([
            dbc.Tab(label="Завантажити", children=[
                html.Div([
                    dcc.Upload(
                        id='upload-audio',
                        children=html.Div([
                            html.I(className="fa-solid fa-file-waveform fa-3x mb-3", style={"color": "#4facfe"}),
                            html.H5("Перетягніть WAV-файл сюди", className="text-white fw-bold"),
                            html.P("або натисніть для вибору", className="text-muted small")
                        ]),
                        className="upload-box",
                        style={'width': '100%', 'height': '180px', 'lineHeight': 'normal', 'textAlign': 'center',
                               'paddingTop': '40px', 'cursor': 'pointer'},
                        multiple=False
                    )
                ], className="mt-3")
            ]),
            dbc.Tab(label="Датасет", children=[
                html.Div([
                    html.Label("Оберіть файл з локальної бази:", className="mt-3 mb-2 text-white-50"),
                    dcc.Dropdown(id='dataset-dropdown', options=get_dataset_files(), placeholder="Оберіть аудіофайл...", className="custom-dropdown"),
                    dbc.Button([html.I(className="fa-solid fa-sync me-2"), "Оновити"], id="refresh-dataset-btn",
                               size="sm", color="info", outline=True, className="mt-3 border-0 shadow-sm")
                ], className="p-2")
            ])
        ]),
        html.Div(id='output-filename', className="mt-3 text-center fw-bold", style={"color": "#4facfe"}),
        html.Div(id='audio-player-container', className="mt-3 d-flex justify-content-center")
    ])
], className="glass-card mb-4", style={"height": "100%"})

results_card = dbc.Card([
    dbc.CardHeader([html.I(className="fa-solid fa-microchip me-2"), "Результати аналізу"], className="glass-header"),
    dbc.CardBody([
        html.H6("Біонічний метод (Простір Хелвага-Щерби)", className="text-white-50 mt-2 fw-bold"),
        html.Div([
            html.Label("Чутливість (Поріг Score):", className="form-label text-light small mt-2"),
            dcc.Slider(min=30, max=100, step=1, value=61, id='bionic-threshold-slider',
                       marks={i: str(i) for i in range(30, 101, 10)}, className="mb-3"),
        ]),
        dcc.Loading(type="dot", color="#4facfe", children=html.Div(id='bionic-verdict', className="mb-4",
                                                                   children=html.Div("Очікування...",
                                                                                     className="text-muted fst-italic mt-2"))),
        html.Hr(style={"borderColor": "rgba(255,255,255,0.1)"}),
        html.H6("ML метод (Random Forest + MFCCs)", className="text-white-50 fw-bold"),
        dcc.Loading(type="dot", color="#00f2fe", children=html.Div(id='ml-verdict', children=html.Div("Очікування...",
                                                                                                      className="text-muted fst-italic mt-2"))),
    ])
], className="glass-card mb-4", style={"height": "100%"})

analysis_tab = html.Div([
    dbc.Row([dbc.Col(upload_card, md=5), dbc.Col(results_card, md=7)], className="g-4"),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([html.I(className="fa-solid fa-chart-bar me-2"), "Explainable AI (Важливість ознак)"],
                               className="glass-header"),
                dbc.CardBody([dcc.Loading(type="dot", color="#4facfe",
                                          children=dcc.Graph(id='graph-importance', style={'height': '350px'},
                                                             config={'displayModeBar': False}))], className="p-1")
            ], className="glass-card mb-4")
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([html.I(className="fa-solid fa-wave-square me-2"), "Осцилограма"],
                               className="glass-header"),
                dbc.CardBody([dcc.Loading(type="dot", color="#4facfe",
                                          children=dcc.Graph(id='graph-signal', style={'height': '350px'},
                                                             config={'displayModeBar': False}))], className="p-1")
            ], className="glass-card mb-4")
        ], md=6)
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([html.I(className="fa-solid fa-project-diagram me-2"), "Біонічний простір v1/v2"],
                               className="glass-header"),
                dbc.CardBody([dcc.Loading(type="dot", color="#4facfe",
                                          children=dcc.Graph(id='graph-scatter', style={'height': '500px'},
                                                             config={'displayModeBar': False}))], className="p-1")
            ], className="glass-card h-100")
        ], width=12)
    ])
])

training_tab = html.Div([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([html.I(className="fa-solid fa-network-wired me-2"), "Генерація Датасету"],
                               className="glass-header"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("HuggingFace Dataset", className="text-white-50"),
                            dcc.Dropdown(
                                id="ds-name",
                                options=[
                                    {'label': 'PolyAI/minds14 (Банкінг)', 'value': 'PolyAI/minds14'},
                                    {'label': 'openslr/librispeech_asr (Аудіокниги)', 'value': 'openslr/librispeech_asr'}
                                ],
                                value='PolyAI/minds14',
                                className="custom-dropdown"
                            )
                        ], md=3),
                        dbc.Col([
                            html.Label("Конфігурація (Config)", className="text-white-50"),
                            dcc.Dropdown(id="ds-config", className="custom-dropdown")
                        ], md=3),
                        dbc.Col([
                            html.Label("Провайдер TTS", className="text-white-50"), 
                            dcc.Dropdown(
                                id="ds-provider", 
                                options=[
                                    {'label': 'Edge TTS (Remote)', 'value': 'edge'}, 
                                    {'label': 'LuxTTS (Clone)', 'value': 'lux'}, 
                                    {'label': 'MOSS-TTS (Local)', 'value': 'mosstts'}, 
                                    {'label': 'Voicebox API', 'value': 'voicebox'}
                                ], 
                                value='edge', 
                                className="custom-dropdown"
                            )
                        ], md=3),
                        dbc.Col([
                            html.Label("Кількість пар", className="text-white-50"),
                            dbc.Input(id="ds-samples", value=10, type="number", className="bg-transparent text-white", style={"height": "40px"})
                        ], md=3),
                    ], className="mb-4"),
                    dbc.Button([html.I(className="fa-solid fa-bolt me-2"), "Почати генерацію"], id="btn-generate-ds",
                               color="info", className="w-100 rounded-pill shadow-sm fw-bold"),
                    dcc.Interval(id='interval-generate', interval=1000, n_intervals=0, disabled=True),
                    dcc.Loading(
                        type="circle", color="#4facfe",
                        children=html.Div(id="ds-generate-output", className="mt-4 p-3 rounded-3 font-monospace small",
                                          style={'whiteSpace': 'pre-wrap', 'background': 'rgba(0,0,0,0.3)',
                                                 'border': '1px solid rgba(255,255,255,0.05)', 'color': '#4facfe', 'minHeight': '50px', 'maxHeight': '300px', 'overflowY': 'auto'})
                    )
                ])
            ], className="glass-card mb-4")
        ], width=12)
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([html.I(className="fa-solid fa-brain me-2"), "Навчання Моделі"],
                               className="glass-header"),
                dbc.CardBody([
                    dbc.Button([html.I(className="fa-solid fa-cogs me-2"), "Запустити ML Training"], id="btn-train-ml",
                               color="success", className="w-100 rounded-pill shadow-sm fw-bold"),
                    dcc.Interval(id='interval-train', interval=1000, n_intervals=0, disabled=True),
                    dcc.Loading(
                        type="circle", color="#00f2fe",
                        children=html.Div(id="ml-train-output", className="mt-4 p-3 rounded-3 font-monospace small",
                                          style={'whiteSpace': 'pre-wrap', 'background': 'rgba(0,0,0,0.3)',
                                                 'border': '1px solid rgba(255,255,255,0.05)', 'color': '#00f2fe', 'minHeight': '50px', 'maxHeight': '300px', 'overflowY': 'auto'})
                    )
                ])
            ], className="glass-card h-100")
        ], width=12)
    ])
])

app.layout = dbc.Container([
    navbar,
    dbc.Tabs([
        dbc.Tab(analysis_tab, label="Лабораторія Аналізу", tab_id="tab-analyze"),
        dbc.Tab(training_tab, label="Навчання & Дані", tab_id="tab-train"),
    ], id="tabs", active_tab="tab-analyze", className="mb-4 border-0")
], fluid=True, className="px-md-5")


def format_verdict(label, subtext=""):
    is_human = "Людина" in label
    color = "#00f2fe" if is_human else "#ff0844"
    bg = "rgba(0, 242, 254, 0.05)" if is_human else "rgba(255, 8, 68, 0.05)"
    icon = "fa-user-check" if is_human else "fa-robot"

    return html.Div([
        html.Div([
            html.Div([html.I(className=f"fa-solid {icon} fa-2x")],
                     style={"color": color, "background": bg, "padding": "15px", "borderRadius": "50%",
                            "marginRight": "20px"}),
            html.Div([
                html.H4(label, className="mb-1 fw-bold", style={"color": color, "textShadow": f"0 0 10px {color}"}),
                html.Div(subtext, className="text-light opacity-75 small")
            ])
        ], className="d-flex align-items-center")
    ], style={"border": f"1px solid rgba(255,255,255,0.05)", "background": "rgba(0,0,0,0.2)", "borderRadius": "20px",
              "padding": "20px"}, className="mt-3")


@app.callback(Output("dataset-dropdown", "options"), Input("refresh-dataset-btn", "n_clicks"))
def refresh_dropdown(n): return get_dataset_files()


@app.callback(
    [Output('output-filename', 'children'), Output('audio-player-container', 'children'),
     Output('graph-signal', 'figure'), Output('graph-scatter', 'figure'), Output('graph-importance', 'figure'),
     Output('bionic-verdict', 'children'), Output('ml-verdict', 'children')],
    [Input('upload-audio', 'contents'), Input('dataset-dropdown', 'value'), Input('bionic-threshold-slider', 'value')],
    [State('upload-audio', 'filename')]
)
def process_audio(contents, dropdown_path, bionic_threshold, filename):
    ctx = dash.callback_context
    if not ctx.triggered: raise dash.exceptions.PreventUpdate
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    empty_fig = go.Figure().update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
                                          plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(visible=False),
                                          yaxis=dict(visible=False))
    empty_scatter = go.Figure().update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
                                              plot_bgcolor='rgba(0,0,0,0)',
                                              xaxis=dict(title="V1", gridcolor="rgba(255,255,255,0.05)"),
                                              yaxis=dict(title="V2", gridcolor="rgba(255,255,255,0.05)"),
                                              xaxis_range=[0, 250], yaxis_range=[0, 250])
    wait_div = html.Div("Очікування...", className="text-muted fst-italic mt-3")

    target_path, audio_src, display_name = None, None, ""
    if trigger_id == 'upload-audio' and contents:
        decoded = base64.b64decode(contents.split(',')[1])
        target_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}_{filename}")
        with open(target_path, 'wb') as f:
            f.write(decoded)
        audio_src, display_name = contents, filename
    elif trigger_id == 'dataset-dropdown' and dropdown_path:
        target_path = dropdown_path
        display_name = os.path.basename(dropdown_path)
        with open(target_path, 'rb') as f:
            audio_src = f"data:audio/wav;base64,{base64.b64encode(f.read()).decode()}"
    else:
        return no_update, no_update, empty_fig, empty_scatter, empty_fig, wait_div, wait_div

    fig_signal, fig_scatter, fig_importance = go.Figure(), go.Figure(), go.Figure()
    audio_player = html.Audio(src=audio_src, controls=True,
                              style={"width": "100%", "borderRadius": "30px", "height": "40px"})

    try:
        try:
            res = bionic_model.analyze_file(target_path, threshold=bionic_threshold)
            b_sub = f"Score: {res['mean_r']:.1f} | Jitter: {res['jitter']:.1f}% | Shimmer: {res['shimmer']:.1f}%"
            b_html = format_verdict(res['verdict'], b_sub)

            fig_signal.add_trace(go.Scatter(y=res['signal'], line=dict(color='#00f2fe', width=1), fill='tozeroy',
                                            fillcolor='rgba(0, 242, 254, 0.1)'))
            fig_signal.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=10, b=10),
                                     paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                     xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))

            if len(res['points']) > 0:
                fig_scatter.add_trace(
                    go.Scatter(x=res['points'][:, 0], y=res['points'][:, 1], mode='markers', name='Points',
                               marker=dict(size=6, color='#00f2fe', opacity=0.6, line=dict(width=1, color='white'))))
                if len(res['centers']) == 3:
                    c = np.vstack([res['centers'], res['centers'][0]])
                    fig_scatter.add_trace(go.Scatter(x=c[:, 0], y=c[:, 1], mode='lines+markers', name='Triangle',
                                                     line=dict(color='#ff0844', width=2),
                                                     marker=dict(size=12, color='#ff0844', symbol='diamond')))
            fig_scatter.update_layout(template="plotly_dark", xaxis_title="V1", yaxis_title="V2", xaxis_range=[0, 250],
                                      yaxis_range=[0, 250], paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                      xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                                      yaxis=dict(gridcolor="rgba(255,255,255,0.05)"))
        except Exception as e:
            b_html = html.Div(f"Bionic Error: {e}", className="text-danger mt-3")

        if ml_model and ml_model.is_trained:
            try:
                ml_label, ml_prob = ml_model.predict(target_path)
                m_html = format_verdict(ml_label, f"Confidence: {ml_prob * 100:.1f}%")
                imp = ml_model.get_feature_importance()[:10]
                if imp:
                    n, v = zip(*imp)
                    fig_importance.add_trace(go.Bar(x=list(v), y=list(n), orientation='h', marker=dict(color='#4facfe',
                                                                                                       line=dict(
                                                                                                           color='rgba(255,255,255,0.5)',
                                                                                                           width=1))))
                    fig_importance.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=10, b=10),
                                                 paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                                 yaxis={'autorange': 'reversed'},
                                                 xaxis=dict(gridcolor="rgba(255,255,255,0.05)"))
            except Exception as e:
                m_html = html.Div(f"ML Error: {e}", className="text-danger mt-3")
        else:
            m_html = html.Div("ML Model not trained", className="text-warning mt-3")
    finally:
        if trigger_id == 'upload-audio' and target_path and os.path.exists(target_path): os.remove(target_path)

    return html.Span([html.I(className="fa-solid fa-headphones me-2"),
                      display_name]), audio_player, fig_signal, fig_scatter, fig_importance, b_html, m_html


@app.callback(
    [Output("ds-config", "options"), Output("ds-config", "value")],
    Input("ds-name", "value")
)
def update_config_dropdown(ds_name):
    from datasets import get_dataset_config_names
    try:
        # Автоматично тягнемо всі доступні конфіги прямо з HuggingFace
        configs = get_dataset_config_names(ds_name)
        if not configs:
            return [{'label': 'default', 'value': 'default'}], 'default'
            
        opts = [{'label': c, 'value': c} for c in configs]
        
        # Намагаємось обрати розумний дефолт
        default_val = configs[0]
        for c in configs:
            if c in ['en-US', 'clean', 'default', 'all']:
                default_val = c
                break
                
        return opts, default_val
    except Exception as e:
        print(f"Failed to fetch configs for {ds_name}: {e}")
        return [{'label': 'default', 'value': 'default'}], 'default'


import threading

def run_command_in_background(cmd, log_file):
    with open(log_file, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)

@app.callback(
    [Output("ds-generate-output", "children"), Output("interval-generate", "disabled")],
    [Input("btn-generate-ds", "n_clicks"), Input("interval-generate", "n_intervals")],
    [State("ds-name", "value"), State("ds-config", "value"), State("ds-provider", "value"), State("ds-samples", "value")],
    prevent_initial_call=True
)
def generate_dataset_callback(n_clicks, n_intervals, ds_name, config_name, provider, samples):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    log_file = os.path.join(tempfile.gettempdir(), "generate_log.txt")

    if trigger_id == "btn-generate-ds":
        # Start background thread
        open(log_file, "w").close() # Clear file
        cmd = [sys.executable, "dataset_builder.py", "--dataset", ds_name, "--config", config_name, "--provider", provider, "--samples", str(samples)]
        thread = threading.Thread(target=run_command_in_background, args=(cmd, log_file))
        thread.start()
        return "Процес запущено...\n", False
    
    elif trigger_id == "interval-generate":
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                logs = f.read()
            # If thread is dead and logs haven't changed, we can optionally stop interval, 
            # but simple file reading is fine for now. Disable interval if word "завершено" or "помилка" is found (simplified logic).
            disabled = "Генерацію завершено!" in logs or "Traceback" in logs or "ПЛАН ДІЙ" in logs
            return logs if logs else "Очікування логів...", disabled
            
    return no_update, no_update


@app.callback(
    [Output("ml-train-output", "children"), Output("interval-train", "disabled")],
    [Input("btn-train-ml", "n_clicks"), Input("interval-train", "n_intervals")],
    prevent_initial_call=True
)
def train_ml(n_clicks, n_intervals):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    log_file = os.path.join(tempfile.gettempdir(), "train_log.txt")

    if trigger_id == "btn-train-ml":
        open(log_file, "w").close()
        cmd = [sys.executable, "src/train_model.py"]
        thread = threading.Thread(target=run_command_in_background, args=(cmd, log_file))
        thread.start()
        return "Навчання запущено...\n", False
        
    elif trigger_id == "interval-train":
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                logs = f.read()
            disabled = "Модель успішно навчена" in logs or "Traceback" in logs
            return logs if logs else "Очікування логів...", disabled

    return no_update, no_update


if __name__ == '__main__':
    app.run(debug=True, port=8050)
