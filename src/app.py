import base64
import os
import sys
import subprocess
import tempfile
import uuid
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.environ.get("HuggingFace_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

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
                    dcc.Dropdown(id='dataset-dropdown', options=get_dataset_files(), placeholder="Оберіть аудіофайл...", className="custom-dropdown mb-3"),
                    dbc.Button([html.I(className="fa-solid fa-sync me-2"), "Оновити"], id="refresh-dataset-btn",
                               size="sm", color="info", outline=True, className="mt-3 border-0 shadow-sm")
                ], className="p-2")
            ])
        ]),
        html.Div(id='output-filename', className="mt-3 text-center fw-bold", style={"color": "#4facfe"}),
    ])
], className="glass-card mb-4", style={"height": "100%"})

results_card = dbc.Card([
    dbc.CardHeader([html.I(className="fa-solid fa-microchip me-2"), "Результати аналізу"], className="glass-header"),
    dbc.CardBody([
        html.Div(id='audio-player-container', className="mb-3 d-flex justify-content-center"),
        dbc.Row([
            dbc.Col([
                html.H6("Біонічний метод", className="text-white-50 fw-bold small mb-2 text-center"),
                dcc.Loading(type="dot", color="#4facfe", children=html.Div(id='bionic-verdict', className="text-center"))
            ], md=6, className="border-end", style={"borderColor": "rgba(255,255,255,0.1)"}),
            dbc.Col([
                html.H6("ML метод (RF)", className="text-white-50 fw-bold small mb-2 text-center"),
                dcc.Loading(type="dot", color="#00f2fe", children=html.Div(id='ml-verdict', className="text-center"))
            ], md=6),
        ], className="align-items-center mb-3"),
        html.Hr(style={"borderColor": "rgba(255,255,255,0.1)"}),
        html.Label("Чутливість біонічного аналізу (Поріг Score):", className="text-white-50 small mb-1"),
        dcc.Slider(min=30, max=100, step=1, value=61, id='bionic-threshold-slider',
                   marks={i: str(i) for i in range(30, 101, 10)}, className="px-0"),
    ])
], className="glass-card mb-4", style={"height": "100%"})

analysis_tab = html.Div([
    dbc.Row([dbc.Col(upload_card, md=4), dbc.Col(results_card, md=8)], className="g-4 mb-4"),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([html.I(className="fa-solid fa-wave-square me-2"), "Осцилограма (Часова область)"],
                               className="glass-header"),
                dbc.CardBody([dcc.Loading(type="dot", color="#4facfe",
                                          children=dcc.Graph(id='graph-signal', style={'height': '300px'},
                                                             config={'displayModeBar': False}))], className="p-1")
            ], className="glass-card mb-4")
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([html.I(className="fa-solid fa-braille me-2"), "Спектрограма (Частотна область)"],
                               className="glass-header"),
                dbc.CardBody([dcc.Loading(type="dot", color="#4facfe",
                                          children=dcc.Graph(id='graph-spectrogram', style={'height': '300px'},
                                                             config={'displayModeBar': False}))], className="p-1")
            ], className="glass-card mb-4")
        ], md=6)
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([html.I(className="fa-solid fa-chart-bar me-2"), "Explainable AI (Важливість ознак)"],
                               className="glass-header"),
                dbc.CardBody([dcc.Loading(type="dot", color="#4facfe",
                                          children=dcc.Graph(id='graph-importance', style={'height': '400px'},
                                                             config={'displayModeBar': False}))], className="p-1")
            ], className="glass-card")
        ], md=5),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([html.I(className="fa-solid fa-project-diagram me-2"), "Біонічний простір v1/v2"],
                               className="glass-header"),
                dbc.CardBody([dcc.Loading(type="dot", color="#4facfe",
                                          children=dcc.Graph(id='graph-scatter', style={'height': '400px'},
                                                             config={'displayModeBar': False}))], className="p-1")
            ], className="glass-card")
        ], md=7)
    ], className="g-4")
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
                            html.Label("HuggingFace Dataset", className="text-white-50 small"),
                            dcc.Dropdown(
                                id="ds-name",
                                options=[
                                    {'label': 'minds14 (Banking)', 'value': 'PolyAI/minds14'},
                                    {'label': 'LibriSpeech (Books)', 'value': 'openslr/librispeech_asr'}
                                ],
                                value='PolyAI/minds14',
                                className="custom-dropdown"
                            )
                        ], md=6, className="mb-3"),
                        dbc.Col([
                            html.Label("Config", className="text-white-50 small"),
                            dcc.Dropdown(id="ds-config", className="custom-dropdown")
                        ], md=3, className="mb-3"),
                        dbc.Col([
                            html.Label("Split", className="text-white-50 small"),
                            dcc.Dropdown(id="ds-split", className="custom-dropdown")
                        ], md=3, className="mb-3"),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Провайдер TTS", className="text-white-50 small"), 
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
                        ], md=8),
                        dbc.Col([
                            html.Label("Кількість пар", className="text-white-50 small"),
                            dbc.Input(id="ds-samples", value=10, type="number", className="bg-transparent text-white", style={"height": "40px"})
                        ], md=4),
                    ], className="mb-4"),
                    dbc.Button([html.I(className="fa-solid fa-bolt me-2"), "Почати генерацію"], id="btn-generate-ds",
                               color="info", className="w-100 rounded-pill shadow-sm fw-bold"),
                    dcc.Interval(id='interval-generate', interval=1000, n_intervals=0, disabled=True),
                    html.Div(id="ds-generate-output", className="mt-4 p-3 rounded-3 font-monospace small",
                                          style={'whiteSpace': 'pre-wrap', 'background': 'rgba(0,0,0,0.3)',
                                                 'border': '1px solid rgba(255,255,255,0.05)', 'color': '#4facfe', 'minHeight': '150px', 'maxHeight': '300px', 'overflowY': 'auto'})
                ])
            ], className="glass-card mb-4")
        ], md=7),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([html.I(className="fa-solid fa-brain me-2"), "Навчання Моделі"],
                               className="glass-header"),
                dbc.CardBody([
                    html.P("Навчання класифікатора Random Forest на основі згенерованого локального датасету.",
                           className="text-white-50 small mb-4"),
                    dbc.Button([html.I(className="fa-solid fa-graduation-cap me-2"), "Навчати модель"], id="btn-train-ml",
                               color="success", className="w-100 rounded-pill shadow-sm fw-bold"),
                    dcc.Interval(id='interval-train', interval=1000, n_intervals=0, disabled=True),
                    html.Div(id="ml-train-output", className="mt-4 p-3 rounded-3 font-monospace small",
                                          style={'whiteSpace': 'pre-wrap', 'background': 'rgba(0,0,0,0.3)',
                                                 'border': '1px solid rgba(255,255,255,0.05)', 'color': '#2ecc71', 'minHeight': '150px', 'maxHeight': '300px', 'overflowY': 'auto'})
                ])
            ], className="glass-card mb-4")
        ], md=5)
    ])
])

# Клієнтський скрипт для авто-скролу
app.clientside_callback(
    """
    function(children, style) {
        const el = document.getElementById('ds-generate-output');
        if (el) {
            // Скролимо лише якщо користувач і так був внизу (з допуском 50px)
            const isNearBottom = el.scrollHeight - el.clientHeight - el.scrollTop < 100;
            if (isNearBottom || el.scrollTop === 0) {
                // Використовуємо setTimeout, щоб дати браузеру час відрендерити новий текст
                setTimeout(() => { el.scrollTop = el.scrollHeight; }, 10);
            }
        }
        return style;
    }
    """,
    Output('ds-generate-output', 'style'),
    Input('ds-generate-output', 'children'),
    State('ds-generate-output', 'style')
)

app.clientside_callback(
    """
    function(children, style) {
        const el = document.getElementById('ml-train-output');
        if (el) {
            const isNearBottom = el.scrollHeight - el.clientHeight - el.scrollTop < 100;
            if (isNearBottom || el.scrollTop === 0) {
                setTimeout(() => { el.scrollTop = el.scrollHeight; }, 10);
            }
        }
        return style;
    }
    """,
    Output('ml-train-output', 'style'),
    Input('ml-train-output', 'children'),
    State('ml-train-output', 'style')
)

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
    bg = "rgba(0, 242, 254, 0.1)" if is_human else "rgba(255, 8, 68, 0.1)"
    icon = "fa-user-check" if is_human else "fa-robot"

    return html.Div([
        html.Div([
            html.I(className=f"fa-solid {icon} fa-xl", style={"color": color, "marginRight": "15px"}),
            html.Div([
                html.Div(label, className="fw-bold", style={"color": color, "fontSize": "1.1rem"}),
                html.Div(subtext, className="text-white-50 small")
            ])
        ], className="d-flex align-items-center p-3 rounded-3", 
           style={"background": bg, "border": f"1px solid {color}33"})
    ], className="mt-2")


@app.callback(Output("dataset-dropdown", "options"), Input("refresh-dataset-btn", "n_clicks"))
def refresh_dropdown(n): return get_dataset_files()


@app.callback(
    [Output('output-filename', 'children'), Output('audio-player-container', 'children'),
     Output('graph-signal', 'figure'), Output('graph-spectrogram', 'figure'), 
     Output('graph-scatter', 'figure'), Output('graph-importance', 'figure'),
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
        return no_update, no_update, empty_fig, empty_fig, empty_scatter, empty_fig, wait_div, wait_div

    fig_signal, fig_scatter, fig_importance, fig_spectrogram = go.Figure(), go.Figure(), go.Figure(), go.Figure()
    audio_player = html.Audio(src=audio_src, controls=True,
                              style={"width": "100%", "borderRadius": "30px", "height": "40px"})

    try:
        try:
            import librosa
            res = bionic_model.analyze_file(target_path, threshold=bionic_threshold)
            b_sub = f"Score: {res['mean_r']:.1f} | Jitter: {res['jitter']:.1f}% | Shimmer: {res['shimmer']:.1f}%"
            b_html = format_verdict(res['verdict'], b_sub)

            fig_signal.add_trace(go.Scatter(y=res['signal'], line=dict(color='#00f2fe', width=1), fill='tozeroy',
                                            fillcolor='rgba(0, 242, 254, 0.1)'))
            fig_signal.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=10, b=10),
                                     paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                     xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))

            # Спектрограма
            D = librosa.stft(res['signal'] / 128.0)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            fig_spectrogram.add_trace(go.Heatmap(z=S_db, colorscale='Viridis', showscale=False))
            fig_spectrogram.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=10, b=10),
                                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                          xaxis=dict(showgrid=False, title="Час (frames)"),
                                          yaxis=dict(showgrid=False, title="Частота (bins)"))

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
            import traceback
            print(f"Bionic Error: {traceback.format_exc()}")
            b_html = html.Div(f"Bionic Error: {e}", className="text-danger mt-3")

        # ML Класифікація
        m_html = html.Div("ML Model not trained", className="text-warning mt-3")
        
        # Створюємо локальний екземпляр для підвантаження актуальних файлів з диску
        local_ml = MLClassifier(model_type='rf')
        
        if local_ml.is_trained:
            try:
                ml_label, ml_prob = local_ml.predict(target_path)
                m_html = format_verdict(ml_label, f"Confidence: {ml_prob * 100:.1f}%")
                imp = local_ml.get_feature_importance()[:10]
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
    finally:
        if trigger_id == 'upload-audio' and target_path and os.path.exists(target_path): os.remove(target_path)

    return html.Span([html.I(className="fa-solid fa-headphones me-2"),
                      display_name]), audio_player, fig_signal, fig_spectrogram, fig_scatter, fig_importance, b_html, m_html


@app.callback(
    [Output("ds-config", "options"), Output("ds-config", "value")],
    Input("ds-name", "value")
)
def update_config_dropdown(ds_name):
    from datasets import get_dataset_config_names
    try:
        # Автоматично тягнемо всі доступні конфіги прямо з HuggingFace
        configs = get_dataset_config_names(ds_name, token=HF_TOKEN)
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


@app.callback(
    [Output("ds-split", "options"), Output("ds-split", "value")],
    [Input("ds-name", "value"), Input("ds-config", "value")]
)
def update_split_dropdown(ds_name, config_name):
    from datasets import get_dataset_split_names
    if not ds_name or not config_name:
        return [], None
    try:
        splits = get_dataset_split_names(ds_name, config_name, token=HF_TOKEN)
        if not splits:
            return [{'label': 'train', 'value': 'train'}], 'train'
        
        opts = [{'label': s, 'value': s} for s in splits]
        
        # Шукаємо 'train' або щось подібне для дефолту
        default_val = splits[0]
        for s in splits:
            if 'train' in s.lower():
                default_val = s
                break
                
        return opts, default_val
    except Exception as e:
        print(f"Failed to fetch splits for {ds_name}/{config_name}: {e}")
        return [{'label': 'train', 'value': 'train'}], 'train'


import threading

def run_command_in_background(cmd, log_file):
    with open(log_file, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)

@app.callback(
    [Output("ds-generate-output", "children"), Output("interval-generate", "disabled"), Output("btn-generate-ds", "disabled")],
    [Input("btn-generate-ds", "n_clicks"), Input("interval-generate", "n_intervals")],
    [State("ds-name", "value"), State("ds-config", "value"), State("ds-split", "value"), State("ds-provider", "value"), State("ds-samples", "value")],
    prevent_initial_call=True
)
def generate_dataset_callback(n_clicks, n_intervals, ds_name, config_name, split_name, provider, samples):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    log_file = os.path.join(tempfile.gettempdir(), "generate_log.txt")

    if trigger_id == "btn-generate-ds":
        # Start background thread
        open(log_file, "w").close() # Clear file
        cmd = [sys.executable, "dataset_builder.py", 
               "--dataset", ds_name, 
               "--config", config_name, 
               "--split", split_name, 
               "--provider", provider, 
               "--samples", str(samples)]
        thread = threading.Thread(target=run_command_in_background, args=(cmd, log_file))
        thread.start()
        return "Процес запущено...\n", False, True

    elif trigger_id == "interval-generate":
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                logs = f.read()
            # Disable interval if word "завершено" or "Процес перервано" is found
            finished = "Генерацію завершено!" in logs or "Процес перервано" in logs or "ПЛАН ДІЙ" in logs
            return logs if logs else "Очікування логів...", finished, finished == False

    return no_update, no_update, no_update


@app.callback(
    [Output("ml-train-output", "children"), Output("interval-train", "disabled"), Output("btn-train-ml", "disabled")],
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
        return "Навчання запущено...\n", False, True
        
    elif trigger_id == "interval-train":
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                logs = f.read()
            finished = "Модель успішно навчена" in logs or "Traceback" in logs or "ПОМИЛКА" in logs
            return logs if logs else "Очікування логів...", finished, finished == False

    return no_update, no_update, no_update


if __name__ == '__main__':
    app.run(debug=True, port=8050)
