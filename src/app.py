import os
import dash
from dash import dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import numpy as np
import base64
import tempfile
import uuid
import subprocess
import glob

from bionic_core import BionicClassifier
from ml_core import MLClassifier

# Ініціалізація алгоритмів
bionic_model = BionicClassifier()
try:
    ml_model = MLClassifier(model_type='rf')
except Exception as e:
    ml_model = None
    print(f"Warning: ML Model not loaded: {e}")

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG, dbc.icons.FONT_AWESOME])
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
    brand="🔊 Аналіз мовного сигналу: Людина чи ШІ",
    brand_style={"fontSize": "24px", "fontWeight": "bold"},
    color="primary",
    dark=True,
    fluid=True,
    className="mb-4 shadow-sm",
)

upload_card = dbc.Card([
    dbc.CardHeader([html.I(className="fa-solid fa-cloud-upload-alt me-2"), "Вхідні дані"], className="fs-5"),
    dbc.CardBody([
        dbc.Tabs([
            dbc.Tab(label="Завантажити файл", children=[
                html.Div([
                    dcc.Upload(
                        id='upload-audio',
                        children=html.Div([
                            html.I(className="fa-solid fa-file-audio fa-3x mb-3 text-primary"),
                            html.H5("Перетягніть WAV-файл сюди", className="text-white"),
                            html.P("або натисніть для вибору", className="text-muted small")
                        ]),
                        style={
                            'width': '100%', 'height': '180px', 'lineHeight': 'normal',
                            'borderWidth': '2px', 'borderStyle': 'dashed', 'borderColor': '#0d6efd',
                            'borderRadius': '12px', 'textAlign': 'center', 'paddingTop': '35px',
                            'cursor': 'pointer', 'backgroundColor': 'rgba(13, 110, 253, 0.05)',
                            'transition': 'all 0.3s'
                        },
                        multiple=False
                    )
                ], className="mt-3")
            ]),
            dbc.Tab(label="Вибрати з Датасету", children=[
                html.Div([
                    html.Label("Оберіть файл з локальної бази:", className="mt-3 mb-2 text-secondary"),
                    dcc.Dropdown(
                        id='dataset-dropdown',
                        options=get_dataset_files(),
                        placeholder="Оберіть аудіофайл...",
                        className="text-dark"
                    ),
                    dbc.Button([html.I(className="fa-solid fa-sync me-2"), "Оновити список"], id="refresh-dataset-btn", size="sm", color="secondary", outline=True, className="mt-3")
                ], className="p-2")
            ])
        ]),
        html.Div(id='output-filename', className="mt-3 text-center fw-bold text-info"),
        html.Div(id='audio-player-container', className="mt-3 d-flex justify-content-center")
    ])
], className="mb-4 shadow border-0 rounded-3", style={"height": "100%"})

results_card = dbc.Card([
    dbc.CardHeader([html.I(className="fa-solid fa-poll me-2"), "Результати класифікації"], className="fs-5 bg-dark text-white"),
    dbc.CardBody([
        html.H5("Біонічний метод (Простір Хелвага-Щерби)", className="text-secondary mt-2"),
        html.Div([
            html.Label("Чутливість (Поріг R):", className="form-label text-light small mt-2"),
            dcc.Slider(min=30, max=100, step=1, value=61, id='bionic-threshold-slider',
                       marks={i: str(i) for i in range(30, 101, 10)},
                       className="mb-3"),
        ]),
        dcc.Loading(
            id="loading-1", type="dot", color="#0d6efd",
            children=html.Div(id='bionic-verdict', className="mb-4", children=html.Div("Очікування...", className="text-muted fst-italic mt-2"))
        ),
        html.Hr(className="my-4"),
        html.H5("ML метод (Random Forest + MFCCs)", className="text-secondary"),
        dcc.Loading(
            id="loading-2", type="dot", color="#198754",
            children=html.Div(id='ml-verdict', children=html.Div("Очікування...", className="text-muted fst-italic mt-2"))
        ),
    ])
], className="mb-4 shadow border-0 rounded-3", style={"height": "100%"})

analysis_tab = html.Div([
    dbc.Row([
        dbc.Col(upload_card, md=5),
        dbc.Col(results_card, md=7)
    ], className="g-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([html.I(className="fa-solid fa-chart-bar me-2"), "Важливість ознак (Explainable AI)"]),
                dbc.CardBody([
                    dcc.Loading(type="dot", children=dcc.Graph(id='graph-importance', style={'height': '350px'}, config={'displayModeBar': False}))
                ], className="p-1")
            ], className="mb-4 shadow border-0 rounded-3")
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([html.I(className="fa-solid fa-wave-square me-2"), "Часовий графік сигналу"]),
                dbc.CardBody([
                    dcc.Loading(type="dot", children=dcc.Graph(id='graph-signal', style={'height': '350px'}, config={'displayModeBar': False}))
                ], className="p-1")
            ], className="mb-4 shadow border-0 rounded-3")
        ], md=6)
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([html.I(className="fa-solid fa-project-diagram me-2"), "Просторове відображення (Біонічний метод)"]),
                dbc.CardBody([
                    dcc.Loading(type="dot", children=dcc.Graph(id='graph-scatter', style={'height': '550px'}, config={'displayModeBar': False}))
                ], className="p-1")
            ], className="mb-5 shadow border-0 rounded-3")
        ], width=12)
    ])
])

training_tab = html.Div([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([html.I(className="fa-solid fa-database me-2"), "Генерація Датасету"], className="fs-5"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("HuggingFace Dataset"),
                            dbc.Input(id="ds-name", value="PolyAI/minds14", type="text")
                        ], md=4),
                        dbc.Col([
                            html.Label("Провайдер TTS"),
                            dcc.Dropdown(id="ds-provider", options=[
                                {'label': 'Edge TTS (Remote)', 'value': 'edge'},
                                {'label': 'Voicebox (Clone)', 'value': 'voicebox'},
                                {'label': 'MOSS-TTS (Local)', 'value': 'mosstts'}
                            ], value='edge', className="text-dark")
                        ], md=4),
                        dbc.Col([
                            html.Label("Кількість пар"),
                            dbc.Input(id="ds-samples", value=10, type="number")
                        ], md=4),
                    ], className="mb-3"),
                    dbc.Button([html.I(className="fa-solid fa-play me-2"), "Згенерувати"], id="btn-generate-ds", color="primary")
                ])
            ], className="mb-4 shadow border-0 rounded-3")
        ], width=12)
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([html.I(className="fa-solid fa-brain me-2"), "Навчання ML Моделі"], className="fs-5"),
                dbc.CardBody([
                    dbc.Button([html.I(className="fa-solid fa-cogs me-2"), "Запустити Навчання"], id="btn-train-ml", color="success"),
                    html.Div(id="ml-train-output", className="mt-4 p-3 bg-dark text-light rounded font-monospace", style={'whiteSpace': 'pre-wrap'})
                ])
            ], className="shadow border-0 rounded-3")
        ], width=12)
    ])
])

app.layout = dbc.Container([
    navbar,
    dbc.Tabs([
        dbc.Tab(analysis_tab, label="Аналіз Аудіофайлів", tab_id="tab-analyze"),
        dbc.Tab(training_tab, label="Датасет та Навчання", tab_id="tab-train"),
    ], id="tabs", active_tab="tab-analyze", className="mb-4")
], fluid=True, className="px-4")

def format_verdict(label, subtext=""):
    color = "success" if "Людина" in label else "danger"
    icon = "fa-user-check" if "Людина" in label else "fa-robot"
    return dbc.Alert([
        html.I(className=f"fa-solid {icon} fa-2x me-3 align-middle"),
        html.Div([
            html.H4(label, className="alert-heading mb-1 fw-bold"),
            html.Small(subtext, className="text-white-50")
        ], className="d-inline-block align-middle")
    ], color=color, className="d-flex align-items-center mt-3 shadow-sm border-0 rounded-3")

@app.callback(Output("dataset-dropdown", "options"), Input("refresh-dataset-btn", "n_clicks"))
def refresh_dropdown(n): return get_dataset_files()

@app.callback(
    [Output('output-filename', 'children'),
     Output('audio-player-container', 'children'),
     Output('graph-signal', 'figure'),
     Output('graph-scatter', 'figure'),
     Output('graph-importance', 'figure'),
     Output('bionic-verdict', 'children'),
     Output('ml-verdict', 'children')],
    [Input('upload-audio', 'contents'),
     Input('dataset-dropdown', 'value'),
     Input('bionic-threshold-slider', 'value')],
    [State('upload-audio', 'filename')]
)
def process_audio(contents, dropdown_path, bionic_threshold, filename):
    ctx = dash.callback_context
    if not ctx.triggered: raise dash.exceptions.PreventUpdate
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    empty_fig = go.Figure().update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(visible=False), yaxis=dict(visible=False))
    empty_scatter = go.Figure().update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(title="V1"), yaxis=dict(title="V2"), xaxis_range=[0,250], yaxis_range=[0,250])
    wait_div = html.Div("Очікування...", className="text-muted fst-italic mt-3")

    target_path, audio_src, display_name = None, None, ""
    if trigger_id == 'upload-audio' and contents:
        decoded = base64.b64decode(contents.split(',')[1])
        target_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}_{filename}")
        with open(target_path, 'wb') as f: f.write(decoded)
        audio_src, display_name = contents, filename
    elif trigger_id == 'dataset-dropdown' and dropdown_path:
        target_path = dropdown_path
        display_name = os.path.basename(dropdown_path)
        with open(target_path, 'rb') as f: audio_src = f"data:audio/wav;base64,{base64.b64encode(f.read()).decode()}"
    else: return no_update, no_update, empty_fig, empty_scatter, empty_fig, wait_div, wait_div

    fig_signal, fig_scatter, fig_importance = go.Figure(), go.Figure(), go.Figure()
    audio_player = html.Audio(src=audio_src, controls=True, style={"width": "100%", "borderRadius": "30px"})
    
    try:
        # Bionic
        try:
            res = bionic_model.analyze_file(target_path, threshold=bionic_threshold)
            b_sub = f"Score: {res['mean_r']:.1f} | Jitter: {res['jitter']:.2f}% | Shimmer: {res['shimmer']:.2f}%"
            b_html = format_verdict(res['verdict'], b_sub)
            fig_signal.add_trace(go.Scatter(y=res['signal'], line=dict(color='#0d6efd', width=1)))
            if len(res['points']) > 0:
                fig_scatter.add_trace(go.Scatter(x=res['points'][:,0], y=res['points'][:,1], mode='markers', name='Points', marker=dict(size=5, opacity=0.5)))
                if len(res['centers']) == 3:
                    c = np.vstack([res['centers'], res['centers'][0]])
                    fig_scatter.add_trace(go.Scatter(x=c[:,0], y=c[:,1], mode='lines+markers', name='Triangle', line=dict(color='#ffc107')))
            fig_signal.update_layout(template="plotly_dark", margin=dict(l=10,r=10,t=10,b=10))
            fig_scatter.update_layout(template="plotly_dark", xaxis_title="V1", yaxis_title="V2", xaxis_range=[0,250], yaxis_range=[0,250])
        except Exception as e: b_html = dbc.Alert(f"Bionic Error: {e}", color="danger")

        # ML
        if ml_model and ml_model.is_trained:
            try:
                ml_label, ml_prob = ml_model.predict(target_path)
                m_html = format_verdict(ml_label, f"Confidence: {ml_prob*100:.1f}%")
                imp = ml_model.get_feature_importance()[:10]
                if imp:
                    n, v = zip(*imp)
                    fig_importance.add_trace(go.Bar(x=list(v), y=list(n), orientation='h', marker_color='#198754'))
                    fig_importance.update_layout(template="plotly_dark", margin=dict(l=10,r=10,t=10,b=10), yaxis={'autorange': 'reversed'})
            except Exception as e: m_html = dbc.Alert(f"ML Error: {e}", color="danger")
        else: m_html = dbc.Alert("ML Model not trained", color="warning")
    finally:
        if trigger_id == 'upload-audio' and target_path and os.path.exists(target_path): os.remove(target_path)
            
    return html.Span([html.I(className="fa-solid fa-file-audio me-2"), display_name]), audio_player, fig_signal, fig_scatter, fig_importance, b_html, m_html

@app.callback(Output("ds-generate-output", "children"), Input("btn-generate-ds", "n_clicks"), [State("ds-name", "value"), State("ds-provider", "value"), State("ds-samples", "value")], prevent_initial_call=True)
def gen_ds(n, name, prov, samp):
    try:
        res = subprocess.run(["python", "dataset_builder.py", "--dataset", name, "--provider", prov, "--samples", str(samp)], capture_output=True, text=True)
        return html.Pre(res.stdout + res.stderr)
    except Exception as e: return html.Pre(str(e))

@app.callback(Output("ml-train-output", "children"), Input("btn-train-ml", "n_clicks"), prevent_initial_call=True)
def train_ml(n):
    try:
        res = subprocess.run(["./venv/bin/python", "src/train_model.py"], capture_output=True, text=True)
        return html.Pre(res.stdout)
    except Exception as e: return html.Pre(str(e))

if __name__ == '__main__': app.run(debug=True, port=8050)
