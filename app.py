import os
import flask
import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import numpy as np
from scipy.integrate import solve_ivp
from numpy import heaviside
import plotly.graph_objects as go

# Configuração de prefixo de rota para suporte tanto em desenvolvimento local quanto em proxy reverso de deploy
prefix = os.getenv('DASH_REQUESTS_PATHNAME_PREFIX', '/dash_sir/')
if not prefix.startswith('/'):
    prefix = '/' + prefix
if not prefix.endswith('/'):
    prefix = prefix + '/'

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.LUMEN, dbc.icons.FONT_AWESOME],
    requests_pathname_prefix=prefix,
    routes_pathname_prefix=prefix,
    suppress_callback_exceptions=True
)
server = app.server

# Redirecionamento amigável da raiz '/' para o prefixo da aplicação em desenvolvimento local
if prefix != '/':
    @server.route('/')
    def root_redirect():
        return flask.redirect(prefix)


# =============================================================================
# SISTEMA DE EQUAÇÕES DIFERENCIAIS ORDINÁRIAS (SIRS COM VACINAÇÃO E MORTALIDADE)
# =============================================================================
def ode_sys(t, state, alpha_disease, beta, gamma, delta, mu, nu, t_begv, t_endv):
    """
    Sistema compartimental SIRS com vacinação temporária e mortalidade por doença.
    
    Estado:
      s: Suscetíveis
      i: Infectados
      r: Recuperados
      d: Óbitos acumulados pela doença
      v_cum: Total de vacinações administradas (acumuladas)
    
    Parâmetros:
      alpha_disease: Taxa de mortalidade induzida pela infecção (ano^-1)
      beta: Coeficiente de transmissão (ano^-1)
      gamma: Período infeccioso em dias (taxa de recuperação = 365 / gamma)
      delta: Período de imunidade em meses (taxa de perda de imunidade = 12 / delta)
      mu: Taxa de mortalidade natural e taxa de natalidade (ano^-1)
      nu: Taxa instantânea de vacinação (ano^-1)
      t_begv, t_endv: Início e fim do programa de vacinação (anos)
    """
    s = max(0.0, float(state[0]))
    i = max(0.0, float(state[1]))
    r = max(0.0, float(state[2]))
    d = max(0.0, float(state[3]))
    v_cum = max(0.0, float(state[4]))
    
    # Pulso de vacinação ativo apenas entre t_begv e t_endv
    vac = nu * heaviside(t - t_begv, 1.0) * heaviside(t_endv - t, 1.0)
    
    n_vivo = s + i + r
    force_inf = (beta * s * i / n_vivo) if n_vivo > 1e-6 else 0.0
    recov = (365.0 / max(0.1, gamma)) * i
    loss_imm = (12.0 / max(0.1, delta)) * r
    vac_flow = vac * s

    # Dinâmica dos compartimentos
    ds_dt = mu * (i + r) - force_inf - vac_flow + loss_imm
    di_dt = force_inf - recov - mu * i - alpha_disease * i
    dr_dt = recov - mu * r + vac_flow - loss_imm
    dd_dt = alpha_disease * i
    dv_cum_dt = vac_flow
    
    return [ds_dt, di_dt, dr_dt, dd_dt, dv_cum_dt]


# =============================================================================
# TEXTOS E DOCUMENTAÇÃO DO MODELO
# =============================================================================
cabecalho = html.Div(
    [
        html.Div(
            [
                html.H1(
                    [
                        html.I(className="fa-solid fa-virus-covid me-2"),
                        "Modelo SIRS com Vacinação e Mortalidade por Doença"
                    ],
                    className="display-6 mb-1"
                ),
                html.P(
                    "Simulador epidemiológico determinístico com perda de imunidade, vacinação intermitente e letalidade induzida por infecção.",
                    className="lead fs-6 mb-0 opacity-75"
                ),
            ],
            className="hero-header p-3 mb-3"
        )
    ]
)

descricao = html.Div([
    dcc.Markdown(
        r'''
        É apresentado um modelo determinístico compartimental do tipo **SIRS** (*Suscetível-Infectado-Recuperado-Suscetível*) com inclusão de estratégia vacinal temporária e taxa de mortalidade pela doença.

        A população viva é dividida em três compartimentos:
        * **$$S$$ (Suscetíveis):** Indivíduos vulneráveis à infecção.
        * **$$I$$ (Infectados):** Indivíduos capazes de transmitir o patógeno.
        * **$$R$$ (Recuperados):** Indivíduos temporariamente imunes após infecção ou vacinação.

        Além disso, o modelo contabiliza:
        * **$$D$$ (Mortes pela doença):** Óbitos acumulados decorrentes diretamente da infecção (letalidade induzida por taxa $$\alpha$$).
        * **$$V_{cum}$$ (Total de vacinados):** Doses/imunizações acumuladas aplicadas aos suscetíveis no período da campanha.

        A imunidade conferida pela infecção natural e pela vacina decai ao longo do tempo com período médio de $$\delta$$ meses, fazendo com que indivíduos recuperados retornem ao compartimento suscetível.
        ''', mathjax=True
    ),
    html.Div([
        html.Img(src=app.get_asset_url('SIR.png'), className="sir-diagram my-2 d-block mx-auto", alt="Diagrama SIR"),
    ]),
    dcc.Markdown(
        r'''
        ### Sistema de Equações Diferenciais:
        $$\begin{aligned}
        \frac{dS}{dt} &= \mu(I + R) - \frac{\beta S I}{N_{vivo}} - \nu(t) S + \frac{12}{\delta} R \\
        \frac{dI}{dt} &= \frac{\beta S I}{N_{vivo}} - \left(\frac{365}{\gamma}\right) I - \mu I - \alpha I \\
        \frac{dR}{dt} &= \left(\frac{365}{\gamma}\right) I - \mu R + \nu(t) S - \frac{12}{\delta} R \\
        \frac{dD}{dt} &= \alpha I \quad \text{(óbitos acumulados pela doença)} \\
        \frac{dV_{cum}}{dt} &= \nu(t) S \quad \text{(imunizações acumuladas)}
        \end{aligned}$$
        em que $$N_{vivo} = S + I + R$$ e $$\nu(t) = \nu_0$$ para $$t \in [t_1, t_2]$$ (e zero fora deste intervalo).
        ''', mathjax=True
    )
])

parametros = dcc.Markdown(
    r'''
    * **$$\beta$$**: Coeficiente de transmissão (taxa de contatos potencialmente infectantes por ano).
    * **Período infeccioso ($$\gamma$$)**: Duração média da infectividade em dias ($$\text{taxa de recuperação} = 365/\gamma \text{ ano}^{-1}$$).
    * **Período de imunidade ($$\delta$$)**: Duração média da imunidade em meses ($$\text{taxa de perda} = 12/\delta \text{ ano}^{-1}$$).
    * **$$\mu$$**: Taxa de mortalidade natural e natalidade ($$ano^{-1}$$).
    * **$$\alpha$$**: Taxa de mortalidade induzida pela doença ($$ano^{-1}$$). Mede a virulência/letalidade do patógeno.
    * **$$\nu_{0}$$**: Taxa de vacinação instantânea durante a campanha ($$ano^{-1}$$).
    * **$$[t_1, t_2]$$**: Anos de início ($$t_1$$) e término ($$t_2$$) da campanha de vacinação.
    * **$$S, I, R$$**: Populações de suscetíveis, infectados e recuperados vivos.
    * **$$D$$**: Total de mortes acumuladas causadas pela doença.
    ''', mathjax=True
)

cond_inicial = dcc.Markdown(
    r'''
    * Coeficiente de transmissão: $$\beta = 80 \space ano^{-1}$$
    * Período infeccioso: $$15 \space \text{dias}$$
    * Período de imunidade: $$6 \space \text{meses}$$
    * Taxa de mortalidade natural / natalidade: $$\mu = 0.10 \space ano^{-1}$$
    * Taxa de mortalidade por doença: $$\alpha = 0.20 \space ano^{-1}$$
    * Taxa de vacinação instantânea: $$\nu_0 = 2.0 \space ano^{-1}$$ entre $$t_1=2$$ e $$t_2=5 \space \text{anos}$$
    * Condições iniciais padrão: $$S(0)=99$$, $$I(0)=1$$, $$R(0)=0$$, $$D(0)=0$$ ($$N_0 = 100$$)
    ''', mathjax=True
)

perguntas = dcc.Markdown(
    r'''
    1. **Estado de equilíbrio sem intervenção:** Adote $$\nu_0 = 0$$ e $$\alpha = 0$$. Em quanto tempo o sistema atinge o estado de equilíbrio? Qual a proporção endêmica de suscetíveis e infectados?
    2. **Impacto do período infeccioso:** Considerando $$\beta = 20 \space ano^{-1}$$ e variando o período infeccioso de $$15$$ para $$30 \space \text{dias}$$, o que ocorre com o número de animais infectados no pico e no equilíbrio?
    3. **Efeito da vacinação temporária:** Retorne à configuração inicial e adote $$\nu_0 = 2.0 \space ano^{-1}$$ com aplicação entre os anos $$2$$ e $$5$$. O que ocorre durante a vacinação? Qual o efeito observado quando a vacinação é encerrada no ano $$5$$?
    4. **Impacto da mortalidade pela doença ($$\alpha$$):** Varie a taxa $$\alpha$$ de $$0.0$$ até $$1.0 \space ano^{-1}$$. Observe como a curva de mortalidade evolui e como ela reduz a população viva remanescente. Como isso altera o pico de infectados?
    5. **Estratégia combinada:** Com $$\alpha = 0.5 \space ano^{-1}$$, qual taxa de vacinação $$\nu_0$$ e cobertura seriam necessárias para minimizar significativamente o total acumulado de mortes?
    ''', mathjax=True
)

textos_descricao = html.Div(
    dbc.Accordion(
        [
            dbc.AccordionItem(descricao, title="📘 Descrição e Formulação do Modelo"),
            dbc.AccordionItem(parametros, title="📊 Parâmetros do Modelo"),
            dbc.AccordionItem(cond_inicial, title="⚙️ Condições Iniciais de Referência"),
            dbc.AccordionItem(perguntas, title="❓ Questões para Estudo e Exploração"),
        ],
        start_collapsed=True,
        flush=False,
    )
)

# =============================================================================
# COMPONENTES DE CONTROLE (SLIDERS)
# =============================================================================
ajuste_parametros_epi = html.Div(
    [
        html.Div(
            [
                html.I(className="fa-solid fa-virus me-2 text-danger"),
                "Parâmetros Epidemiológicos"
            ],
            className="card-header"
        ),
        html.Div(
            [
                html.Div([
                    dbc.Label(dcc.Markdown(r'$$\beta$$: Transmissão ($$ano^{-1}$$)', mathjax=True), className="mb-0"),
                    html.Span(id="badge_beta", className="slider-val-badge")
                ], className="slider-label"),
                dcc.Slider(id="beta", min=0, max=100, step=1, value=80, tooltip={"placement": "bottom", "always_visible": False}),
            ],
            className="slider-group",
        ),
        html.Div(
            [
                html.Div([
                    dbc.Label("Período infeccioso (dias)", className="mb-0"),
                    html.Span(id="badge_gamma", className="slider-val-badge")
                ], className="slider-label"),
                dcc.Slider(id="gamma", min=1, max=60, step=1, value=15, tooltip={"placement": "bottom", "always_visible": False}),
            ],
            className="slider-group",
        ),
        html.Div(
            [
                html.Div([
                    dbc.Label("Período de imunidade (meses)", className="mb-0"),
                    html.Span(id="badge_delta", className="slider-val-badge")
                ], className="slider-label"),
                dcc.Slider(id="delta", min=1, max=36, step=1, value=6, tooltip={"placement": "bottom", "always_visible": False}),
            ],
            className="slider-group",
        ),
        html.Div(
            [
                html.Div([
                    dbc.Label(dcc.Markdown(r'$$\mu$$: Mortalidade natural / natalidade ($$ano^{-1}$$)', mathjax=True), className="mb-0"),
                    html.Span(id="badge_mu", className="slider-val-badge")
                ], className="slider-label"),
                dcc.Slider(id="mu", min=0.01, max=0.50, step=0.01, value=0.10, tooltip={"placement": "bottom", "always_visible": False}),
            ],
            className="slider-group",
        ),
        html.Div(
            [
                html.Div([
                    dbc.Label(dcc.Markdown(r'$$\alpha$$: Mortalidade pela doença ($$ano^{-1}$$)', mathjax=True), className="mb-0 text-danger fw-bold"),
                    html.Span(id="badge_alpha", className="slider-val-badge bg-danger text-white")
                ], className="slider-label"),
                dcc.Slider(id="alpha", min=0.0, max=2.0, step=0.02, value=0.20, tooltip={"placement": "bottom", "always_visible": False}),
            ],
            className="slider-group",
        ),
    ],
    className="card-control mb-3",
)

ajuste_vacinacao = html.Div(
    [
        html.Div(
            [
                html.I(className="fa-solid fa-syringe me-2 text-info"),
                "Estratégia de Vacinação"
            ],
            className="card-header"
        ),
        html.Div(
            [
                html.Div([
                    dbc.Label(dcc.Markdown(r'$$\nu_0$$: Taxa de vacinação ($$ano^{-1}$$)', mathjax=True), className="mb-0 text-info fw-bold"),
                    html.Span(id="badge_nu", className="slider-val-badge bg-info text-white")
                ], className="slider-label"),
                dcc.Slider(id="nu", min=0, max=10, step=0.2, value=2.0, tooltip={"placement": "bottom", "always_visible": False}),
            ],
            className="slider-group",
        ),
        html.Div(
            [
                html.Div([
                    dbc.Label("Período da vacinação (anos)", className="mb-0"),
                    html.Span(id="badge_vacinacao", className="slider-val-badge")
                ], className="slider-label"),
                dcc.RangeSlider(id='vacinacao', min=0, max=10, step=0.5, value=[2, 5], tooltip={"placement": "bottom", "always_visible": False}),
            ],
            className="slider-group",
        ),
    ],
    className="card-control mb-3",
)

ajuste_condicoes_iniciais = html.Div(
    [
        html.Div(
            [
                html.I(className="fa-solid fa-users me-2 text-success"),
                "Ajuste das Condições Iniciais"
            ],
            className="card-header"
        ),
        html.Div(
            [
                html.Div([
                    dbc.Label(dcc.Markdown(r'$$S(0)$$: Suscetíveis iniciais', mathjax=True), className="mb-0"),
                    html.Span(id="badge_s_init", className="slider-val-badge")
                ], className="slider-label"),
                dcc.Slider(id="s_init", min=0, max=500, step=1, value=99, tooltip={"placement": "bottom", "always_visible": False}),
            ],
            className="slider-group",
        ),
        html.Div(
            [
                html.Div([
                    dbc.Label(dcc.Markdown(r'$$I(0)$$: Infectados iniciais', mathjax=True), className="mb-0"),
                    html.Span(id="badge_i_init", className="slider-val-badge")
                ], className="slider-label"),
                dcc.Slider(id="i_init", min=0, max=100, step=1, value=1, tooltip={"placement": "bottom", "always_visible": False}),
            ],
            className="slider-group",
        ),
        html.Div(
            [
                html.Div([
                    dbc.Label(dcc.Markdown(r'$$R(0)$$: Recuperados iniciais', mathjax=True), className="mb-0"),
                    html.Span(id="badge_r_init", className="slider-val-badge")
                ], className="slider-label"),
                dcc.Slider(id="r_init", min=0, max=100, step=1, value=0, tooltip={"placement": "bottom", "always_visible": False}),
            ],
            className="slider-group",
        ),
    ],
    className="card-control mb-3",
)


# =============================================================================
# CARDS DE MÉTRICAS (KPIS)
# =============================================================================
cards_metricas = dbc.Row(
    [
        dbc.Col(
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Total Vacinados", className="kpi-title"),
                            html.Div(id="kpi_vacinados_val", className="kpi-value"),
                            html.Div(id="kpi_vacinados_sub", className="kpi-subtitle"),
                            html.I(className="fa-solid fa-syringe kpi-icon"),
                        ],
                        className="kpi-card-inner"
                    )
                ],
                className="kpi-card kpi-vaccine"
            ),
            width=6, sm=6, lg=3, className="mb-2"
        ),
        dbc.Col(
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Mortes pela Doença", className="kpi-title"),
                            html.Div(id="kpi_mortes_val", className="kpi-value"),
                            html.Div(id="kpi_mortes_sub", className="kpi-subtitle"),
                            html.I(className="fa-solid fa-skull-crossbones kpi-icon"),
                        ],
                        className="kpi-card-inner"
                    )
                ],
                className="kpi-card kpi-deaths"
            ),
            width=6, sm=6, lg=3, className="mb-2"
        ),
        dbc.Col(
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Pico de Infectados", className="kpi-title"),
                            html.Div(id="kpi_pico_val", className="kpi-value"),
                            html.Div(id="kpi_pico_sub", className="kpi-subtitle"),
                            html.I(className="fa-solid fa-virus kpi-icon"),
                        ],
                        className="kpi-card-inner"
                    )
                ],
                className="kpi-card kpi-peak"
            ),
            width=6, sm=6, lg=3, className="mb-2"
        ),
        dbc.Col(
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("População Viva Final", className="kpi-title"),
                            html.Div(id="kpi_vivos_val", className="kpi-value"),
                            html.Div(id="kpi_vivos_sub", className="kpi-subtitle"),
                            html.I(className="fa-solid fa-users kpi-icon"),
                        ],
                        className="kpi-card-inner"
                    )
                ],
                className="kpi-card kpi-alive"
            ),
            width=6, sm=6, lg=3, className="mb-2"
        ),
    ],
    className="mb-2"
)


# =============================================================================
# LAYOUT PRINCIPAL DO DASHBOARD
# =============================================================================
app.layout = dbc.Container(
    [
        cabecalho,
        dbc.Row(
            [
                # Coluna Esquerda: Painel de Parâmetros e Controles (4 colunas)
                dbc.Col(
                    [
                        ajuste_parametros_epi,
                        ajuste_vacinacao,
                        ajuste_condicoes_iniciais,
                    ],
                    width=12, lg=4
                ),
                
                # Coluna Direita: Visualização Principal (8 colunas)
                dbc.Col(
                    [
                        # Linha com os 4 Cards de Métricas (KPIs)
                        cards_metricas,
                        
                        # Card do Gráfico com Seletor de Escala
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.I(className="fa-solid fa-chart-line me-2 text-primary"),
                                                html.Span("Dinâmica do Modelo SIRS com Vacinação e Mortalidade", className="fw-bold text-dark me-3"),
                                            ],
                                            className="d-flex align-items-center mb-1 mb-md-0"
                                        ),
                                        html.Div(
                                            [
                                                dbc.Label("Escala:", className="me-2 mb-0 small text-muted fw-semibold"),
                                                dbc.RadioItems(
                                                    id="unit_display",
                                                    options=[
                                                        {"label": " Proporção (%)", "value": "pct"},
                                                        {"label": " Indivíduos", "value": "abs"},
                                                    ],
                                                    value="pct",
                                                    inline=True,
                                                    className="small"
                                                ),
                                            ],
                                            className="d-flex align-items-center"
                                        )
                                    ],
                                    className="d-flex flex-wrap justify-content-between align-items-center mb-2 px-1"
                                ),
                                dcc.Loading(
                                    dcc.Graph(
                                        id='population_chart',
                                        style={'height': '530px'},
                                        config={
                                            "displayModeBar": True,
                                            "responsive": True,
                                            "modeBarButtonsToRemove": ["lasso2d", "select2d"]
                                        }
                                    ),
                                    type="circle"
                                )
                            ],
                            className="graph-card mb-3"
                        ),
                        
                        # Accordion Teórico e Perguntas de Estudo logo abaixo do gráfico
                        html.Div(textos_descricao, className="mb-4")
                    ],
                    width=12, lg=8
                ),
            ],
            className="g-3"
        ),
    ],
    fluid=True,
    className="py-2 px-3"
)



# =============================================================================
# CALLBACKS DE ATUALIZAÇÃO DOS BADGES DOS SLIDERS
# =============================================================================
@app.callback(
    [
        Output("badge_beta", "children"),
        Output("badge_gamma", "children"),
        Output("badge_delta", "children"),
        Output("badge_mu", "children"),
        Output("badge_alpha", "children"),
        Output("badge_nu", "children"),
        Output("badge_vacinacao", "children"),
        Output("badge_s_init", "children"),
        Output("badge_i_init", "children"),
        Output("badge_r_init", "children"),
    ],
    [
        Input("beta", "value"),
        Input("gamma", "value"),
        Input("delta", "value"),
        Input("mu", "value"),
        Input("alpha", "value"),
        Input("nu", "value"),
        Input("vacinacao", "value"),
        Input("s_init", "value"),
        Input("i_init", "value"),
        Input("r_init", "value"),
    ]
)
def update_badges(beta, gamma, delta, mu, alpha, nu, vacinacao, s_init, i_init, r_init):
    return (
        f"{beta} ano⁻¹",
        f"{gamma} dias",
        f"{delta} meses",
        f"{mu:.2f} ano⁻¹",
        f"{alpha:.2f} ano⁻¹",
        f"{nu:.1f} ano⁻¹",
        f"{vacinacao[0]} a {vacinacao[1]} anos",
        f"{s_init}",
        f"{i_init}",
        f"{r_init}",
    )


# =============================================================================
# CALLBACK PRINCIPAL: INTEGRAÇÃO E RENDERIZAÇÃO
# =============================================================================
@app.callback(
    [
        Output('population_chart', 'figure'),
        Output('kpi_vacinados_val', 'children'),
        Output('kpi_vacinados_sub', 'children'),
        Output('kpi_mortes_val', 'children'),
        Output('kpi_mortes_sub', 'children'),
        Output('kpi_pico_val', 'children'),
        Output('kpi_pico_sub', 'children'),
        Output('kpi_vivos_val', 'children'),
        Output('kpi_vivos_sub', 'children'),
    ],
    [
        Input('s_init', 'value'),
        Input('i_init', 'value'),
        Input('r_init', 'value'),
        Input('beta', 'value'),
        Input('gamma', 'value'),
        Input('delta', 'value'),
        Input('mu', 'value'),
        Input('alpha', 'value'),
        Input('nu', 'value'),
        Input('vacinacao', 'value'),
        Input('unit_display', 'value'),
    ]
)
def simula_e_atualiza(s_init, i_init, r_init, beta, gamma, delta, mu, alpha, nu, vacinacao, unit_display):
    # Proteção de sanidade para entradas nulas ou vazias
    s_0 = float(s_init) if s_init is not None else 99.0
    i_0 = float(i_init) if i_init is not None else 1.0
    r_0 = float(r_init) if r_init is not None else 0.0
    d_0 = 0.0
    v_cum_0 = 0.0
    
    n_0 = max(s_0 + i_0 + r_0, 1e-6)
    
    t_begin = 0.0
    t_end = 10.0
    t_span = (t_begin, t_end)
    t_nsamples = 1000
    t_eval = np.linspace(t_begin, t_end, t_nsamples)
    
    t1_vac, t2_vac = vacinacao[0], vacinacao[1]
    
    # Integração numérica do sistema de equações diferenciais ordinárias
    sol = solve_ivp(
        fun=ode_sys,
        t_span=t_span,
        y0=[s_0, i_0, r_0, d_0, v_cum_0],
        args=(alpha, beta, gamma, delta, mu, nu, t1_vac, t2_vac),
        t_eval=t_eval,
        method='Radau'
    )
    
    # Extração dos resultados
    s_arr = sol.y[0]
    i_arr = sol.y[1]
    r_arr = sol.y[2]
    d_arr = sol.y[3]
    v_arr = sol.y[4]
    
    # Métricas finais e estatísticas de pico
    total_vacinados = float(v_arr[-1])
    total_mortes = float(d_arr[-1])
    pop_viva_final = float(s_arr[-1] + i_arr[-1] + r_arr[-1])
    
    idx_pico = int(np.argmax(i_arr))
    pico_infectados = float(i_arr[idx_pico])
    tempo_pico = float(sol.t[idx_pico])
    
    # Cálculo das curvas conforme unidade selecionada
    if unit_display == 'pct':
        y_s = (s_arr / n_0) * 100.0
        y_i = (i_arr / n_0) * 100.0
        y_r = (r_arr / n_0) * 100.0
        y_d = (d_arr / n_0) * 100.0
        y_axis_label = "Proporção da População Inicial (%)"
        hover_unit = "%"
    else:
        y_s = s_arr
        y_i = i_arr
        y_r = r_arr
        y_d = d_arr
        y_axis_label = "Número de Indivíduos"
        hover_unit = " indivíduos"
        
    # Construção do gráfico com Plotly
    fig = go.Figure()
    
    # Faixa sombreada indicando a campanha de vacinação
    if nu > 0 and t2_vac > t1_vac:
        fig.add_vrect(
            x0=t1_vac, x1=t2_vac,
            fillcolor="#06b6d4", opacity=0.12,
            layer="below", line_width=1, line_dash="dot",
            line_color="#0891b2",
            annotation_text="<b>Campanha de Vacinação</b>",
            annotation_position="top",
            annotation_font=dict(size=11, color="#0891b2")
        )
        
    fig.add_trace(go.Scatter(
        x=sol.t, y=y_s,
        name='Suscetível (S)',
        line=dict(color='#16a34a', width=3.5),
        hovertemplate="Tempo: %{x:.2f} anos<br>Suscetíveis: %{y:.1f}" + hover_unit + "<extra></extra>"
    ))
    
    fig.add_trace(go.Scatter(
        x=sol.t, y=y_i,
        name='Infectado (I)',
        line=dict(color='#dc2626', width=3.5, dash='dot'),
        hovertemplate="Tempo: %{x:.2f} anos<br>Infectados: %{y:.1f}" + hover_unit + "<extra></extra>"
    ))
    
    fig.add_trace(go.Scatter(
        x=sol.t, y=y_r,
        name='Recuperado (R)',
        line=dict(color='#2563eb', width=3.5, dash='dashdot'),
        hovertemplate="Tempo: %{x:.2f} anos<br>Recuperados: %{y:.1f}" + hover_unit + "<extra></extra>"
    ))
    
    fig.add_trace(go.Scatter(
        x=sol.t, y=y_d,
        name='Mortes por Doença (D)',
        line=dict(color='#111827', width=3.5, dash='dash'),
        hovertemplate="Tempo: %{x:.2f} anos<br>Óbitos acumulados: %{y:.1f}" + hover_unit + "<extra></extra>"
    ))
    
    fig.update_layout(
        template='plotly_white',
        title=dict(
            text='<b>Evolução Temporal das Populações</b>',
            font=dict(size=14, color="#1e293b"),
            x=0.02,
            xanchor='left',
            y=0.96,
            yref='container',
            yanchor='top'
        ),
        xaxis=dict(
            title='Tempo (anos)',
            range=[0, 10],
            gridcolor='#f1f5f9',
            zerolinecolor='#cbd5e1'
        ),
        yaxis=dict(
            title=y_axis_label,
            range=[0, 105 if unit_display == 'pct' else max(100.0, n_0 * 1.05)],
            gridcolor='#f1f5f9',
            zerolinecolor='#cbd5e1'
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#e2e8f0',
            borderwidth=1
        ),
        margin=dict(l=55, r=30, t=55, b=85),
        hovermode="x unified"
    )
    
    # Formatação dos cards de métricas
    # Total de vacinados
    card_vac_val = f"{total_vacinados:.1f}"
    card_vac_sub = f"aplicações acumuladas (ν₀={nu:.1f}/ano)" if nu > 0 else "vacinação desativada (ν₀=0)"
    
    # Mortes por doença
    card_mortes_val = f"{total_mortes:.1f}"
    pct_mortes = (total_mortes / n_0) * 100.0
    card_mortes_sub = f"{pct_mortes:.1f}% da população inicial (α={alpha:.2f}/ano)"
    
    # Pico de infectados
    pico_pct = (pico_infectados / n_0) * 100.0
    card_pico_val = f"{pico_pct:.1f}%" if unit_display == 'pct' else f"{pico_infectados:.1f}"
    card_pico_sub = f"atingido no ano {tempo_pico:.2f}"
    
    # População viva final
    pop_pct = (pop_viva_final / n_0) * 100.0
    card_vivos_val = f"{pop_viva_final:.1f}"
    card_vivos_sub = f"{pop_pct:.1f}% de N₀ ({s_arr[-1]:.0f} S, {i_arr[-1]:.0f} I, {r_arr[-1]:.0f} R)"
    
    return (
        fig,
        card_vac_val, card_vac_sub,
        card_mortes_val, card_mortes_sub,
        card_pico_val, card_pico_sub,
        card_vivos_val, card_vivos_sub
    )


if __name__ == '__main__':
    # Execução local do servidor Dash
    port = int(os.getenv('PORT', 8050))
    host = os.getenv('HOST', '0.0.0.0')
    app.run(debug=False, port=port, host=host)
