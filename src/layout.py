"""
=============================================================================
Layout da Interface Visual: Modelo SIRS com Vacinação e Mortalidade
=============================================================================
Define os componentes de apresentação: Hero Header, Cards de Métricas (KPIs),
Painéis de Controle com Sliders, Botões de Cenários Didáticos, Gráfico Principal,
Acordeom de Fundamentação Teórica e Rodapé Institucional com Link para o GitHub.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc


# =============================================================================
# CABEÇALHO HERO
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
# PAINEL DE CENÁRIOS RÁPIDOS E QUESTÕES PEDAGÓGICAS
# =============================================================================
ajuste_cenarios = html.Div(
    [
        html.Div(
            [
                html.I(className="fa-solid fa-bookmark me-2 text-primary"),
                "Cenários Rápidos e Perguntas de Estudo"
            ],
            className="card-header"
        ),
        html.Div(
            [
                html.P(
                    "Carregue com um clique os cenários exatos das questões pedagógicas:",
                    className="small text-muted mb-2"
                ),
                html.Div(
                    [
                        dbc.Button(
                            "Padrão (Referência)",
                            id="btn_padrao",
                            size="sm",
                            color="primary",
                            outline=True,
                            className="btn-scenario"
                        ),
                        dbc.Button(
                            "Q1: Sem Intervenção (ν₀=0, α=0)",
                            id="btn_q1",
                            size="sm",
                            color="secondary",
                            outline=True,
                            className="btn-scenario"
                        ),
                        dbc.Button(
                            "Q2: Transmissão Lenta (β=20, γ=30d)",
                            id="btn_q2",
                            size="sm",
                            color="warning",
                            outline=True,
                            className="btn-scenario"
                        ),
                        dbc.Button(
                            "Q4: Alta Letalidade (α=1.0)",
                            id="btn_q4",
                            size="sm",
                            color="danger",
                            outline=True,
                            className="btn-scenario"
                        ),
                        dbc.Button(
                            "Q5: Estratégia Combinada (α=0.5, ν₀=5)",
                            id="btn_q5",
                            size="sm",
                            color="success",
                            outline=True,
                            className="btn-scenario"
                        ),
                    ],
                    className="d-flex flex-wrap"
                )
            ],
            className="p-3"
        )
    ],
    className="card-control mb-3"
)


# =============================================================================
# COMPONENTES DE CONTROLE (SLIDERS E BADGES)
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
                    html.Span(id="badge_beta", className="slider-val-badge badge-blue")
                ], className="slider-label"),
                dcc.Slider(id="beta", min=0, max=100, step=1, value=80, tooltip={"placement": "bottom", "always_visible": False}),
            ],
            className="slider-group",
        ),
        html.Div(
            [
                html.Div([
                    dbc.Label("Período infeccioso (dias)", className="mb-0"),
                    html.Span(id="badge_gamma", className="slider-val-badge badge-blue")
                ], className="slider-label"),
                dcc.Slider(id="gamma", min=1, max=60, step=1, value=15, tooltip={"placement": "bottom", "always_visible": False}),
            ],
            className="slider-group",
        ),
        html.Div(
            [
                html.Div([
                    dbc.Label("Período de imunidade (meses)", className="mb-0"),
                    html.Span(id="badge_delta", className="slider-val-badge badge-blue")
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
                    html.Span(id="badge_alpha", className="slider-val-badge badge-red")
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
                    html.Span(id="badge_nu", className="slider-val-badge badge-blue")
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
                    html.Span(id="badge_s_init", className="slider-val-badge badge-green")
                ], className="slider-label"),
                dcc.Slider(id="s_init", min=0, max=500, step=1, value=99, tooltip={"placement": "bottom", "always_visible": False}),
            ],
            className="slider-group",
        ),
        html.Div(
            [
                html.Div([
                    dbc.Label(dcc.Markdown(r'$$I(0)$$: Infectados iniciais', mathjax=True), className="mb-0"),
                    html.Span(id="badge_i_init", className="slider-val-badge badge-orange")
                ], className="slider-label"),
                dcc.Slider(id="i_init", min=0, max=100, step=1, value=1, tooltip={"placement": "bottom", "always_visible": False}),
            ],
            className="slider-group",
        ),
        html.Div(
            [
                html.Div([
                    dbc.Label(dcc.Markdown(r'$$R(0)$$: Recuperados iniciais', mathjax=True), className="mb-0"),
                    html.Span(id="badge_r_init", className="slider-val-badge badge-blue")
                ], className="slider-label"),
                dcc.Slider(id="r_init", min=0, max=100, step=1, value=0, tooltip={"placement": "bottom", "always_visible": False}),
            ],
            className="slider-group",
        ),
    ],
    className="card-control mb-3",
)


# =============================================================================
# DOCUMENTAÇÃO TEÓRICA E ACORDEOM DIDÁTICO
# =============================================================================
def _criar_secao_descricao(image_url):
    return html.Div([
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
            html.Img(src=image_url, className="sir-diagram my-2 d-block mx-auto", alt="Diagrama SIR"),
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

perguntas = html.Div(
    [
        html.Div(
            [
                html.Div("1. Estado de equilíbrio sem intervenção", className="study-question-title"),
                dcc.Markdown(
                    r"Adote $$\nu_0 = 0$$ e $$\alpha = 0$$. Em quanto tempo o sistema atinge o estado de equilíbrio? Qual a proporção endêmica de suscetíveis e infectados?",
                    mathjax=True
                )
            ],
            className="study-question-box"
        ),
        html.Div(
            [
                html.Div("2. Impacto do período infeccioso", className="study-question-title"),
                dcc.Markdown(
                    r"Considerando $$\beta = 20 \space ano^{-1}$$ e variando o período infeccioso de $$15$$ para $$30 \space \text{dias}$$, o que ocorre com o número de animais infectados no pico e no equilíbrio?",
                    mathjax=True
                )
            ],
            className="study-question-box"
        ),
        html.Div(
            [
                html.Div("3. Efeito da vacinação temporária", className="study-question-title"),
                dcc.Markdown(
                    r"Retorne à configuração inicial e adote $$\nu_0 = 2.0 \space ano^{-1}$$ com aplicação entre os anos $$2$$ e $$5$$. O que ocorre durante a vacinação? Qual o efeito observado quando a vacinação é encerrada no ano $$5$$?",
                    mathjax=True
                )
            ],
            className="study-question-box"
        ),
        html.Div(
            [
                html.Div("4. Impacto da mortalidade pela doença (α)", className="study-question-title"),
                dcc.Markdown(
                    r"Varie a taxa $$\alpha$$ de $$0.0$$ até $$1.0 \space ano^{-1}$$. Observe como a curva de mortalidade evolui e como ela reduz a população viva remanescente. Como isso altera o pico de infectados?",
                    mathjax=True
                )
            ],
            className="study-question-box"
        ),
        html.Div(
            [
                html.Div("5. Estratégia combinada", className="study-question-title"),
                dcc.Markdown(
                    r"Com $$\alpha = 0.5 \space ano^{-1}$$, qual taxa de vacinação $$\nu_0$$ e cobertura seriam necessárias para minimizar significativamente o total acumulado de mortes?",
                    mathjax=True
                )
            ],
            className="study-question-box"
        ),
    ]
)


def _criar_acordeom(image_url):
    return html.Div(
        dbc.Accordion(
            [
                dbc.AccordionItem(_criar_secao_descricao(image_url), title="📘 Descrição e Formulação do Modelo"),
                dbc.AccordionItem(parametros, title="📊 Parâmetros do Modelo"),
                dbc.AccordionItem(cond_inicial, title="⚙️ Condições Iniciais de Referência"),
                dbc.AccordionItem(perguntas, title="❓ Questões para Estudo e Exploração"),
            ],
            start_collapsed=True,
            flush=False,
        )
    )


# =============================================================================
# RODAPÉ INSTITUCIONAL (FOOTER COM LINK PARA O GITHUB)
# =============================================================================
rodape = html.Footer(
    html.Div(
        [
            html.Small(
                [
                    "Simulador do Modelo SIRS com Vacinação e Mortalidade por Doença • Desenvolvido com Dash, Plotly, SciPy e Bootstrap • ",
                    html.A(
                        [
                            html.I(className="fa-brands fa-github me-1"),
                            "Código Aberto no GitHub"
                        ],
                        href="https://github.com/fferreir/dash-sir",
                        target="_blank",
                        className="text-decoration-none text-muted fw-semibold"
                    )
                ],
                className="text-muted"
            )
        ],
        className="text-center py-3 mt-4 border-top"
    )
)


# =============================================================================
# FUNÇÃO PRINCIPAL DE CRIAÇÃO DO LAYOUT
# =============================================================================
def create_layout(app=None):
    """
    Constrói a árvore de componentes da aplicação Dash.
    """
    if app is not None and hasattr(app, 'get_asset_url'):
        image_url = app.get_asset_url('SIR.png')
    else:
        image_url = "assets/SIR.png"

    textos_descricao = _criar_acordeom(image_url)

    return dbc.Container(
        [
            cabecalho,
            cards_metricas,
            dbc.Row(
                [
                    # Coluna lateral de controles (4 colunas em telas grandes)
                    dbc.Col(
                        [
                            ajuste_cenarios,
                            ajuste_parametros_epi,
                            ajuste_vacinacao,
                            ajuste_condicoes_iniciais,
                        ],
                        width=12, lg=4
                    ),

                    # Coluna principal do gráfico e documentação (8 colunas em telas grandes)
                    dbc.Col(
                        [
                            # Card do Gráfico com Seletor de Escala
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.I(className="fa-solid fa-chart-line me-2 text-primary"),
                                                    html.Span(
                                                        "Dinâmica do Modelo SIRS com Vacinação e Mortalidade",
                                                        className="fw-bold text-dark me-3"
                                                    ),
                                                ],
                                                className="d-flex align-items-center mb-1 mb-md-0"
                                            ),
                                            html.Div(
                                                [
                                                    dbc.Label(
                                                        "Escala:",
                                                        className="me-2 mb-0 small text-muted fw-semibold"
                                                    ),
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
                                            style={'height': '540px'},
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

                            # Acordeom Pedagógico e Teórico logo abaixo do gráfico
                            html.Div(textos_descricao, className="mb-4")
                        ],
                        width=12, lg=8
                    ),
                ],
                className="g-3"
            ),
            rodape,
        ],
        fluid=True,
        className="py-2 px-3"
    )
