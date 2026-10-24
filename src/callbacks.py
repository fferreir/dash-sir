"""
=============================================================================
Callbacks da Aplicação: Modelo SIRS com Vacinação e Mortalidade
=============================================================================
Gerencia a reatividade da interface:
1. Carregamento instantâneo de cenários didáticos.
2. Atualização dos badges numéricos dos controles.
3. Integração numérica e renderização gráfica interativa com Plotly e KPIs.
"""

import dash
from dash import Input, Output, callback_context
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

from src.model import simular_sir


def register_callbacks(app):
    """
    Registra todos os callbacks reativos na instância da aplicação Dash.
    """

    # =========================================================================
    # CALLBACK 1: CARREGAMENTO DE CENÁRIOS RÁPIDOS / QUESTÕES DIDÁTICAS
    # =========================================================================
    @app.callback(
        [
            Output('beta', 'value'),
            Output('gamma', 'value'),
            Output('delta', 'value'),
            Output('mu', 'value'),
            Output('alpha', 'value'),
            Output('nu', 'value'),
            Output('vacinacao', 'value'),
            Output('s_init', 'value'),
            Output('i_init', 'value'),
            Output('r_init', 'value'),
        ],
        [
            Input('btn_padrao', 'n_clicks'),
            Input('btn_q1', 'n_clicks'),
            Input('btn_q2', 'n_clicks'),
            Input('btn_q4', 'n_clicks'),
            Input('btn_q5', 'n_clicks'),
        ],
        prevent_initial_call=True
    )
    def carregar_cenario(btn_padrao, btn_q1, btn_q2, btn_q4, btn_q5):
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate

        botao_acionado = ctx.triggered[0]['prop_id'].split('.')[0]

        # Retorna: beta, gamma, delta, mu, alpha, nu, vacinacao, s_init, i_init, r_init
        if botao_acionado == 'btn_padrao':
            return 80, 15, 6, 0.10, 0.20, 2.0, [2, 5], 99, 1, 0

        elif botao_acionado == 'btn_q1':
            # Q1: Sem intervenção (nu_0 = 0 e alpha = 0)
            return 80, 15, 6, 0.10, 0.0, 0.0, [2, 5], 99, 1, 0

        elif botao_acionado == 'btn_q2':
            # Q2: Transmissão mais lenta e período infeccioso mais longo (beta = 20, gamma = 30)
            return 20, 30, 6, 0.10, 0.20, 2.0, [2, 5], 99, 1, 0

        elif botao_acionado == 'btn_q4':
            # Q4: Alta letalidade pela doença (alpha = 1.0)
            return 80, 15, 6, 0.10, 1.0, 2.0, [2, 5], 99, 1, 0

        elif botao_acionado == 'btn_q5':
            # Q5: Estratégia combinada intensiva (alpha = 0.5, nu_0 = 5.0, janela ampliada [1, 6])
            return 80, 15, 6, 0.10, 0.50, 5.0, [1, 6], 99, 1, 0

        raise PreventUpdate

    # =========================================================================
    # CALLBACK 2: ATUALIZAÇÃO DOS BADGES DOS SLIDERS
    # =========================================================================
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
        vac_str = f"{vacinacao[0]} a {vacinacao[1]} anos" if vacinacao else "desativada"
        return (
            f"{beta} ano⁻¹",
            f"{gamma} dias",
            f"{delta} meses",
            f"{mu:.2f} ano⁻¹",
            f"{alpha:.2f} ano⁻¹",
            f"{nu:.1f} ano⁻¹",
            vac_str,
            f"{s_init}",
            f"{i_init}",
            f"{r_init}",
        )

    # =========================================================================
    # CALLBACK 3: INTEGRAÇÃO E RENDERIZAÇÃO GRÁFICA + KPIS
    # =========================================================================
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
        # Executa simulação numérica através do módulo matemático
        res = simular_sir(
            s_init=s_init,
            i_init=i_init,
            r_init=r_init,
            beta=beta,
            gamma=gamma,
            delta=delta,
            mu=mu,
            alpha=alpha,
            nu=nu,
            vacinacao=vacinacao,
            t_end=10.0,
            n_samples=1000
        )

        t_arr = res["t"]
        n_0 = res["n_0"]
        total_vacinados = res["total_vacinados"]
        total_mortes = res["total_mortes"]
        pop_viva_final = res["pop_viva_final"]
        pico_infectados = res["pico_infectados"]
        tempo_pico = res["tempo_pico"]
        t1_vac = res["t1_vac"]
        t2_vac = res["t2_vac"]

        # Configura curvas conforme unidade selecionada (proporção % ou contagem absoluta)
        if unit_display == 'pct':
            y_s = res["prop_s"]
            y_i = res["prop_i"]
            y_r = res["prop_r"]
            y_d = res["prop_d"]
            y_axis_label = "Proporção da População Inicial (%)"
            hover_unit = "%"
        else:
            y_s = res["s"]
            y_i = res["i"]
            y_r = res["r"]
            y_d = res["d"]
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
            x=t_arr, y=y_s,
            name='Suscetível (S)',
            line=dict(color='#16a34a', width=3.5),
            hovertemplate="Tempo: %{x:.2f} anos<br>Suscetíveis: %{y:.1f}" + hover_unit + "<extra></extra>"
        ))

        fig.add_trace(go.Scatter(
            x=t_arr, y=y_i,
            name='Infectado (I)',
            line=dict(color='#dc2626', width=3.5, dash='dot'),
            hovertemplate="Tempo: %{x:.2f} anos<br>Infectados: %{y:.1f}" + hover_unit + "<extra></extra>"
        ))

        fig.add_trace(go.Scatter(
            x=t_arr, y=y_r,
            name='Recuperado (R)',
            line=dict(color='#2563eb', width=3.5, dash='dashdot'),
            hovertemplate="Tempo: %{x:.2f} anos<br>Recuperados: %{y:.1f}" + hover_unit + "<extra></extra>"
        ))

        fig.add_trace(go.Scatter(
            x=t_arr, y=y_d,
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

        # Formatação dos cards de métricas (KPIs)
        card_vac_val = f"{total_vacinados:.1f}"
        card_vac_sub = f"aplicações acumuladas (ν₀={nu:.1f}/ano)" if nu > 0 else "vacinação desativada (ν₀=0)"

        card_mortes_val = f"{total_mortes:.1f}"
        pct_mortes = (total_mortes / n_0) * 100.0
        card_mortes_sub = f"{pct_mortes:.1f}% da população inicial (α={alpha:.2f}/ano)"

        pico_pct = (pico_infectados / n_0) * 100.0
        card_pico_val = f"{pico_pct:.1f}%" if unit_display == 'pct' else f"{pico_infectados:.1f}"
        card_pico_sub = f"atingido no ano {tempo_pico:.2f}"

        pop_pct = (pop_viva_final / n_0) * 100.0
        card_vivos_val = f"{pop_viva_final:.1f}"
        card_vivos_sub = f"{pop_pct:.1f}% de N₀ ({res['s'][-1]:.0f} S, {res['i'][-1]:.0f} I, {res['r'][-1]:.0f} R)"

        return (
            fig,
            card_vac_val, card_vac_sub,
            card_mortes_val, card_mortes_sub,
            card_pico_val, card_pico_sub,
            card_vivos_val, card_vivos_sub
        )
