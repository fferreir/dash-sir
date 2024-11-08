import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from numpy import heaviside
from textwrap import dedent
import plotly.graph_objects as go


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUMEN, dbc.icons.FONT_AWESOME])

cabecalho = html.H1("Modelo SIRS com vacinação",className="bg-primary text-white p-2 mb-4")

descricao = dcc.Markdown(
    '''
    É apresentado um modelo determinístico do tipo *SIRS* com vacinação. Neste modelo a população é dividida em três 
    compartimentos: suscetíveis ($$S$$), infectados($$I$$) e recuperados ($$R$$). A população é considerada fechada, 
    ou seja, a taxa de mortalidade é igual a de natalidade e a doença não causa mortalidade significativa. A imunidade
    conferida pela infecção e/ou vacinação não é permanente.
    ''', mathjax=True
)

parametros = dcc.Markdown(
    '''
    * $$\\alpha$$: taxa de natalidade = $$\\mu$$: taxa de mortalidade
    * $$\\beta$$: coeficiente de transmissão (taxa de contatos potencialmente infectantes)
    * $$\\gamma$$: taxa de recuperação = inverso do período infeccioso
    * $$\\delta$$: taxa de perda de imunidade = inverso do período de imunidade
    * $$\\nu_{0}$$: taxa de vacinação instantânea (taxa relacionada à proporção de cobertura vacinal)
    * $$t_1$$ e $$t_2$$: $$t_1$$ ano de início da aplicação da vacina e $$t_2$$ ano final de aplicação da vacina
    * $$S$$: número de indivíduos suscetíveis
    * $$I$$: número de indivíduos infectados
    * $$R$$: número de indivíduos recuperados
    ''', mathjax=True
)
cond_inicial = dcc.Markdown(
    '''
    * coeficiente de transmissão: $$\\beta=0.5 \space ano^{-1}$$
    * taxa de recuperação: $$\\gamma=0.01 \space ano^{-1}$$
    * taxa de vacinação: $$\\nu_0=0.0 \space ano^{-1}$$, $$t_1=20$$ e $$t_2=50$$
    * texa de perda de imunidade: $$\delta=1 \space ano^{-1}$$
    * taxa de natalidade (= taxa de mortalidade): $$\\alpha=\\mu=0.15 \space ano^{-1}$$
    * condições iniciais: $$S(0)=100$$, $$I(0)=1$$, $$R(0)=0$$
    ''', mathjax=True
)

perguntas = dcc.Markdown(
    '''
    1. Para a configuração inicial dos parâmetros, em quanto tempo
    se atinge o estado de equilíbrio? Estime a proporção de animais positivos e o número de animais infectados.
    2. Responda às mesmas questões do item 1, considerando uma taxa de recuperação de $$\\gamma=1 \space ano^{-1}$$
    e $$\\beta=0.05 \space ano^{-1}$$.
    3. Retornando à configuração inicial de parâmetros, adote $$\\nu_0=1.0 \space ano^{-1}$$ e
    $$\gamma=0.01 \space ano^{-1}$$. Qual o efeito sobre o número de suscetíveis de se abandonar o programa de vacinação?
    Como fica o número de animais infectados?
    4. Reduzindo o período infeccioso, por meio do aumento da taxa de recuperação para $$\\gamma=0.1 \space ano^{-1}$$ e
    diminuindo a taxa de vacinação para $$\\nu_0=0.2 \space ano^{-1}$$, observe o que ocorre com o número de suscetíveis
    e infectados.
    ''', mathjax=True
)

textos_descricao = html.Div(
        dbc.Accordion(
            [
                dbc.AccordionItem(
                    descricao, title="Descrição do modelo"
                ),
                dbc.AccordionItem(
                    parametros, title="Parâmetros do modelo"
                ),
                dbc.AccordionItem(
                    cond_inicial, title="Condições iniciais"
                ),
                dbc.AccordionItem(
                    perguntas, title="Perguntas"
                ),
            ],
            start_collapsed=True,
        )
    )

ajuste_condicoes_iniciais = html.Div(
        [
            html.P("Ajuste das condições iniciais", className="card-header border-dark mb-3"),
            html.Div(
                [
                    dbc.Label(dcc.Markdown('''$$S$$ total de suscetíveis''', mathjax=True), html_for="s_init"),
                    dcc.Slider(id="s_init", min=0, max=10000, value=10000, tooltip={"placement": "bottom", "always_visible": False}),
                ],
                className="m-2",
            ),
            html.Div(
                [
                    dbc.Label(dcc.Markdown('''$$I$$ total de infectados ''', mathjax=True), html_for="i_init"),
                    dcc.Slider(id="i_init", min=0, max=10000, value=10, tooltip={"placement": "bottom", "always_visible": False}, className="card-text"),
                ],
                className="m-1",
            ),
            html.Div(
                [
                    dbc.Label(dcc.Markdown('''$$R$$ total de recuperados ''', mathjax=True), html_for="r_init"),
                    dcc.Slider(id="r_init", min=0, max=10000, value=0, tooltip={"placement": "bottom", "always_visible": False}, className="card-text"),
                ],
                className="m-1",
            ),

        ],
        className="card border-dark mb-3",
    )

ajuste_parametros = html.Div(
        [
            html.P("Ajuste dos parâmetros", className="card-header border-dark mb-3"),
            html.Div(
                [
                    dbc.Label(dcc.Markdown('''$$\\alpha$$ taxa de natalidade e $$\\mu$$ taxa de mortalidade''', mathjax=True), html_for="alpha"),
                    dcc.Slider(id="alpha", min=0.1, max=0.15, value=0.2, tooltip={"placement": "bottom", "always_visible": False}),
                ],
                className="m-2",
            ),
            html.Div(
                [
                    dbc.Label(dcc.Markdown('''$$\\beta$$ coeficiente de transmissão ''', mathjax=True), html_for="beta"),
                    dcc.Slider(id="beta", min=60, max=100, value=80, tooltip={"placement": "bottom", "always_visible": False}, className="card-text"),
                ],
                className="m-1",
            ),
            html.Div(
                [
                    dbc.Label(dcc.Markdown('''$$\\gamma$$ taxa de recuperação ''', mathjax=True), html_for="gamma"),
                    dcc.Slider(id="gamma", min=18, max=36, value=20, tooltip={"placement": "bottom", "always_visible": False}, className="card-text"),
                ],
                className="m-1",
            ),
            html.Div(
                [
                    dbc.Label(dcc.Markdown('''$$\\delta$$ taxa de perda de imunidade ''', mathjax=True), html_for="delta"),
                    dcc.Slider(id="delta", min=0.5, max=2, value=1, tooltip={"placement": "bottom", "always_visible": False}, className="card-text"),
                ],
                className="m-1",
            ),
            html.Div(
                [
                    dbc.Label(dcc.Markdown('''$$\\nu_0$$ taxa de vacinação instantânea''', mathjax=True), html_for="nu"),
                    dcc.Slider(id="nu", min=0, max=10, value=5, tooltip={"placement": "bottom", "always_visible": False}, className="card-text"),
                ],
                className="m-1",
            ),
            html.Div(
                [
                    dbc.Label(dcc.Markdown('''Período de aplicação da vacina''', mathjax=True), html_for="vacinacao"),
                    dcc.RangeSlider(id='vacinacao', min=0, max=10, step=1, value=[2, 5], tooltip={"placement": "bottom", "always_visible": False}, className="card-text" ),
                ],
                className="m-1",
            ),
        ],
        className="card border-dark mb-3",
    )

def ode_sys(t, state, alpha, beta, gamma, delta, mu, nu, t_begv, t_endv):
    s, i, r=state
    vac=nu*heaviside(t-t_begv, 1)*heaviside(t_endv-t, 1)
    ds_dt=mu*(i+r)-beta*s*i/(s+i+r)-vac*s+delta*r
    di_dt=beta*s*i/(s+i+r)-gamma*i-mu*i
    dr_dt=gamma*i-mu*r+vac*s-delta*r
    return [ds_dt, di_dt, dr_dt]

@app.callback(Output('population_chart', 'figure'),
              [Input('s_init', 'value'),
              Input('i_init', 'value'),
              Input('r_init', 'value'),
              Input('alpha', 'value'),
              Input('beta', 'value'),
              Input('gamma', 'value'),
              Input('delta', 'value'),
              Input('nu', 'value'),
              Input('vacinacao', 'value')])
def gera_grafico(s_init, i_init, r_init, alpha, beta, gamma, delta, nu, vacinacao):
    t_begin = 0.
    t_end = 10.
    t_nsamples = 1000
    t_space = np.linspace(t_begin, t_end, t_nsamples)
    mu = alpha
    sol = odeint(ode_sys, [s_init, i_init, r_init], t_space, (alpha, beta, gamma, delta, mu, nu, vacinacao[0], vacinacao[1]),
                 tfirst=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_space, y=sol[:,0]/(sol[:,0]+sol[:,1]+sol[:,2])*100, name='Suscetível',
                             line=dict(color='#00b400', width=4)))
    fig.add_trace(go.Scatter(x=t_space, y=sol[:,1]/(sol[:,0]+sol[:,1]+sol[:,2])*100, name ='Infectado',
                             line=dict(color='#ff0000', width=4)))
    fig.add_trace(go.Scatter(x=t_space, y=sol[:,2]/(sol[:,0]+sol[:,1]+sol[:,2])*100, name='Recuperado',
                             line=dict(color='#0000ff', width=4)))
    fig.update_layout(title='Dinâmica Modelo SIRS com vacinação',
                       xaxis_title='Tempo',
                       yaxis_title='Proporção da população')
    return fig

app.layout = dbc.Container([
                cabecalho,
                dbc.Row([
                        dbc.Col(html.Div(ajuste_parametros), width=3),
                        dbc.Col(html.Div([ajuste_condicoes_iniciais,html.Div(textos_descricao)]), width=3),
                        dbc.Col(dcc.Graph(id='population_chart'), width=6),
                        ]),
              ], fluid=True),


if __name__ == '__main__':
    app.run(port=8050, host='localhost', debug=True)
