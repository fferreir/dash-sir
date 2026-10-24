"""
=============================================================================
Módulo Matemático e Numérico: Modelo SIRS com Vacinação e Mortalidade
=============================================================================
Contém as Equações Diferenciais Ordinárias (EDOs), rotinas de integração
numérica via solve_ivp (método Radau para sistemas rígidos/stiff) e o cálculo
de métricas epidemiológicas fundamentais (R0, limiar crítico e indicadores).
"""

import numpy as np
from scipy.integrate import solve_ivp
from numpy import heaviside


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
      t: Tempo em anos
      alpha_disease: Taxa de mortalidade induzida pela infecção (ano^-1)
      beta: Coeficiente de transmissão / contatos infectantes (ano^-1)
      gamma: Período infeccioso em dias (taxa de recuperação = 365 / gamma)
      delta: Período de imunidade em meses (taxa de perda de imunidade = 12 / delta)
      mu: Taxa de mortalidade natural e taxa de natalidade (ano^-1)
      nu: Taxa instantânea de vacinação (ano^-1)
      t_begv, t_endv: Início e fim do programa de vacinação (anos)

    Retorna:
      [ds_dt, di_dt, dr_dt, dd_dt, dv_cum_dt]
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


def calcular_metricas_epidemiologicas(beta, gamma, delta, mu, alpha, nu=0.0):
    """
    Calcula métricas analíticas e limiares clássicos do modelo epidemiológico SIRS.

    Retorna um dicionário com:
      - r0: Número básico de reprodução (beta / (365/gamma + mu + alpha))
      - taxa_recuperacao: Taxa anual de recuperação (365 / gamma)
      - taxa_perda_imunidade: Taxa anual de perda de imunidade (12 / delta)
      - p_critico_vac: Cobertura vacinal crítica para erradicação (1 - 1/R0)
      - status_epidemia: 'Potencial Epidêmico / Endêmico' ou 'Extinção / Subcrítico'
    """
    beta = float(beta)
    gamma = max(0.1, float(gamma))
    delta = max(0.1, float(delta))
    mu = float(mu)
    alpha = float(alpha)

    taxa_recuperacao = 365.0 / gamma
    taxa_perda_imunidade = 12.0 / delta
    taxa_remocao = taxa_recuperacao + mu + alpha

    if taxa_remocao > 0:
        r0 = beta / taxa_remocao
    else:
        r0 = float('inf')

    if r0 > 1.0:
        p_critico = (1.0 - (1.0 / r0)) * 100.0
        status_epidemia = "Potencial Epidêmico / Endêmico"
    else:
        p_critico = 0.0
        status_epidemia = "Extinção da Infecção (Subcrítico)"

    return {
        "r0": r0,
        "taxa_recuperacao": taxa_recuperacao,
        "taxa_perda_imunidade": taxa_perda_imunidade,
        "p_critico_vac": p_critico,
        "status_epidemia": status_epidemia
    }


def simular_sir(
    s_init=99.0,
    i_init=1.0,
    r_init=0.0,
    beta=80.0,
    gamma=15.0,
    delta=6.0,
    mu=0.10,
    alpha=0.20,
    nu=2.0,
    vacinacao=None,
    t_end=10.0,
    n_samples=1000
):
    """
    Executa a simulação numérica temporal completa do modelo SIRS com vacinação e mortalidade.

    Retorna um dicionário com trajetórias temporais, indicadores acumulados e estatísticas de pico.
    """
    if vacinacao is None:
        vacinacao = [2.0, 5.0]

    s_0 = float(s_init) if s_init is not None else 99.0
    i_0 = float(i_init) if i_init is not None else 1.0
    r_0 = float(r_init) if r_init is not None else 0.0
    d_0 = 0.0
    v_cum_0 = 0.0

    n_0 = max(s_0 + i_0 + r_0, 1e-6)

    t_begin = 0.0
    t_end = max(1.0, float(t_end))
    t_span = (t_begin, t_end)
    t_eval = np.linspace(t_begin, t_end, int(n_samples))

    t1_vac, t2_vac = float(vacinacao[0]), float(vacinacao[1])

    # Integração numérica com método implícito Radau (ideal para saltos de Heaviside)
    sol = solve_ivp(
        fun=ode_sys,
        t_span=t_span,
        y0=[s_0, i_0, r_0, d_0, v_cum_0],
        args=(float(alpha), float(beta), float(gamma), float(delta), float(mu), float(nu), t1_vac, t2_vac),
        t_eval=t_eval,
        method='Radau'
    )

    s_arr = np.maximum(0.0, sol.y[0])
    i_arr = np.maximum(0.0, sol.y[1])
    r_arr = np.maximum(0.0, sol.y[2])
    d_arr = np.maximum(0.0, sol.y[3])
    v_arr = np.maximum(0.0, sol.y[4])

    # Totais acumulados e métricas finais
    total_vacinados = float(v_arr[-1])
    total_mortes = float(d_arr[-1])
    pop_viva_final = float(s_arr[-1] + i_arr[-1] + r_arr[-1])

    idx_pico = int(np.argmax(i_arr))
    pico_infectados = float(i_arr[idx_pico])
    tempo_pico = float(sol.t[idx_pico])

    # Proporções (%) da população inicial N0
    prop_s = (s_arr / n_0) * 100.0
    prop_i = (i_arr / n_0) * 100.0
    prop_r = (r_arr / n_0) * 100.0
    prop_d = (d_arr / n_0) * 100.0

    return {
        "t": sol.t,
        "s": s_arr,
        "i": i_arr,
        "r": r_arr,
        "d": d_arr,
        "v_cum": v_arr,
        "prop_s": prop_s,
        "prop_i": prop_i,
        "prop_r": prop_r,
        "prop_d": prop_d,
        "n_0": n_0,
        "total_vacinados": total_vacinados,
        "total_mortes": total_mortes,
        "pop_viva_final": pop_viva_final,
        "pico_infectados": pico_infectados,
        "tempo_pico": tempo_pico,
        "t1_vac": t1_vac,
        "t2_vac": t2_vac,
        "nu": float(nu),
        "alpha": float(alpha)
    }
