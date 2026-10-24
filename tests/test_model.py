"""
Testes unitários para o módulo matemático e epidemiológico (src/model.py).
"""

import pytest
import numpy as np
from src.model import ode_sys, calcular_metricas_epidemiologicas, simular_sir


def test_ode_sys_extinction_zeros():
    """Se todas as populações forem zero, as taxas devem ser zero."""
    state = [0.0, 0.0, 0.0, 0.0, 0.0]
    derivs = ode_sys(
        t=0.0,
        state=state,
        alpha_disease=0.2,
        beta=80.0,
        gamma=15.0,
        delta=6.0,
        mu=0.10,
        nu=2.0,
        t_begv=2.0,
        t_endv=5.0
    )
    assert derivs == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_ode_sys_conservation_law():
    """
    Verifica se a soma das taxas de variação d/dt(S + I + R + D) é identicamente nula,
    garantindo a conservação estrita da coorte populacional total.
    """
    state = [70.0, 20.0, 10.0, 5.0, 12.0]
    derivs = ode_sys(
        t=3.0,  # dentro da campanha de vacinação
        state=state,
        alpha_disease=0.25,
        beta=75.0,
        gamma=12.0,
        delta=8.0,
        mu=0.12,
        nu=3.0,
        t_begv=1.0,
        t_endv=4.0
    )
    ds_dt, di_dt, dr_dt, dd_dt, _ = derivs
    assert pytest.approx(ds_dt + di_dt + dr_dt + dd_dt, abs=1e-8) == 0.0


def test_metricas_epidemiologicas_calculo():
    """Valida o cálculo analítico de R0, taxas e cobertura crítica de vacinação."""
    beta = 80.0
    gamma = 15.0
    delta = 6.0
    mu = 0.10
    alpha = 0.20
    m = calcular_metricas_epidemiologicas(beta=beta, gamma=gamma, delta=delta, mu=mu, alpha=alpha)

    # Taxa de recuperação: 365 / 15 = 24.3333...
    assert pytest.approx(m["taxa_recuperacao"], 1e-4) == 24.3333

    # Taxa de perda de imunidade: 12 / 6 = 2.0 ano^-1
    assert pytest.approx(m["taxa_perda_imunidade"], 1e-4) == 2.0

    # R0 = 80 / (24.3333 + 0.10 + 0.20) = 80 / 24.6333... ≈ 3.2476
    assert pytest.approx(m["r0"], 1e-3) == 3.2476

    # Cobertura crítica p_c = (1 - 1/R0) * 100 ≈ 69.21%
    assert pytest.approx(m["p_critico_vac"], 1e-2) == 69.21
    assert m["status_epidemia"] == "Potencial Epidêmico / Endêmico"


def test_metricas_epidemiologicas_subcritico():
    """Quando beta é muito baixo tal que R0 < 1, o status deve indicar extinção e p_c=0."""
    m = calcular_metricas_epidemiologicas(beta=10.0, gamma=15.0, delta=6.0, mu=0.10, alpha=0.20)
    assert m["r0"] < 1.0
    assert m["p_critico_vac"] == 0.0
    assert m["status_epidemia"] == "Extinção da Infecção (Subcrítico)"


def test_simular_sir_baseline():
    """Execução da simulação de referência: preserva não-negatividade e coerência das séries."""
    res = simular_sir(
        s_init=99.0, i_init=1.0, r_init=0.0,
        beta=80.0, gamma=15.0, delta=6.0, mu=0.10, alpha=0.20,
        nu=2.0, vacinacao=[2.0, 5.0], t_end=10.0, n_samples=1000
    )

    assert len(res["t"]) == 1000
    assert (res["s"] >= 0.0).all()
    assert (res["i"] >= 0.0).all()
    assert (res["r"] >= 0.0).all()
    assert (res["d"] >= 0.0).all()
    assert (res["v_cum"] >= 0.0).all()

    # O pico de infectados deve ser positivo e ocorrer antes de 1 ano
    assert res["pico_infectados"] > 1.0
    assert 0.0 < res["tempo_pico"] < 2.0

    # Total de vacinados e mortes devem ser positivos
    assert res["total_vacinados"] > 0.0
    assert res["total_mortes"] > 0.0
    assert res["pop_viva_final"] > 0.0


def test_simular_sir_sem_vacinacao():
    """Quando nu=0, o total acumulado de vacinações deve ser identicamente nulo."""
    res = simular_sir(
        s_init=99.0, i_init=1.0, r_init=0.0,
        beta=80.0, gamma=15.0, delta=6.0, mu=0.10, alpha=0.20,
        nu=0.0, vacinacao=[2.0, 5.0], t_end=10.0
    )
    assert pytest.approx(res["total_vacinados"], 1e-9) == 0.0
    assert (res["v_cum"] == 0.0).all()


def test_simular_sir_sem_mortalidade_doenca():
    """Quando alpha=0, mortes pela doença devem ser zero e população viva deve se manter N0."""
    res = simular_sir(
        s_init=99.0, i_init=1.0, r_init=0.0,
        beta=80.0, gamma=15.0, delta=6.0, mu=0.10, alpha=0.0,
        nu=0.0, vacinacao=[2.0, 5.0], t_end=10.0
    )
    assert pytest.approx(res["total_mortes"], 1e-9) == 0.0
    # População viva final deve permanecer estritamente em torno de N0 = 100
    assert pytest.approx(res["pop_viva_final"], abs=1e-4) == 100.0
