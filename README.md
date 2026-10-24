# 🦠 dash-sir: Simulador do Modelo SIRS com Vacinação e Mortalidade

Aplicação web interativa desenvolvida com **Dash**, **Plotly**, **Bootstrap Components** e **SciPy** para simulação e estudo de dinâmicas epidemiológicas através de um modelo compartimental determinístico do tipo **SIRS** (*Suscetível – Infectado – Recuperado – Suscetível*), com estratégia de vacinação por pulsos temporários e taxa de mortalidade induzida pela doença.

---

## 📌 Sumário
- [Visão Geral](#-visão-geral)
- [Formulação Matemática](#-formulação-matemática)
- [Parâmetros do Modelo](#-parâmetros-do-modelo)
- [Principais Funcionalidades](#-principais-funcionalidades)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Instalação e Execução](#-instalação-e-execução)
- [Testes Automatizados](#-testes-automatizados)
- [Roteiro de Estudo e Perguntas](#-roteiro-de-estudo-e-perguntas)
- [Licença](#-licença)

---

## 🔬 Visão Geral

O simulador modela a propagação de um agente infeccioso em uma população onde a imunidade adquirida (tanto por infecção natural quanto por vacinação) decai ao longo do tempo. 

Diferenciais implementados no modelo:
1. **Campanha de Vacinação Temporária:** Aplicação de vacina com taxa instantânea $\nu_0$ restrita a uma janela de tempo $[t_1, t_2]$ (anos).
2. **Mortalidade Induzida pela Doença ($\alpha$):** Além da mortalidade natural demográfica ($\mu$), indivíduos infectados podem sucumbir à doença a uma taxa $\alpha$, permitindo avaliar a virulência do patógeno e a redução na população sobrevivente.
3. **Indicadores em Tempo Real (KPIs):** Cards de acompanhamento do total acumulado de imunizações administradas, total acumulado de mortes, pico epidêmico e sobreviventes finais.
4. **Visualização Flexível:** Alternância dinâmica entre proporção da população inicial (%) e contagem absoluta de indivíduos.
5. **Cenários Didáticos Rápidos:** Botões de predefinição rápida para carregar instantaneamente as questões pedagógicas de estudo.
6. **Rodapé Institucional:** Rodapé com link para o código-fonte aberto no GitHub.

---

## 📐 Formulação Matemática

A população viva total em qualquer instante $t$ é dada por:

$$N_{vivo}(t) = S(t) + I(t) + R(t)$$

O sistema de equações diferenciais ordinárias (EDOs) que governa a transição entre os compartimentos é:

$$\begin{aligned}
\frac{dS}{dt} &= \mu(I + R) - \frac{\beta S I}{N_{vivo}} - \nu(t) S + \left(\frac{12}{\delta}\right) R \\
\frac{dI}{dt} &= \frac{\beta S I}{N_{vivo}} - \left(\frac{365}{\gamma}\right) I - \mu I - \alpha I \\
\frac{dR}{dt} &= \left(\frac{365}{\gamma}\right) I - \mu R + \nu(t) S - \left(\frac{12}{\delta}\right) R \\
\frac{dD}{dt} &= \alpha I \quad \text{(óbitos acumulados por doença)} \\
\frac{dV_{cum}}{dt} &= \nu(t) S \quad \text{(imunizações acumuladas no período)}
\end{aligned}$$

Em que a função de vacinação $\nu(t)$ é modelada via função degrau de Heaviside $H(\cdot)$:

$$\nu(t) = \nu_0 \cdot H(t - t_1) \cdot H(t_2 - t)$$

### Conservação Populacional:
A taxa de nascimentos naturais $\mu(I + R) = \mu N_{vivo} - \mu S$ repõe a mortalidade natural demográfica da população viva. Na presença de mortalidade pela doença ($\alpha > 0$), a coorte total incluindo os óbitos permanece rigorosamente conservada:

$$\frac{d}{dt}\big(S(t) + I(t) + R(t) + D(t)\big) = 0 \implies S(t) + I(t) + R(t) + D(t) = N_0$$

---

## ⚙️ Parâmetros do Modelo

| Símbolo | Nome do Parâmetro | Unidade | Descrição | Padrão |
| :---: | :--- | :---: | :--- | :---: |
| **$\beta$** | Coeficiente de Transmissão | $\text{ano}^{-1}$ | Taxa de contatos potencialmente infectantes | $80$ |
| **$\gamma$** | Período Infeccioso | $\text{dias}$ | Duração média da fase transmissível ($\text{taxa} = 365/\gamma$) | $15$ |
| **$\delta$** | Período de Imunidade | $\text{meses}$ | Duração média da proteção pós-cura ou pós-vacina ($\text{taxa} = 12/\delta$) | $6$ |
| **$\mu$** | Mortalidade Natural / Natalidade | $\text{ano}^{-1}$ | Taxa basal de renovação demográfica | $0.10$ |
| **$\alpha$** | Taxa de Mortalidade por Doença | $\text{ano}^{-1}$ | Letalidade/virulência induzida pela infecção | $0.20$ |
| **$\nu_0$** | Taxa Instantânea de Vacinação | $\text{ano}^{-1}$ | Intensidade de aplicação vacinal na população suscetível | $2.0$ |
| **$[t_1, t_2]$**| Janela da Campanha Vacinal | $\text{anos}$ | Intervalo de início e término da intervenção | $[2, 5]$ |
| **$S(0)$** | Suscetíveis Iniciais | indivíduos | População suscetível no instante $t=0$ | $99$ |
| **$I(0)$** | Infectados Iniciais | indivíduos | População infectada no instante $t=0$ | $1$ |
| **$R(0)$** | Recuperados Iniciais | indivíduos | População recuperada/imune no instante $t=0$ | $0$ |

---

## 🌟 Principais Funcionalidades

- **Controle Interativo por Sliders:** Ajuste instantâneo de todos os parâmetros epidemiológicos, demográficos e vacinais com exibição dinâmica dos valores selecionados em badges.
- **Cenários Didáticos Rápidos:** Botões para carregar com um clique os cenários de referência e as questões pedagógicas.
- **Curva de Mortalidade por Doença ($D$):** Acompanhamento contínuo da evolução temporal dos óbitos induzidos pela infecção.
- **Cards de Métricas (KPIs):**
  - 💉 **Total de Vacinados:** Integral acumulada $\int \nu(t) S(t) dt$ refletindo o esforço vacinal total.
  - ⚰️ **Mortes pela Doença:** Total acumulado de óbitos e porcentagem em relação à população inicial.
  - 🦠 **Pico de Infectados:** Prevalência máxima atingida e ano de ocorrência.
  - 👥 **População Viva Final:** Quantidade de indivíduos vivos ao término dos 10 anos de simulação e distribuição entre $S, I, R$.
- **Faixa da Campanha Vacinal no Gráfico:** Região sombreada no gráfico destacando visualmente o período $[t_1, t_2]$.
- **Integração Numérica Robusta:** Utilização do método implícito `Radau` (`scipy.integrate.solve_ivp`), adequado para sistemas rígidos (*stiff*) que surgem com mudanças bruscas em campanhas de vacinação.
- **Compatibilidade de Deploy e Desenvolvimento:** Suporte a rota de produção `/dash_sir/` com redirecionamento automático da rota raiz `/` em desenvolvimento local.
- **Rodapé com Link para o GitHub:** Identificação visual e link direto para o repositório oficial no GitHub.

---

## 📁 Estrutura do Projeto

O projeto adota a arquitetura modular estabelecida em [dash_sli](https://github.com/fferreir/dash_sli):

```text
dash-sir/
├── app.py                 # Ponto de entrada leve da aplicação Dash e rotas
├── assets/
│   ├── estilos.css        # Folha de estilos (Design system: Hero, KPIs, Sliders)
│   └── SIR.png            # Diagrama explicativo do modelo compartimental
├── pytest.ini             # Configuração do executor de testes (pythonpath = .)
├── requirements.txt       # Dependências do projeto (Dash, SciPy, Plotly, pytest)
├── LICENSE                # Licença GNU GPLv3
├── README.md              # Documentação completa do projeto
├── src/
│   ├── __init__.py        # Inicialização do pacote src
│   ├── model.py           # EDOs, métricas analíticas (R0) e rotina de simulação numérica
│   ├── layout.py          # Interface visual, hero header, controles, acordeom e rodapé com link GitHub
│   └── callbacks.py       # Callbacks reativos (cenários, badges, simulação e gráficos)
└── tests/
    ├── __init__.py        # Inicialização do módulo de testes
    ├── test_app.py        # Testes de integração de interface, validação do rodapé e layout
    └── test_model.py      # Testes unitários das EDOs, lei de conservação e métricas
```

---

## 🚀 Instalação e Execução

### Pré-requisitos
- Python 3.10 ou superior.

### 1. Clonar o Repositório
```bash
git clone https://github.com/fferreir/dash-sir.git
cd dash-sir
```

### 2. Criar e Ativar Ambiente Virtual

Com `venv` padrão:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Ou de forma ultra-rápida com [`uv`](https://docs.astral.sh/uv/):
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 3. Executar a Aplicação
```bash
python app.py
```

Abra o navegador em:
- [http://127.0.0.1:8050/](http://127.0.0.1:8050/) (redireciona automaticamente para `/dash_sir/`) ou diretamente em [http://127.0.0.1:8050/dash_sir/](http://127.0.0.1:8050/dash_sir/).

> **Nota de Deploy:** Para personalizar o subcaminho em servidores proxy reverso (Nginx/Traefik), defina a variável de ambiente:
> ```bash
> export DASH_REQUESTS_PATHNAME_PREFIX="/dash_sir/"
> ```

---

## 🧪 Testes Automatizados

A bateria de testes automatizados com `pytest` valida tanto o modelo matemático e suas leis de conservação quanto os componentes da interface gráfica:

```bash
pytest -v
```

---

## 📚 Roteiro de Estudo e Perguntas

1. **Estado de Equilíbrio sem Intervenção:**
   - Configure $\nu_0 = 0$ (sem vacina) e $\alpha = 0$ (sem mortalidade por doença).
   - *Em quanto tempo o sistema atinge o estado endêmico? Qual a proporção de suscetíveis e infectados no equilíbrio?*

2. **Duração do Período Infeccioso:**
   - Adote $\beta = 20 \space \text{ano}^{-1}$ e compare períodos infecciosos de $\gamma = 15 \space \text{dias}$ contra $\gamma = 30 \space \text{dias}$.
   - *O que acontece com a intensidade do pico de infecção e a prevalência endêmica?*

3. **Efeito de Rebote Vacinal:**
   - Com os parâmetros padrão, ative $\nu_0 = 2.0 \space \text{ano}^{-1}$ entre os anos $2$ e $5$.
   - *Observe o controle da doença durante a campanha e o que ocorre logo após o encerramento da vacinação no ano 5 devido à perda de imunidade ($\delta = 6 \space \text{meses}$).*

4. **Impacto da Mortalidade por Doença ($\alpha$):**
   - Varie o slider de $\alpha$ de $0.0$ até $1.0 \space \text{ano}^{-1}$.
   - *Como a mortalidade induzida afeta a curva de óbitos acumulados, o número de sobreviventes e a circulação sustentada do patógeno?*

5. **Otimização de Políticas de Saúde:**
   - Para um patógeno letal ($\alpha = 0.5 \space \text{ano}^{-1}$), qual intensidade de vacinação $\nu_0$ é necessária para mitigar drasticamente o total de óbitos no período?

---

## 📄 Licença

Este projeto é distribuído sob a licença **GNU General Public License v3.0**. Consulte o arquivo [LICENSE](LICENSE) para mais detalhes.
