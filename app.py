"""
=============================================================================
Aplicação Dash: Modelo SIRS com Vacinação e Mortalidade por Doença
=============================================================================
"""

import os
import sys
import flask
import dash
import dash_bootstrap_components as dbc

# Garante que o diretório raiz do projeto esteja no sys.path para imports locais e WSGI
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.layout import create_layout
from src.callbacks import register_callbacks

# =============================================================================
# CONFIGURAÇÃO DE PREFIXO DE ROTA (Compatibilidade Local e Deploy / Reverse Proxy)
# =============================================================================
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
app.title = "Modelo SIRS com Vacinação e Mortalidade por Doença"
server = app.server

# Redirecionamento amigável da raiz '/' para o prefixo em ambiente de desenvolvimento local
if prefix != '/':
    @server.route('/')
    def root_redirect():
        return flask.redirect(prefix)

# Definição do layout da interface a partir do módulo src.layout
app.layout = create_layout(app)

# Registro de todos os callbacks reativos
register_callbacks(app)

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8050))
    host = os.getenv('HOST', '0.0.0.0')
    app.run(debug=False, port=port, host=host)
