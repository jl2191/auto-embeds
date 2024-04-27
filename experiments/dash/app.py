# %%
from uuid import uuid4

import dash_bootstrap_components as dbc
import diskcache
import pandas as pd
from dash import Dash, DiskcacheManager

from experiments.dash.callbacks import register_callbacks
from experiments.dash.layout import layout

launch_uid = uuid4()

cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache, cache_by=[lambda: launch_uid])

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.CERULEAN],
    background_callback_manager=background_callback_manager,
)
app.layout = layout
register_callbacks(app)

if __name__ == "__main__":
    app.run_server(debug=True)
