import dash
from dash import dcc, html
import joblib
import pandas as pd
import shap
import plotly.graph_objs as go
import numpy as np

model = joblib.load('model.joblib')
shap_values = joblib.load('shap_values.pkl')

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("False News Risk Dashboard"),
    dcc.Graph(id='shap-summary')
])

@app.callback(
    dash.dependencies.Output('shap-summary', 'figure'),
    []
)
def update_plot():
    summary = np.abs(shap_values).mean(axis=0)
    fig = go.Figure([go.Bar(x=list(range(len(summary))), y=summary)])
    fig.update_layout(title="SHAP Feature Importance Summary")
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
