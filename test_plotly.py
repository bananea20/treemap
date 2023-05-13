from jupyter_dash import JupyterDash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import json

app = JupyterDash(__name__)

df = pd.DataFrame({
    'ids': ['A', 'B', 'B1', 'B2', 'C', 'C1'],
    'labels': ['A', 'B', 'B1', 'B2', 'C', 'C1'],
    'parents': ['', 'A', 'B', 'B', 'A', 'C'],
    'extra_info': ['info A', 'info B', 'info B1', 'info B2', 'info C', 'info C1'],
    'click_info': ['Information A', 'Information B', 'Information B1', 'Information B2', 'Information C', 'Information C1']
})

fig = go.Figure(go.Treemap(
    ids = df['ids'],
    labels = df['labels'],
    parents = df['parents'],
    text = df['extra_info'].tolist()
))

app.layout = html.Div([
    dcc.Graph(
        id='my-graph',
        figure=fig
    ),
    html.Pre(id='click-data', style={'padding': '10px'}),
    html.Div(id='text-storage', style={'display': 'none'}),
    html.Div(id='node-clicked', style={'display': 'none'})
])

@app.callback(
    Output('my-graph', 'figure'),
    Output('text-storage', 'children'),
    Output('node-clicked', 'children'),
    Input('my-graph', 'clickData'),
    State('my-graph', 'figure'),
    State('text-storage', 'children'),
    State('node-clicked', 'children'))
def update_figure(clickData, fig, stored_text, clicked_node):
    if clickData:
        point_number = clickData['points'][0]['pointNumber']
        if stored_text is None:
            stored_text = json.dumps(fig['data'][0]['text'])
        if clicked_node is None or clicked_node != point_number:
            fig['data'][0]['text'] = json.loads(stored_text)
            fig['data'][0]['text'][point_number] = df.loc[point_number, 'click_info']
            clicked_node = point_number
        else:
            fig['data'][0]['text'] = json.loads(stored_text)
            clicked_node = None
    else:
        if stored_text is not None:
            fig['data'][0]['text'] = json.loads(stored_text)
        clicked_node = None

    return fig, stored_text, clicked_node

if __name__ == '__main__':
    app.run_server(debug=True)
