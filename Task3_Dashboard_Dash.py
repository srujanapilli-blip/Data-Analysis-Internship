
# Task3_Dashboard_Dash.py
# Dash application to visualize trip data (works with parquet or CSV).
# Install: pip install dash pandas pyarrow plotly
import os
import pandas as pd
from datetime import datetime
from dash import Dash, dcc, html, Input, Output, dash_table
import plotly.express as px

DATA_PARQUET = './data/taxi_synth/trips_parquet'  # produced by Task1 notebook if you ran it
DATA_CSV_FALLBACK = './data/taxi_synth/_tmp_csv'  # fallback CSV shards

def load_data():
    # try parquet first
    if os.path.exists(DATA_PARQUET):
        try:
            df = pd.read_parquet(DATA_PARQUET)
            print('Loaded parquet from', DATA_PARQUET)
            return df
        except Exception as e:
            print('Failed to read parquet:', e)
    # fallback: read CSV shards
    if os.path.exists(DATA_CSV_FALLBACK):
        try:
            df = pd.concat([pd.read_csv(os.path.join(DATA_CSV_FALLBACK, f)) for f in os.listdir(DATA_CSV_FALLBACK) if f.endswith('.csv')], ignore_index=True)
            print('Loaded CSV shards from', DATA_CSV_FALLBACK)
            return df
        except Exception as e:
            print('Failed to read CSV shards:', e)
    # if neither exists, create a small synthetic sample
    print('No data found; creating a small synthetic sample for demo.')
    import numpy as np
    rng = np.random.default_rng(42)
    n = 2000
    base = datetime(2023,1,1)
    pickup_offsets = rng.integers(0, 90*24*3600, size=n)
    pickup_ts = [base + pd.Timedelta(seconds=int(s)) for s in pickup_offsets]
    df = pd.DataFrame({
        'trip_id': range(1, n+1),
        'pickup_ts': pickup_ts,
        'pickup_zone': rng.choice([f'Z{z:03d}' for z in range(1,51)], size=n),
        'dropoff_zone': rng.choice([f'Z{z:03d}' for z in range(1,51)], size=n),
        'fare_amount': rng.uniform(50, 500, size=n),
        'tip_amount': rng.uniform(0, 100, size=n),
    })
    df['revenue'] = df['fare_amount'] + df['tip_amount']
    df['date'] = pd.to_datetime(df['pickup_ts']).dt.date
    return df

df = load_data()
df['pickup_ts'] = pd.to_datetime(df['pickup_ts'])
df['date'] = df['pickup_ts'].dt.date
df['hour'] = df['pickup_ts'].dt.hour

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

zones = sorted(df['pickup_zone'].unique())[:200]

app.layout = html.Div([
    html.H2('Task 3 - Interactive Dashboard (Trips Revenue)'),
    html.Div([
        html.Div([
            html.Label('Date range'),
            dcc.DatePickerRange(
                id='date-range',
                start_date=df['date'].min(),
                end_date=df['date'].max(),
                display_format='YYYY-MM-DD'
            )
        ], style={'display':'inline-block', 'margin-right':'20px'}),
        html.Div([
            html.Label('Pickup zone (multi)'),
            dcc.Dropdown(id='zone-filter', options=[{'label':z,'value':z} for z in zones], multi=True, value=zones[:5], placeholder='Select zones')
        ], style={'display':'inline-block', 'width':'40%'}),
        html.Div([
            html.Label('Metric'),
            dcc.RadioItems(id='metric', options=[{'label':'Total revenue','value':'revenue'},{'label':'Trip count','value':'count'}], value='revenue', labelStyle={'display':'inline-block','margin-right':'10px'})
        ], style={'display':'inline-block','margin-left':'20px'})
    ], style={'margin-bottom':'20px'}),
    html.Div([
        dcc.Graph(id='timeseries', style={'width':'65%','display':'inline-block'}),
        html.Div([
            dcc.Graph(id='hourly', style={'height':'300px'}),
            html.H4('Top corridors'),
            dash_table.DataTable(id='top-corridors', page_size=8, style_table={'overflowX':'auto'})
        ], style={'display':'inline-block','width':'33%','verticalAlign':'top','paddingLeft':'10px'})
    ]),
    html.Div([
        html.H5('Notes'),
        html.Ul([html.Li('This dashboard reads partitioned parquet if available (Task1 output).'),
                html.Li('Use the date range and zone filter to slice data and discover patterns.'),
                html.Li('To run: `python Task3_Dashboard_Dash.py` and open the printed local URL.')])
    ], style={'marginTop':'20px'})
])

@app.callback(
    Output('timeseries','figure'),
    Output('top-corridors','data'),
    Output('top-corridors','columns'),
    Output('hourly','figure'),
    Input('date-range','start_date'),
    Input('date-range','end_date'),
    Input('zone-filter','value'),
    Input('metric','value')
)
def update(start_date, end_date, zones_selected, metric):
    dff = df.copy()
    if start_date:
        dff = dff[dff['date'] >= pd.to_datetime(start_date).date()]
    if end_date:
        dff = dff[dff['date'] <= pd.to_datetime(end_date).date()]
    if zones_selected and len(zones_selected)>0:
        dff = dff[dff['pickup_zone'].isin(zones_selected)]
    if metric == 'revenue':
        ts = dff.groupby('date').agg(revenue=('fare_amount','sum'))  # revenue column exists in some data; fallback to fare only
        ts['revenue'] = dff['fare_amount'] + dff['tip_amount'] if 'tip_amount' in dff.columns else dff['fare_amount']
        ts = dff.groupby('date').apply(lambda x: (x['fare_amount'] + x.get('tip_amount',0)).sum()).rename('revenue').reset_index()
        fig_ts = px.line(ts, x='date', y='revenue', title='Daily Revenue')
    else:
        ts = dff.groupby('date').agg(trips=('trip_id','count')).reset_index()
        fig_ts = px.line(ts, x='date', y='trips', title='Daily Trip Count')
    # top corridors
    if 'dropoff_zone' in dff.columns:
        corridors = (dff.assign(revenue=(dff.get('fare_amount',0) + dff.get('tip_amount',0)))
                     .groupby(['pickup_zone','dropoff_zone'])
                     .agg(trips=('trip_id','count'), revenue=('revenue','sum'))
                     .reset_index()
                     .sort_values('revenue', ascending=False).head(10))
        table_data = corridors.to_dict('records')
        table_cols = [{'name':c,'id':c} for c in corridors.columns]
    else:
        corridors = pd.DataFrame(columns=['pickup_zone','dropoff_zone','trips','revenue'])
        table_data = corridors.to_dict('records')
        table_cols = []
    # hourly
    hourly = dff.groupby('hour').apply(lambda x: (x.get('fare_amount',0) + x.get('tip_amount',0)).sum() if metric=='revenue' else x['trip_id'].count()).rename('value').reset_index()
    fig_hour = px.bar(hourly, x='hour', y='value', title='Hourly '+('Revenue' if metric=='revenue' else 'Trips'))
    return fig_ts, table_data, table_cols, fig_hour

if __name__ == '__main__':
    port = 8050
    print('Starting Dash app on http://127.0.0.1:%d' % port)
    app.run_server(debug=True, port=port)
