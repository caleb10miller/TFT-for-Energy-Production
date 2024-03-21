# Imports
import calendar
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# Load EDA dataset
eda_df = pd.read_csv('./data/power_consumption_by_fuel_type.csv')
eda_df['period'] = pd.to_datetime(eda_df['period'])
eda_df['month'] = eda_df['period'].dt.month
eda_df['year'] = eda_df['period'].dt.year
eda_df['fueltype'] = eda_df['fueltype'].replace({'WND': 'WIND'})  # Convert 'WND' to 'WIND'
eda_df['month_name'] = eda_df['month'].apply(lambda x: calendar.month_abbr[x])

# Assuming preprocessing for excluding 2018 and computing monthly average
monthly_avg = eda_df[eda_df['year'] != 2018].groupby('month')['value'].mean().reset_index()
monthly_avg['month_name'] = monthly_avg['month'].apply(lambda x: calendar.month_abbr[x])

# Load main dataset
merged_df = pd.read_csv('./data/vizData.csv')
merged_df['residual'] = merged_df['value'] - merged_df['0.5']

# Create Dash app instance
app = dash.Dash(__name__)

# Define app layout
app.layout = html.Div(style={'backgroundColor': 'lightgrey'}, children=[
    html.Div(style={'backgroundColor': '#00437B', 'marginLeft':'2%', 'marginRight':'2%', 'padding':'20px'}, children=[
        html.H1("Power Production Visual Analysis", style={'textAlign':'center', 'color':'lightgrey'}),
        html.H4("Project completed by Caleb Miller, Luke Chesley, Hashim Afzal, and Lauren Miller at Drexel University", style={'textAlign':'center', 'color':'lightgrey'}),
        html.H4("Dashboard completed by Caleb Miller", style={'textAlign':'center', 'color':'lightgrey'}),
        html.Div(style={'height': '20px'}),
        html.H3("Discussion of Project:", style={'textAlign': 'center', 'text-decoration':'underline', 'color':'lightgrey'}),
        html.P("In this project we will be utilizing data from the U.S. Energy Information Administration. The U.S. Energy Information Administration has free and open data available through an Application Programming Interface (API) and its open data tools. EIA's API is multi-facetted and contains the following time-series data sets organized by the main energy categories. We are analyzing hourly energy consumption data from 2018 to present. This data is also categorized by region/state. In this analysis, we aim to examine patterns within the data, including variations across locations/regions, trends over different time frames/date categories, and shifts in types of energy consumption.", style={'textAlign': 'center', 'color':'lightgrey'}),
        html.Div([
            html.Div([
                html.H3("Application of Investigation:", style={'textAlign': 'center', 'text-decoration':'underline', 'color':'lightgrey'}),
            ], style={'width': '49%', 'display': 'inline-block', 'verticalAlign': 'top'}),

            html.Div([
                html.H3("Training and Output:", style={'textAlign': 'center', 'text-decoration':'underline', 'color':'lightgrey'}),
            ], style={'width': '49%', 'display': 'inline-block', 'verticalAlign': 'top'}),

            html.Div([
                html.P("In this project we will be utilizing data from the U.S. Energy Information Administration. The U.S. Energy Information Administration has free and open data available through an Application Programming Interface (API) and its open data tools. EIA's API is multi-facetted and contains the following time-series data sets organized by the main energy categories. We are analyzing hourly energy consumption data from 2018 to present. This data is also categorized by region/state. In this analysis, we aim to examine patterns within the data, including variations across locations/regions, trends over different time frames/date categories, and shifts in types of energy consumption.", style={'textAlign': 'left', 'color':'lightgrey'}),
            ], style={'width': '49%', 'display': 'inline-block', 'verticalAlign': 'top'}),

            html.Div([
                html.P("Instead of just a single value, the TFT predicts quantiles of the distribution of target ŷ using a special quantile loss function. Prediction intervals via quantile forecasts to determine the range of likely target values at each prediction horizon. The training data spans 2018-07-01 to 2023-12-31. The prediction data spans 2024-01-01 to 2024-01-08.", style={'textAlign': 'left', 'color':'lightgrey'}),
            ], style={'width': '49%', 'display': 'inline-block', 'verticalAlign': 'top', 'float': 'right'})
        ]),
        html.Div(style={'height': '20px'}),
        html.H2("Dynamic Analysis Based on Fuel Type", style={'textAlign':'center', 'text-decoration':'underline', 'color':'lightgrey'}),
        html.H3("Select Fuel Type Below: ", style={'textAlign':'left', 'color':'lightgrey'}),
        dcc.Dropdown(
            id='fuel-type-dropdown',
            options=[{'label': i, 'value': i} for i in merged_df['Fuel_Type'].unique()],
            value=merged_df['Fuel_Type'].unique()[0]
        ),
        html.Div([
            html.H3("Exploratory Data Analysis:", style={'textAlign':'left', 'color':'lightgrey'}),
            dcc.Graph(id='eda-graph', style={'display': 'inline-block', 'width':'49%'}),  # EDA graph placeholder
            dcc.Graph(id='fuel-trends-graph', style={'display': 'inline-block', 'width':'49%'}),  # Fuel consumption trends over time graph
            html.H3("Temporal Fusion Transformer Prediction Analysis:", style={'textAlign':'left', 'color':'lightgrey'}),
            dcc.Graph(id='residuals-graph', style={'display': 'inline-block', 'width': '49%'}),
            dcc.Graph(id='quantiles-graph', style={'display': 'inline-block', 'width': '49%'}), 
            html.P("This visualizations show a residual analysis of our model predictions. The blue shading highlights where our model under-predicted and the red shading shows where our model over-predicted. We see that certain fuel types are notorious for under-prediction, and vice versa. For example, coal, natrual gas, and nuclear energy are almost entirely under-predicted. On the other hand, wind energy is almost entirely over-predicted", style={'display':'inline-block', 'textAlign':'left', 'width':'48%', 'color':'lightgrey'}),
            html.P("This graph shows how the true value compared to our 25th and 75th quantiles. Ideally, all of the true values would lie within the grey band. We see that certain fuel types are much more predictable than others. Example: With coal, most of the true values lie within the grey band whereas with nuclear energy, there is not a single value that lies in this interval", style={'display':'inline-block','textAlign':'left', 'width':'48%', 'verticalAlign':'top', 'color':'lightgrey'})
        ])
    ])
])

# Define callback to update graphs based on dropdown selection
@app.callback(
    [Output('eda-graph', 'figure'),
     Output('fuel-trends-graph', 'figure'),
     Output('residuals-graph', 'figure'),
     Output('quantiles-graph', 'figure')],
    [Input('fuel-type-dropdown', 'value')]
)
def update_graph(selected_fuel_type):
    # EDA Graph - Filtered by selected fuel type and year != 2018
    filtered_eda_df = eda_df[(eda_df['fueltype'] == selected_fuel_type) & (eda_df['year'] != 2018)]
    monthly_avg_filtered = filtered_eda_df.groupby('month_name')['value'].mean().reset_index()
    
    # Compute annual average monthly consumption for trend line
    annual_avg = filtered_eda_df.groupby('month_name')['value'].mean().mean()
    trendline = pd.DataFrame({'month_name': monthly_avg_filtered['month_name'], 'value': [annual_avg] * len(monthly_avg_filtered)})
    
    eda_fig = px.bar(monthly_avg_filtered, x='month_name', y='value', title=f'Monthly Average Consumption for {selected_fuel_type}', color_discrete_sequence=['#07294D'])
    eda_fig.add_trace(go.Scatter(x=trendline['month_name'], y=trendline['value'], mode='lines', name='Annual Average', line=dict(color='#FFD700')))
    eda_fig.update_layout(title=f'Monthly Average Consumption for {selected_fuel_type}', title_x=0.5, paper_bgcolor='lightgrey')
    
    # Fuel consumption trends over time graph
    fuel_trends_fig = px.line(filtered_eda_df, x='period', y='value', title=f'Fuel Consumption Trends Over Time for {selected_fuel_type}', color_discrete_sequence=['#07294D'])
    fuel_trends_fig.update_layout(title=f'Fuel Consumption Trends Over Time for {selected_fuel_type}', title_x=0.5, paper_bgcolor='lightgrey')
  
    # Prepare the dataset for the residuals and quantiles graphs
    df_ = merged_df[merged_df['Fuel_Type'] == selected_fuel_type]
    df_['Hour'] = pd.to_numeric(df_['Hour'], errors='coerce')
    
    # Residuals Graph
    residuals_fig = go.Figure()
    residuals_fig.add_trace(go.Scatter(x=df_['Hour'], y=df_['residual'], mode='lines', name='Actual - Predicted', line=dict(color='#07294D')))
    # Highlight over-predicted area (positive residual)
    residuals_fig.add_trace(go.Scatter(x=df_['Hour'], y=np.maximum(df_['residual'], 0), mode='none', fill='tozeroy', fillcolor='rgba(0, 67, 123, 0.5)', name='Under-predicted'))
    # Highlight under-predicted area (negative residual)
    residuals_fig.add_trace(go.Scatter(x=df_['Hour'], y=np.minimum(df_['residual'], 0), mode='none', fill='tozeroy', fillcolor='rgba(255, 218, 2, 0.5)', name='Over-predicted'))
    residuals_fig.update_layout(title=f'Residuals for {selected_fuel_type}', xaxis_title='Hour', yaxis_title='Residual', title_x=0.5, paper_bgcolor='lightgrey')


    # Quantiles Graph
    quantiles_fig = go.Figure()
    quantiles_fig.add_trace(go.Scatter(x=df_['Hour'], y=df_['0.25'], mode='lines', name='0.25 Quantile', line=dict(color='blue', dash='dash')))
    quantiles_fig.add_trace(go.Scatter(x=df_['Hour'], y=df_['value'], mode='lines', name='True Value', line=dict(color='#07294D')))
    quantiles_fig.add_trace(go.Scatter(x=df_['Hour'], y=df_['0.75'], mode='lines', name='0.75 Quantile', line=dict(color='red', dash='dash')))
    quantiles_fig.add_trace(go.Scatter(x=df_['Hour'], y=df_['0.25'], mode='none', fill='tonexty', fillcolor='rgba(255, 218, 2, 0.5)', name='Fill', showlegend=False))
    quantiles_fig.update_layout(title=f'True Value vs Quantiles for {selected_fuel_type}', xaxis_title='Hour', yaxis_title='Output', title_x=0.5, paper_bgcolor='lightgrey')

    return eda_fig, fuel_trends_fig, residuals_fig, quantiles_fig

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')
