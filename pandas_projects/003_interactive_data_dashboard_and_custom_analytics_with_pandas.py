import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html, Input, Output
import numpy as np

# Load the dataset
file_path = "expanded_employee_dataset.xlsx"
df = pd.read_excel(file_path)

# Data preprocessing
df['Join_Year'] = df['Join_Date'].dt.year

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the dashboard
app.layout = html.Div([
    html.H1("Interactive Employee Data Dashboard", style={'textAlign': 'center'}),

    # Filters
    html.Div([
        html.Label("Select Department:"),
        dcc.Dropdown(
            id='department_filter',
            options=[{'label': dept, 'value': dept} for dept in df['Department'].unique()],
            multi=True,
            placeholder="Select departments"
        ),
        html.Label("Minimum Salary:"),
        dcc.Input(id='min_salary', type='number', value=0, placeholder="Enter minimum salary"),
        html.Label("Join Year Range:"),
        dcc.RangeSlider(
            id='join_year_filter',
            min=df['Join_Year'].min(),
            max=df['Join_Year'].max(),
            # marks={year: str(year) for year in df['Join_Year'].unique()},
            value=[df['Join_Year'].min(), df['Join_Year'].max()]
        )
    ], style={'padding': '20px'}),

    # KPI Cards
    html.Div([
        html.Div(id='kpi_total_employees', style={'fontSize': '20px'}),
        html.Div(id='kpi_avg_salary', style={'fontSize': '20px'}),
        html.Div(id='kpi_avg_performance', style={'fontSize': '20px'}),
    ], style={'display': 'flex', 'justifyContent': 'space-around', 'padding': '20px'}),

    # Charts
    html.Div([
        dcc.Graph(id='bar_chart'),
        dcc.Graph(id='heatmap')
    ])
])


# Callbacks for interactivity
@app.callback(
    [
        Output('kpi_total_employees', 'children'),
        Output('kpi_avg_salary', 'children'),
        Output('kpi_avg_performance', 'children'),
        Output('bar_chart', 'figure'),
        Output('heatmap', 'figure')
    ],
    [
        Input('department_filter', 'value'),
        Input('min_salary', 'value'),
        Input('join_year_filter', 'value')
    ]
)
def update_dashboard(department_filter, min_salary, join_year_range):
    # Filter data
    filtered_df = df[
        (df['Salary'] >= min_salary) &
        (df['Join_Year'] >= join_year_range[0]) &
        (df['Join_Year'] <= join_year_range[1])
        ]
    if department_filter:
        filtered_df = filtered_df[filtered_df['Department'].isin(department_filter)]

    # KPIs
    total_employees = f"Total Employees: {len(filtered_df)}"
    avg_salary = f"Average Salary: ${filtered_df['Salary'].mean():,.2f}"
    avg_performance = f"Average Performance Score: {filtered_df['Performance_Score'].mean():.2f}"

    # Bar chart
    bar_chart = px.bar(
        filtered_df,
        x='Department',
        y='Performance_Score',
        color='Department',
        title="Performance Scores by Department"
    )

    # Heatmap
    heatmap_data = filtered_df.pivot_table(
        index='Department',
        columns='Join_Year',
        values='Salary',
        aggfunc='mean'
    ).fillna(0)
    heatmap = px.imshow(
        heatmap_data,
        labels={'x': 'Year', 'y': 'Department', 'color': 'Average Salary'},
        title="Heatmap of Average Salaries by Department and Year"
    )

    return total_employees, avg_salary, avg_performance, bar_chart, heatmap


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
