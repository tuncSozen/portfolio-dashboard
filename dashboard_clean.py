"""
Portfolio Dashboard - Clean Version
Main dashboard application with separated utilities.
"""

import dash
from dash import dcc, html, Input, Output, callback, dash_table
import plotly.graph_objects as go
import pandas as pd
import os
import sys

# Handle distutils compatibility for Python 3.12+
if sys.version_info >= (3, 12):
    try:
        import setuptools
        # Only map the modules that exist
        sys.modules['distutils'] = setuptools
        if hasattr(setuptools, 'util'):
            sys.modules['distutils.util'] = setuptools.util
        if hasattr(setuptools, 'version'):
            sys.modules['distutils.version'] = setuptools.version
        if hasattr(setuptools, 'errors'):
            sys.modules['distutils.errors'] = setuptools.errors
    except ImportError:
        print("Warning: setuptools not available, distutils compatibility disabled")

try:
    import dash_auth
except ImportError:
    print("Warning: dash_auth not available. Authentication will be disabled.")
    dash_auth = None
from data_utils import (load_data, process_data, create_pie_chart, create_twrr_chart, 
                       get_color_mapping, create_bank_comparison_chart, 
                       create_portfolio_evolution_chart, get_top_holdings, get_portfolio_holdings_data,
                       get_bank_portfolio_data, process_portfolio_data, get_evolution_data,
                       load_benchmarks, create_benchmark_ytd, create_benchmark_summary_chart,
                       create_bonds_stacked_chart, create_usd_index_chart, create_credit_risk_chart,
                       create_reinvestment_risk_chart, create_concentration_risk_chart)

# Load and process data
raw_data = load_data()
data = process_data(raw_data)

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[
    "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap"
], suppress_callback_exceptions=True)
app.title = "Portfolio Dashboard"

# Authentication setup
VALID_USERNAME_PASSWORD_PAIRS = {
    os.environ.get('DASH_USERNAME', 'admin'): os.environ.get('DASH_PASSWORD', 'admin123'),
    os.environ.get('DASH_USERNAME_2', 'user'): os.environ.get('DASH_PASSWORD_2', 'user123')
}

if dash_auth:
    auth = dash_auth.BasicAuth(
        app,
        VALID_USERNAME_PASSWORD_PAIRS
    )
else:
    print("Authentication disabled - dash_auth not available")
    auth = None

# Define the layout
app.layout = html.Div([
    html.Link(rel="stylesheet", href="/assets/dashboard_new.css"),
    html.Div([
        # Sidebar
        html.Div([
            html.Div([
                html.H1("📈 Portfolio Analytics", className="sidebar-header"),
                html.Div([
                    html.Button("📊 Summary", id="summary-btn", n_clicks=0, className="nav-link active"),
                    html.Button("🏦 Bank Analysis", id="banks-btn", n_clicks=0, className="nav-link"),
                    html.Button("📋 Portfolio Holdings", id="holdings-btn", n_clicks=0, className="nav-link"),
                    html.Button("⚠️ Risk Analysis", id="risk-btn", n_clicks=0, className="nav-link")
                ], className="nav-menu")
            ])
        ], className="sidebar"),
        
        # Main content
        html.Div([
            html.Div(id="main-content", className="main-content")
        ])
    ], className="dashboard-container")
])

# Callback for asset allocation charts
@app.callback(
    [Output('bbva-allocation-chart', 'figure'),
     Output('ubs-allocation-chart', 'figure'),
     Output('lo-allocation-chart', 'figure'),
     Output('bbva-btn-type', 'className'),
     Output('bbva-btn-subtype', 'className'),
     Output('bbva-btn-currency', 'className'),
     Output('bbva-btn-label', 'className'),
     Output('ubs-btn-type', 'className'),
     Output('ubs-btn-subtype', 'className'),
     Output('ubs-btn-currency', 'className'),
     Output('ubs-btn-label', 'className'),
     Output('lo-btn-type', 'className'),
     Output('lo-btn-subtype', 'className'),
     Output('lo-btn-currency', 'className'),
     Output('lo-btn-label', 'className')],
    [Input('bbva-btn-type', 'n_clicks'),
     Input('bbva-btn-subtype', 'n_clicks'),
     Input('bbva-btn-currency', 'n_clicks'),
     Input('bbva-btn-label', 'n_clicks'),
     Input('ubs-btn-type', 'n_clicks'),
     Input('ubs-btn-subtype', 'n_clicks'),
     Input('ubs-btn-currency', 'n_clicks'),
     Input('ubs-btn-label', 'n_clicks'),
     Input('lo-btn-type', 'n_clicks'),
     Input('lo-btn-subtype', 'n_clicks'),
     Input('lo-btn-currency', 'n_clicks'),
     Input('lo-btn-label', 'n_clicks')]
)
def update_allocation_charts(bbva_type, bbva_subtype, bbva_currency, bbva_label,
                           ubs_type, ubs_subtype, ubs_currency, ubs_label,
                           lo_type, lo_subtype, lo_currency, lo_label):
    if data is None:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#e5e5e5',
            height=300,
            annotations=[dict(text="No data available", x=0.5, y=0.5, showarrow=False, font=dict(size=16, color='#e5e5e5'))]
        )
        return (empty_fig, empty_fig, empty_fig, 
                "allocation-btn", "allocation-btn", "allocation-btn", "allocation-btn",
                "allocation-btn", "allocation-btn", "allocation-btn", "allocation-btn",
                "allocation-btn", "allocation-btn", "allocation-btn", "allocation-btn")
    
    bank_data = data['portfolio']
    
    # Determine active filter for each bank based on button clicks
    def get_active_filter(type_clicks, subtype_clicks, currency_clicks, label_clicks):
        if label_clicks > currency_clicks and label_clicks > subtype_clicks and label_clicks > type_clicks:
            return 'Label'
        elif currency_clicks > subtype_clicks and currency_clicks > type_clicks:
            return 'Currency'
        elif subtype_clicks > type_clicks:
            return 'Subtype'
        else:
            return 'Type'
    
    # Get active filters for each bank
    bbva_filter = get_active_filter(bbva_type, bbva_subtype, bbva_currency, bbva_label)
    ubs_filter = get_active_filter(ubs_type, ubs_subtype, ubs_currency, ubs_label)
    lo_filter = get_active_filter(lo_type, lo_subtype, lo_currency, lo_label)
    
    # Create charts
    bbva_fig = create_pie_chart(bank_data['bbva'], 'BBVA', bbva_filter)
    ubs_fig = create_pie_chart(bank_data['ubs'], 'UBS', ubs_filter)
    lo_fig = create_pie_chart(bank_data['lo'], 'LO', lo_filter)
    
    # Helper function to get button classes
    def get_button_classes(active_filter):
        return (
            "allocation-btn active" if active_filter == 'Type' else "allocation-btn",
            "allocation-btn active" if active_filter == 'Subtype' else "allocation-btn",
            "allocation-btn active" if active_filter == 'Currency' else "allocation-btn",
            "allocation-btn active" if active_filter == 'Label' else "allocation-btn"
        )
    
    # Get button classes for each bank
    bbva_classes = get_button_classes(bbva_filter)
    ubs_classes = get_button_classes(ubs_filter)
    lo_classes = get_button_classes(lo_filter)
    
    return (bbva_fig, ubs_fig, lo_fig, 
            *bbva_classes, *ubs_classes, *lo_classes)


# Callback for Summary page asset allocation chart
@app.callback(
    [Output('summary-allocation-chart', 'figure'),
     Output('summary-btn-type', 'className'),
     Output('summary-btn-subtype', 'className'),
     Output('summary-btn-currency', 'className'),
     Output('summary-btn-label', 'className')],
    [Input('summary-btn-type', 'n_clicks'),
     Input('summary-btn-subtype', 'n_clicks'),
     Input('summary-btn-currency', 'n_clicks'),
     Input('summary-btn-label', 'n_clicks')]
)
def update_summary_allocation_chart(btn_type_clicks, btn_subtype_clicks, btn_currency_clicks, btn_label_clicks):
    # Determine which button was clicked
    ctx = dash.callback_context
    if not ctx.triggered:
        groupby_value = 'Type'  # Default value
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'summary-btn-type':
            groupby_value = 'Type'
        elif button_id == 'summary-btn-subtype':
            groupby_value = 'Subtype'
        elif button_id == 'summary-btn-currency':
            groupby_value = 'Currency'
        elif button_id == 'summary-btn-label':
            groupby_value = 'Label'
        else:
            groupby_value = 'Type'  # Default fallback
    
    if data is None:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#e5e5e5',
            height=300,
            annotations=[dict(text="No data available", x=0.5, y=0.5, showarrow=False, font=dict(size=16, color='#e5e5e5'))]
        )
        return empty_fig, 'allocation-btn', 'allocation-btn', 'allocation-btn', 'allocation-btn'
    
    # Create combined portfolio data
    combined_data = pd.concat([data['portfolio']['bbva'], data['portfolio']['ubs'], data['portfolio']['lo']])
    
    # Create the pie chart
    fig = create_pie_chart(combined_data, 'Overall', groupby_value)
    
    # Update button states
    active_button = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'summary-btn-type'
    
    return (
        fig,
        'allocation-btn active' if active_button == 'summary-btn-type' else 'allocation-btn',
        'allocation-btn active' if active_button == 'summary-btn-subtype' else 'allocation-btn',
        'allocation-btn active' if active_button == 'summary-btn-currency' else 'allocation-btn',
        'allocation-btn active' if active_button == 'summary-btn-label' else 'allocation-btn'
    )


# Callback for benchmark comparison chart
@app.callback(
    Output('benchmark-comparison-chart', 'figure'),
    [Input('summary-btn', 'n_clicks'),
     Input('banks-btn', 'n_clicks'),
     Input('holdings-btn', 'n_clicks')]
)
def update_benchmark_comparison_chart(summary_clicks, banks_clicks, holdings_clicks):
    """Update benchmark comparison chart on Summary page load"""
    # Only update when on summary page
    ctx = dash.callback_context
    if not ctx.triggered:
        # Initial load - show on summary page
        pass
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id != 'summary-btn':
            # Not on summary page, return empty figure
            return go.Figure().update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e5e5e5',
                height=400
            )
    
    try:
        # Load benchmark data
        benchmarks_df = load_benchmarks()
        if benchmarks_df is None:
            return go.Figure().add_annotation(
                text="No benchmark data available",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color='#e5e5e5')
            ).update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e5e5e5',
                height=400
            )
        
        # Create YTD data
        monthly_returns, cumulative_returns = create_benchmark_ytd(benchmarks_df)
        if cumulative_returns is None:
            return go.Figure().add_annotation(
                text="No cumulative returns data available",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color='#e5e5e5')
            ).update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e5e5e5',
                height=400
            )
        
        # Check if TWRR data is available
        if data is None or 'twrr' not in data:
            return go.Figure().add_annotation(
                text="No TWRR data available",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color='#e5e5e5')
            ).update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e5e5e5',
                height=400
            )
        
        # Create the benchmark summary chart
        chart = create_benchmark_summary_chart(cumulative_returns, data['twrr'])
        if chart is None:
            return go.Figure().add_annotation(
                text="Error creating benchmark chart",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color='#e5e5e5')
            ).update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e5e5e5',
                height=400
            )
        
        return chart
        
    except Exception as e:
        print(f"Error updating benchmark comparison chart: {e}")
        return go.Figure().add_annotation(
            text=f"Error loading benchmark data: {str(e)}",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color='#e5e5e5')
        ).update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#e5e5e5',
            height=400
        )


@app.callback(
    [Output("main-content", "children"),
     Output("summary-btn", "className"),
     Output("banks-btn", "className"),
     Output("holdings-btn", "className"),
     Output("risk-btn", "className")],
    [Input("summary-btn", "n_clicks"),
     Input("banks-btn", "n_clicks"),
     Input("holdings-btn", "n_clicks"),
     Input("risk-btn", "n_clicks")]
)
def update_main_content(summary_clicks, banks_clicks, holdings_clicks, risk_clicks):
    if data is None:
        return (html.Div([
            html.H1("Portfolio Dashboard", className="page-title"),
            html.P("Error loading data. Please check your Excel files.", className="error-message")
        ]), "nav-link", "nav-link", "nav-link", "nav-link")
    
    # Determine which page to show
    ctx = dash.callback_context
    if not ctx.triggered:
        page = "summary"  # Default to summary page
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'summary-btn':
            page = "summary"
        elif button_id == 'banks-btn':
            page = "banks"
        elif button_id == 'holdings-btn':
            page = "holdings"
        elif button_id == 'risk-btn':
            page = "risk"
        else:
            page = "summary"  # Default fallback
    
    # Get data
    banks = ['BBVA', 'UBS', 'LO']
    bank_data = data['portfolio']
    twrr_data = data['twrr']
    evolution_data = data['evolution']
    
    if page == "summary":
        return (create_summary_page(bank_data, twrr_data, evolution_data), 
                "nav-link active", "nav-link", "nav-link", "nav-link")
    elif page == "banks":
        return (create_banks_page(bank_data, twrr_data, evolution_data), 
                "nav-link", "nav-link active", "nav-link", "nav-link")
    elif page == "holdings":
        return (create_holdings_page(bank_data), 
                "nav-link", "nav-link", "nav-link active", "nav-link")
    else:  # risk page
        return (create_risk_page(bank_data), 
                "nav-link", "nav-link", "nav-link", "nav-link active")

def create_summary_page(bank_data, twrr_data, evolution_data):
    """Create the Summary page layout"""
    # Calculate overall metrics using Master data
    total_value = evolution_data['master'].iloc[-1]['Value']
    total_pnl = evolution_data['master'].loc['Profit & Loss', 'Value']
    
    # Calculate liabilities as sum of Cash assets with negative values
    total_liabilities = 0
    for bank in ['bbva', 'ubs', 'lo']:
        cash_assets = bank_data[bank][bank_data[bank]['Type'] == 'Cash']
        negative_cash = cash_assets[cash_assets['Value_USD'] < 0]
        total_liabilities += negative_cash['Value_USD'].sum()
    
    # Get YTD return from Master data
    total_ytd_return = twrr_data['master'].iloc[-1]['Cum. TWRR']
    
    return html.Div([
        html.Div([
            # Page Header
            html.Div([
                html.H1("Portfolio Summary", className="page-title"),
                html.P("Overview of your entire portfolio performance across all banking institutions", className="page-subtitle"),
            ], className="page-header"),
            
            # Summary Grid (2 rows, 3 columns)
            html.Div([
                # Row 1
                html.Div([
                    # Container 1: Metrics (2x2 grid)
                    html.Div([
                        html.H4("Portfolio Metrics", className="chart-title"),
                        html.Div([
                            # Top Left: Total Value
                            html.Div([
                                html.H4("Total Value ($)", className="metric-title"),
                                html.P(f"${total_value:,.0f}", className="metric-value"),
                            ], className="metric-window"),
                            
                            # Top Right: YTD Return
                            html.Div([
                                html.H4("YTD Return", className="metric-title"),
                                html.P(f"{'+' if total_ytd_return >= 0 else ''}{total_ytd_return:.2f}%", 
                                       className="metric-value",
                                       style={"color": "#22c55e" if total_ytd_return >= 0 else "#ef4444"}),
                            ], className="metric-window"),
                            
                            # Bottom Left: P&L
                            html.Div([
                                html.H4("Total P&L ($)", className="metric-title"),
                                html.P(f"{'+' if total_pnl >= 0 else ''}${total_pnl:,.0f}", 
                                       className="metric-value",
                                       style={"color": "#22c55e" if total_pnl >= 0 else "#ef4444"}),
                            ], className="metric-window"),
                            
                            # Bottom Right: Liabilities
                            html.Div([
                                html.H4("Liabilities ($)", className="metric-title"),
                                html.P(f"-${abs(total_liabilities):,.0f}", 
                                       className="metric-value",
                                       style={"color": "#ef4444" if total_liabilities < 0 else "#e5e5e5"}),
                            ], className="metric-window"),
                            
                            # Additional Metric 1: Empty
                            html.Div([
                                html.H4("", className="metric-title"),
                                html.P("", className="metric-value"),
                            ], className="metric-window"),
                            
                            # Additional Metric 2: Empty
                            html.Div([
                                html.H4("", className="metric-title"),
                                html.P("", className="metric-value"),
                            ], className="metric-window"),
                        ], className="metrics-grid-2x3")
                    ], className="summary-container summary-metrics"),
                    
                    # Container 2: Asset Allocation by Type
                    html.Div([
                        html.H4("Asset Allocation", className="chart-title"),
                        html.Div([
                            html.Button("Type", id="summary-btn-type", n_clicks=0, className="allocation-btn active"),
                            html.Button("Subtype", id="summary-btn-subtype", n_clicks=0, className="allocation-btn"),
                            html.Button("Currency", id="summary-btn-currency", n_clicks=0, className="allocation-btn"),
                            html.Button("Label", id="summary-btn-label", n_clicks=0, className="allocation-btn")
                        ], className="button-group"),
                        dcc.Graph(
                            id='summary-allocation-chart',
                            config={'displayModeBar': False}
                        )
                    ], className="summary-container"),
                    
                    # Container 3: Benchmark Comparison
                    html.Div([
                        html.H4("Benchmark Comparison", className="chart-title"),
                        dcc.Graph(
                            id='benchmark-comparison-chart',
                            config={'displayModeBar': False}
                        )
                    ], className="summary-container")
                ], className="summary-row"),
                
                # Row 2
                html.Div([
                    # Container 4: Bank Performance Comparison
                    html.Div([
                        html.H4("Bank Performance Comparison", className="chart-title"),
                        dcc.Graph(
                            figure=create_bank_comparison_chart(twrr_data),
                            config={'displayModeBar': False}
                        )
                    ], className="summary-container"),
                    
                    # Container 5: Top Holdings
                    html.Div([
                        html.H4("Top Holdings", className="chart-title"),
                        dash_table.DataTable(
                            data=get_top_holdings(bank_data),
                            columns=[
                                {"name": "Issuer", "id": "Issuer"},
                                {"name": "Bank", "id": "Bank"},
                                {"name": "Value (USD)", "id": "Value_USD", "type": "numeric", "format": {"specifier": ",.2f"}}
                            ],
                            style_cell={
                                'backgroundColor': '#1a1a1a',
                                'color': '#ffffff',
                                'textAlign': 'left',
                                'fontFamily': 'Inter, sans-serif',
                                'fontSize': '12px',
                                'padding': '8px',
                                'whiteSpace': 'normal',
                                'height': 'auto',
                                'minWidth': '80px'
                            },
                            style_header={
                                'backgroundColor': '#2a2a2a',
                                'color': '#f5f5f5',
                                'fontWeight': '600',
                                'border': '1px solid #404040',
                                'textAlign': 'center',
                                'fontSize': '12px',
                                'padding': '8px',
                                'minWidth': '80px'
                            },
                            style_data={
                                'border': '1px solid #404040',
                                'whiteSpace': 'normal',
                                'height': 'auto'
                            },
                            style_data_conditional=[
                                {
                                    'if': {'row_index': 'odd'},
                                    'backgroundColor': '#252525'
                                },
                                {
                                    'if': {'column_id': 'Value_USD'},
                                    'textAlign': 'right'
                                }
                            ],
                            page_size=8,
                            sort_action="native",
                            style_table={'overflowX': 'auto', 'height': '300px', 'width': '100%'}
                        )
                    ], className="summary-container"),
                    
                    # Container 6: Portfolio Evolution
                    html.Div([
                        html.H4("Portfolio Evolution", className="chart-title"),
                        dash_table.DataTable(
                            data=get_evolution_data(evolution_data, 'master').to_dict('records'),
                            columns=[
                                {"name": "P&L", "id": "P&L"},
                                {"name": "Value", "id": "Value", "type": "numeric", "format": {"specifier": ","}}
                            ],
                            style_cell={
                                'backgroundColor': '#1a1a1a',
                                'color': '#ffffff',
                                'textAlign': 'left',
                                'fontFamily': 'Inter, sans-serif',
                                'fontSize': '13px',
                                'padding': '10px',
                                'whiteSpace': 'normal',
                                'height': 'auto',
                                'minWidth': '80px'
                            },
                            style_header={
                                'backgroundColor': '#2a2a2a',
                                'color': '#f5f5f5',
                                'fontWeight': '600',
                                'border': '1px solid #404040',
                                'textAlign': 'center',
                                'fontSize': '13px',
                                'padding': '10px',
                                'minWidth': '80px'
                            },
                            style_data={
                                'border': '1px solid #404040',
                                'whiteSpace': 'normal',
                                'height': 'auto'
                            },
                            style_data_conditional=[
                                {
                                    'if': {'row_index': 'odd'},
                                    'backgroundColor': '#252525'
                                },
                                {
                                    'if': {'column_id': 'Value'},
                                    'textAlign': 'right'
                                },
                                {
                                    'if': {
                                        'filter_query': '{P&L} = "Profit & Loss" && {Value} > 0'
                                    },
                                    'color': '#22c55e',
                                    'fontWeight': 'bold'
                                },
                                {
                                    'if': {
                                        'filter_query': '{P&L} = "Profit & Loss" && {Value} < 0'
                                    },
                                    'color': '#ef4444',
                                    'fontWeight': 'bold'
                                }
                            ],
                            page_size=10,
                            sort_action="native",
                            style_table={'overflowX': 'auto', 'height': '300px', 'width': '100%'}
                        )
                    ], className="summary-container")
                ], className="summary-row")
            ], className="summary-grid")
        ], className="page")
    ])

def create_banks_page(bank_data, twrr_data, evolution_data):
    """Create the Bank Analysis page layout"""
    
    return html.Div([
        html.Div([
            # Page Header
            html.Div([
                html.H1("Bank Performance Analysis", className="page-title"),
                html.P("Detailed analysis of portfolio performance across all banking institutions", className="page-subtitle"),
            ], className="page-header"),
            
            # Main Bank Columns - All containers for each bank in single columns
            html.Div([
                # BBVA Main Column
                html.Div([
                    # BBVA Header
                    html.Div([
                        html.H2("BBVA", className="bank-name")
                    ], className="bank-header bbva-header"),
                    
                    # BBVA Metrics
                    html.Div([
                        html.Div([
                            html.H4("Total Value ($)", className="metric-title"),
                            html.P(f"${evolution_data['bbva'].iloc[-1]['Value']:,.0f}", className="metric-value"),
                        ], className="metric-window"),
                        html.Div([
                            html.H4("YTD Return", className="metric-title"),
                            html.P(f"{'+' if twrr_data['bbva'].iloc[-1]['Cum. TWRR'] >= 0 else ''}{twrr_data['bbva'].iloc[-1]['Cum. TWRR']:.2f}%", 
                                   className="metric-value",
                                   style={"color": "#22c55e" if twrr_data['bbva'].iloc[-1]['Cum. TWRR'] >= 0 else "#ef4444"}),
                        ], className="metric-window")
                    ], className="metrics-container"),
                    
                    # BBVA Asset Allocation Chart
                    html.Div([
                        html.H4("Asset Allocation", className="chart-title"),
                        html.Div([
                            html.Button("Type", id="bbva-btn-type", n_clicks=0, className="allocation-btn active"),
                            html.Button("Subtype", id="bbva-btn-subtype", n_clicks=0, className="allocation-btn"),
                            html.Button("Currency", id="bbva-btn-currency", n_clicks=0, className="allocation-btn"),
                            html.Button("Label", id="bbva-btn-label", n_clicks=0, className="allocation-btn")
                        ], className="button-group"),
                        dcc.Graph(
                            id='bbva-allocation-chart',
                            style={"height": "350px"},
                            config={'displayModeBar': False}
                        )
                    ], className="chart-container bbva-column"),
                    
                    # BBVA Portfolio Table
                    html.Div([
                        html.H4("Portfolio Holdings", className="chart-title"),
                        dash_table.DataTable(
                            data=get_bank_portfolio_data(bank_data, 'bbva').to_dict('records'),
                            columns=[
                                {"name": "Issuer", "id": "Issuer"},
                                {"name": "Subtype", "id": "Subtype"},
                                {"name": "Change %", "id": "Change_pct", "type": "numeric", "format": {"specifier": ","}},
                                {"name": "Value (USD)", "id": "Value_USD", "type": "numeric", "format": {"specifier": ","}}
                            ],
                            style_cell={
                                'backgroundColor': '#1a1a1a',
                                'color': '#ffffff',
                                'textAlign': 'left',
                                'fontFamily': 'Inter, sans-serif',
                                'fontSize': '10px',
                                'padding': '6px',
                                'whiteSpace': 'normal',
                                'height': 'auto',
                                'minWidth': '80px'
                            },
                            style_header={
                                'backgroundColor': '#2a2a2a',
                                'color': '#f5f5f5',
                                'fontWeight': '600',
                                'border': '1px solid #404040',
                                'textAlign': 'center',
                                'fontSize': '10px',
                                'padding': '6px',
                                'minWidth': '80px'
                            },
                            style_data={
                                'border': '1px solid #404040',
                                'whiteSpace': 'normal',
                                'height': 'auto'
                            },
                            style_data_conditional=[
                                {
                                    'if': {'row_index': 'odd'},
                                    'backgroundColor': '#252525'
                                },
                                {
                                    'if': {'column_id': ['Change_pct', 'Value_USD']},
                                    'textAlign': 'right'
                                },
                                {
                                    'if': {
                                        'filter_query': '{Change_pct} > 0',
                                        'column_id': 'Change_pct'
                                    },
                                    'color': '#22c55e',
                                    'fontWeight': 'bold'
                                },
                                {
                                    'if': {
                                        'filter_query': '{Change_pct} < 0',
                                        'column_id': 'Change_pct'
                                    },
                                    'color': '#ef4444',
                                    'fontWeight': 'bold'
                                }
                            ],
                            page_size=8,
                            sort_action="native",
                            style_table={'overflowX': 'auto', 'height': '250px', 'width': '100%'}
                        )
                    ], className="chart-container bbva-column"),
                    
                    # BBVA TWRR Performance Chart
                    html.Div([
                        html.H4("TWRR Performance", className="chart-title"),
                        dcc.Graph(
                            figure=create_twrr_chart(twrr_data['bbva'], 'BBVA', '#9ca3af'),
                            config={'displayModeBar': False}
                        )
                    ], className="chart-container bbva-column"),
                    
                    # BBVA P&L Evolution Table
                    html.Div([
                        html.H4("P&L Evolution", className="chart-title"),
                        dash_table.DataTable(
                            data=get_evolution_data(evolution_data, 'bbva').to_dict('records'),
                            columns=[
                                {"name": "P&L", "id": "P&L"},
                                {"name": "Value", "id": "Value", "type": "numeric", "format": {"specifier": ","}}
                            ],
                            style_cell={
                                'backgroundColor': '#1a1a1a',
                                'color': '#ffffff',
                                'textAlign': 'left',
                                'fontFamily': 'Inter, sans-serif',
                                'fontSize': '13px',
                                'padding': '10px',
                                'whiteSpace': 'normal',
                                'height': 'auto',
                                'minWidth': '80px'
                            },
                            style_header={
                                'backgroundColor': '#2a2a2a',
                                'color': '#f5f5f5',
                                'fontWeight': '600',
                                'border': '1px solid #404040',
                                'textAlign': 'center',
                                'fontSize': '13px',
                                'padding': '10px',
                                'minWidth': '80px'
                            },
                            style_data={
                                'border': '1px solid #404040',
                                'whiteSpace': 'normal',
                                'height': 'auto'
                            },
                            style_data_conditional=[
                                {
                                    'if': {'row_index': 'odd'},
                                    'backgroundColor': '#252525'
                                },
                                {
                                    'if': {'column_id': 'Value'},
                                    'textAlign': 'right'
                                },
                                {
                                    'if': {
                                        'filter_query': '{P&L} = "Profit & Loss" && {Value} > 0'
                                    },
                                    'color': '#22c55e',
                                    'fontWeight': 'bold'
                                },
                                {
                                    'if': {
                                        'filter_query': '{P&L} = "Profit & Loss" && {Value} < 0'
                                    },
                                    'color': '#ef4444',
                                    'fontWeight': 'bold'
                                }
                            ],
                            page_size=10,
                            sort_action="native",
                            style_table={'overflowX': 'auto', 'height': '300px', 'width': '100%'}
                        )
                    ], className="chart-container bbva-column")
                ], className="main-bank-column bbva-main-column"),
                
                # UBS Main Column
                html.Div([
                    # UBS Header
                    html.Div([
                        html.H2("UBS", className="bank-name")
                    ], className="bank-header ubs-header"),
                    
                    # UBS Metrics
                    html.Div([
                        html.Div([
                            html.H4("Total Value ($)", className="metric-title"),
                            html.P(f"${evolution_data['ubs'].iloc[-1]['Value']:,.0f}", className="metric-value"),
                        ], className="metric-window"),
                        html.Div([
                            html.H4("YTD Return", className="metric-title"),
                            html.P(f"{'+' if twrr_data['ubs'].iloc[-1]['Cum. TWRR'] >= 0 else ''}{twrr_data['ubs'].iloc[-1]['Cum. TWRR']:.2f}%", 
                                   className="metric-value",
                                   style={"color": "#22c55e" if twrr_data['ubs'].iloc[-1]['Cum. TWRR'] >= 0 else "#ef4444"}),
                        ], className="metric-window")
                    ], className="metrics-container"),
                    
                    # UBS Asset Allocation Chart
                    html.Div([
                        html.H4("Asset Allocation", className="chart-title"),
                        html.Div([
                            html.Button("Type", id="ubs-btn-type", n_clicks=0, className="allocation-btn active"),
                            html.Button("Subtype", id="ubs-btn-subtype", n_clicks=0, className="allocation-btn"),
                            html.Button("Currency", id="ubs-btn-currency", n_clicks=0, className="allocation-btn"),
                            html.Button("Label", id="ubs-btn-label", n_clicks=0, className="allocation-btn")
                        ], className="button-group"),
                        dcc.Graph(
                            id='ubs-allocation-chart',
                            style={"height": "350px"},
                            config={'displayModeBar': False}
                        )
                    ], className="chart-container ubs-column"),
                    
                    # UBS Portfolio Table
                    html.Div([
                        html.H4("Portfolio Holdings", className="chart-title"),
                        dash_table.DataTable(
                            data=get_bank_portfolio_data(bank_data, 'ubs').to_dict('records'),
                            columns=[
                                {"name": "Issuer", "id": "Issuer"},
                                {"name": "Subtype", "id": "Subtype"},
                                {"name": "Change %", "id": "Change_pct", "type": "numeric", "format": {"specifier": ","}},
                                {"name": "Value (USD)", "id": "Value_USD", "type": "numeric", "format": {"specifier": ","}}
                            ],
                            style_cell={
                                'backgroundColor': '#1a1a1a',
                                'color': '#ffffff',
                                'textAlign': 'left',
                                'fontFamily': 'Inter, sans-serif',
                                'fontSize': '10px',
                                'padding': '6px',
                                'whiteSpace': 'normal',
                                'height': 'auto',
                                'minWidth': '80px'
                            },
                            style_header={
                                'backgroundColor': '#2a2a2a',
                                'color': '#f5f5f5',
                                'fontWeight': '600',
                                'border': '1px solid #404040',
                                'textAlign': 'center',
                                'fontSize': '10px',
                                'padding': '6px',
                                'minWidth': '80px'
                            },
                            style_data={
                                'border': '1px solid #404040',
                                'whiteSpace': 'normal',
                                'height': 'auto'
                            },
                            style_data_conditional=[
                                {
                                    'if': {'row_index': 'odd'},
                                    'backgroundColor': '#252525'
                                },
                                {
                                    'if': {'column_id': ['Change_pct', 'Value_USD']},
                                    'textAlign': 'right'
                                },
                                {
                                    'if': {
                                        'filter_query': '{Change_pct} > 0',
                                        'column_id': 'Change_pct'
                                    },
                                    'color': '#22c55e',
                                    'fontWeight': 'bold'
                                },
                                {
                                    'if': {
                                        'filter_query': '{Change_pct} < 0',
                                        'column_id': 'Change_pct'
                                    },
                                    'color': '#ef4444',
                                    'fontWeight': 'bold'
                                }
                            ],
                            page_size=8,
                            sort_action="native",
                            style_table={'overflowX': 'auto', 'height': '250px', 'width': '100%'}
                        )
                    ], className="chart-container ubs-column"),
                    
                    # UBS TWRR Performance Chart
                    html.Div([
                        html.H4("TWRR Performance", className="chart-title"),
                        dcc.Graph(
                            figure=create_twrr_chart(twrr_data['ubs'], 'UBS', '#3b82f6'),
                            config={'displayModeBar': False}
                        )
                    ], className="chart-container ubs-column"),
                    
                    # UBS P&L Evolution Table
                    html.Div([
                        html.H4("P&L Evolution", className="chart-title"),
                        dash_table.DataTable(
                            data=get_evolution_data(evolution_data, 'ubs').to_dict('records'),
                            columns=[
                                {"name": "P&L", "id": "P&L"},
                                {"name": "Value", "id": "Value", "type": "numeric", "format": {"specifier": ","}}
                            ],
                            style_cell={
                                'backgroundColor': '#1a1a1a',
                                'color': '#ffffff',
                                'textAlign': 'left',
                                'fontFamily': 'Inter, sans-serif',
                                'fontSize': '13px',
                                'padding': '10px',
                                'whiteSpace': 'normal',
                                'height': 'auto',
                                'minWidth': '80px'
                            },
                            style_header={
                                'backgroundColor': '#2a2a2a',
                                'color': '#f5f5f5',
                                'fontWeight': '600',
                                'border': '1px solid #404040',
                                'textAlign': 'center',
                                'fontSize': '13px',
                                'padding': '10px',
                                'minWidth': '80px'
                            },
                            style_data={
                                'border': '1px solid #404040',
                                'whiteSpace': 'normal',
                                'height': 'auto'
                            },
                            style_data_conditional=[
                                {
                                    'if': {'row_index': 'odd'},
                                    'backgroundColor': '#252525'
                                },
                                {
                                    'if': {'column_id': 'Value'},
                                    'textAlign': 'right'
                                },
                                {
                                    'if': {
                                        'filter_query': '{P&L} = "Profit & Loss" && {Value} > 0'
                                    },
                                    'color': '#22c55e',
                                    'fontWeight': 'bold'
                                },
                                {
                                    'if': {
                                        'filter_query': '{P&L} = "Profit & Loss" && {Value} < 0'
                                    },
                                    'color': '#ef4444',
                                    'fontWeight': 'bold'
                                }
                            ],
                            page_size=10,
                            sort_action="native",
                            style_table={'overflowX': 'auto', 'height': '300px', 'width': '100%'}
                        )
                    ], className="chart-container ubs-column")
                ], className="main-bank-column ubs-main-column"),
                
                # LO Main Column
                html.Div([
                    # LO Header
                    html.Div([
                        html.H2("LO", className="bank-name")
                    ], className="bank-header lo-header"),
                    
                    # LO Metrics
                    html.Div([
                        html.Div([
                            html.H4("Total Value ($)", className="metric-title"),
                            html.P(f"${evolution_data['lo'].iloc[-1]['Value']:,.0f}", className="metric-value"),
                        ], className="metric-window"),
                        html.Div([
                            html.H4("YTD Return", className="metric-title"),
                            html.P(f"{'+' if twrr_data['lo'].iloc[-1]['Cum. TWRR'] >= 0 else ''}{twrr_data['lo'].iloc[-1]['Cum. TWRR']:.2f}%", 
                                   className="metric-value",
                                   style={"color": "#22c55e" if twrr_data['lo'].iloc[-1]['Cum. TWRR'] >= 0 else "#ef4444"}),
                        ], className="metric-window")
                    ], className="metrics-container"),
                    
                    # LO Asset Allocation Chart
                    html.Div([
                        html.H4("Asset Allocation", className="chart-title"),
                        html.Div([
                            html.Button("Type", id="lo-btn-type", n_clicks=0, className="allocation-btn active"),
                            html.Button("Subtype", id="lo-btn-subtype", n_clicks=0, className="allocation-btn"),
                            html.Button("Currency", id="lo-btn-currency", n_clicks=0, className="allocation-btn"),
                            html.Button("Label", id="lo-btn-label", n_clicks=0, className="allocation-btn")
                        ], className="button-group"),
                        dcc.Graph(
                            id='lo-allocation-chart',
                            style={"height": "350px"},
                            config={'displayModeBar': False}
                        )
                    ], className="chart-container lo-column"),
                    
                    # LO Portfolio Table
                    html.Div([
                        html.H4("Portfolio Holdings", className="chart-title"),
                        dash_table.DataTable(
                            data=get_bank_portfolio_data(bank_data, 'lo').to_dict('records'),
                            columns=[
                                {"name": "Issuer", "id": "Issuer"},
                                {"name": "Subtype", "id": "Subtype"},
                                {"name": "Change %", "id": "Change_pct", "type": "numeric", "format": {"specifier": ","}},
                                {"name": "Value (USD)", "id": "Value_USD", "type": "numeric", "format": {"specifier": ","}}
                            ],
                            style_cell={
                                'backgroundColor': '#1a1a1a',
                                'color': '#ffffff',
                                'textAlign': 'left',
                                'fontFamily': 'Inter, sans-serif',
                                'fontSize': '10px',
                                'padding': '6px',
                                'whiteSpace': 'normal',
                                'height': 'auto',
                                'minWidth': '80px'
                            },
                            style_header={
                                'backgroundColor': '#2a2a2a',
                                'color': '#f5f5f5',
                                'fontWeight': '600',
                                'border': '1px solid #404040',
                                'textAlign': 'center',
                                'fontSize': '10px',
                                'padding': '6px',
                                'minWidth': '80px'
                            },
                            style_data={
                                'border': '1px solid #404040',
                                'whiteSpace': 'normal',
                                'height': 'auto'
                            },
                            style_data_conditional=[
                                {
                                    'if': {'row_index': 'odd'},
                                    'backgroundColor': '#252525'
                                },
                                {
                                    'if': {'column_id': ['Change_pct', 'Value_USD']},
                                    'textAlign': 'right'
                                },
                                {
                                    'if': {
                                        'filter_query': '{Change_pct} > 0',
                                        'column_id': 'Change_pct'
                                    },
                                    'color': '#22c55e',
                                    'fontWeight': 'bold'
                                },
                                {
                                    'if': {
                                        'filter_query': '{Change_pct} < 0',
                                        'column_id': 'Change_pct'
                                    },
                                    'color': '#ef4444',
                                    'fontWeight': 'bold'
                                }
                            ],
                            page_size=8,
                            sort_action="native",
                            style_table={'overflowX': 'auto', 'height': '250px', 'width': '100%'}
                        )
                    ], className="chart-container lo-column"),
                    
                    # LO TWRR Performance Chart
                    html.Div([
                        html.H4("TWRR Performance", className="chart-title"),
                        dcc.Graph(
                            figure=create_twrr_chart(twrr_data['lo'], 'LO', '#93c5fd'),
                            config={'displayModeBar': False}
                        )
                    ], className="chart-container lo-column"),
                    
                    # LO P&L Evolution Table
                    html.Div([
                        html.H4("P&L Evolution", className="chart-title"),
                        dash_table.DataTable(
                            data=get_evolution_data(evolution_data, 'lo').to_dict('records'),
                            columns=[
                                {"name": "P&L", "id": "P&L"},
                                {"name": "Value", "id": "Value", "type": "numeric", "format": {"specifier": ","}}
                            ],
                            style_cell={
                                'backgroundColor': '#1a1a1a',
                                'color': '#ffffff',
                                'textAlign': 'left',
                                'fontFamily': 'Inter, sans-serif',
                                'fontSize': '13px',
                                'padding': '10px',
                                'whiteSpace': 'normal',
                                'height': 'auto',
                                'minWidth': '80px'
                            },
                            style_header={
                                'backgroundColor': '#2a2a2a',
                                'color': '#f5f5f5',
                                'fontWeight': '600',
                                'border': '1px solid #404040',
                                'textAlign': 'center',
                                'fontSize': '13px',
                                'padding': '10px',
                                'minWidth': '80px'
                            },
                            style_data={
                                'border': '1px solid #404040',
                                'whiteSpace': 'normal',
                                'height': 'auto'
                            },
                            style_data_conditional=[
                                {
                                    'if': {'row_index': 'odd'},
                                    'backgroundColor': '#252525'
                                },
                                {
                                    'if': {'column_id': 'Value'},
                                    'textAlign': 'right'
                                },
                                {
                                    'if': {
                                        'filter_query': '{P&L} = "Profit & Loss" && {Value} > 0'
                                    },
                                    'color': '#22c55e',
                                    'fontWeight': 'bold'
                                },
                                {
                                    'if': {
                                        'filter_query': '{P&L} = "Profit & Loss" && {Value} < 0'
                                    },
                                    'color': '#ef4444',
                                    'fontWeight': 'bold'
                                }
                            ],
                            page_size=10,
                            sort_action="native",
                            style_table={'overflowX': 'auto', 'height': '300px', 'width': '100%'}
                        )
                    ], className="chart-container lo-column")
                ], className="main-bank-column lo-main-column"),
            ], className="row main-bank-row")
        ], className="page")
    ])

def create_holdings_page(bank_data):
    """Create the Portfolio Holdings page layout"""
    # Get processed portfolio holdings data
    filtered_data = get_portfolio_holdings_data(bank_data)
    
    return html.Div([
        html.Div([
            # Page Header
            html.Div([
                html.H1("Portfolio Holdings", className="page-title"),
                html.P("Complete view of all portfolio holdings across all banking institutions", className="page-subtitle"),
            ], className="page-header"),
            
            # Portfolio Holdings Table
            html.Div([
                html.H4("All Holdings", className="chart-title"),
                dash_table.DataTable(
                    data=filtered_data.to_dict('records'),
                    columns=[
                        {"name": "Type", "id": "Type"},
                        {"name": "Subtype", "id": "Subtype"},
                        {"name": "Currency", "id": "Currency"},
                        {"name": "Bank", "id": "Bank"},
                        {"name": "Label", "id": "Label"},
                        {"name": "Maturity", "id": "Maturity", "type": "datetime", "format": {"specifier": "%Y-%m-%d"}},
                        {"name": "Issuer", "id": "Issuer"},
                        {"name": "Change %", "id": "Change_pct", "type": "numeric", "format": {"specifier": ","}},
                        {"name": "Change % USD", "id": "Change_pct_USD", "type": "numeric", "format": {"specifier": ","}},
                        {"name": "Value (USD)", "id": "Value_USD", "type": "numeric", "format": {"specifier": ","}}
                    ],
                    style_cell={
                        'backgroundColor': '#1a1a1a',
                        'color': '#ffffff',
                        'textAlign': 'left',
                        'fontFamily': 'Inter, sans-serif',
                        'fontSize': '12px',
                        'padding': '8px',
                        'whiteSpace': 'normal',
                        'height': 'auto',
                        'minWidth': '80px'
                    },
                    style_header={
                        'backgroundColor': '#2a2a2a',
                        'color': '#f5f5f5',
                        'fontWeight': '600',
                        'border': '1px solid #404040',
                        'textAlign': 'center',
                        'fontSize': '12px',
                        'padding': '8px',
                        'minWidth': '80px'
                    },
                    style_data={
                        'border': '1px solid #404040',
                        'whiteSpace': 'normal',
                        'height': 'auto'
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': '#252525'
                        },
                        {
                            'if': {'column_id': ['Change_pct', 'Change_pct_USD', 'Value_USD']},
                            'textAlign': 'right'
                        },
                        {
                            'if': {'column_id': 'Issuer'},
                            'color': '#FF8C00',
                            'fontWeight': '600'
                        },
                        {
                            'if': {
                                'filter_query': '{Change_pct} > 0',
                                'column_id': 'Change_pct'
                            },
                            'color': '#22c55e',
                            'fontWeight': 'bold'
                        },
                        {
                            'if': {
                                'filter_query': '{Change_pct} < 0',
                                'column_id': 'Change_pct'
                            },
                            'color': '#ef4444',
                            'fontWeight': 'bold'
                        },
                        {
                            'if': {
                                'filter_query': '{Change_pct_USD} > 0',
                                'column_id': 'Change_pct_USD'
                            },
                            'color': '#22c55e',
                            'fontWeight': 'bold'
                        },
                        {
                            'if': {
                                'filter_query': '{Change_pct_USD} < 0',
                                'column_id': 'Change_pct_USD'
                            },
                            'color': '#ef4444',
                            'fontWeight': 'bold'
                        }
                    ],
                    page_size=20,
                    sort_action="native",
                    filter_action="native",
                    style_table={'overflowX': 'auto', 'height': '600px', 'width': '100%'}
                )
            ], className="summary-container")
        ], className="page")
    ])


def create_risk_page(bank_data):
    """Create the Risk Analysis page layout"""
    return html.Div([
        html.Div([
            html.H1("Risk Analysis", className="page-title"),
            
            # Risk Analysis Grid - 2 rows by 3 columns with wider containers
            html.Div([
                # Row 1 - Three containers side by side
                html.Div([
                    # Container 1: Reinvestment Risk
                    html.Div([
                        html.H4("Reinvestment Risk", className="chart-title"),
                        # Filter buttons
                        html.Div([
                            html.Div([
                                html.Label("Filter by:", style={"color": "#e5e5e5", "fontSize": "14px", "marginRight": "15px"}),
                                html.Button(
                                    "Bank",
                                    id="reinvestment-filter-bank",
                                    n_clicks=1,
                                    className="filter-button active",
                                    style={
                                        "marginRight": "8px",
                                        "padding": "8px 16px",
                                        "backgroundColor": "#3b82f6",
                                        "color": "white",
                                        "border": "none",
                                        "borderRadius": "6px",
                                        "cursor": "pointer",
                                        "fontSize": "14px",
                                        "fontWeight": "500"
                                    }
                                ),
                                html.Button(
                                    "Currency",
                                    id="reinvestment-filter-currency",
                                    n_clicks=0,
                                    className="filter-button",
                                    style={
                                        "marginRight": "8px",
                                        "padding": "8px 16px",
                                        "backgroundColor": "#374151",
                                        "color": "#e5e5e5",
                                        "border": "1px solid #4b5563",
                                        "borderRadius": "6px",
                                        "cursor": "pointer",
                                        "fontSize": "14px",
                                        "fontWeight": "500"
                                    }
                                ),
                                html.Button(
                                    "Risk",
                                    id="reinvestment-filter-label",
                                    n_clicks=0,
                                    className="filter-button",
                                    style={
                                        "padding": "8px 16px",
                                        "backgroundColor": "#374151",
                                        "color": "#e5e5e5",
                                        "border": "1px solid #4b5563",
                                        "borderRadius": "6px",
                                        "cursor": "pointer",
                                        "fontSize": "14px",
                                        "fontWeight": "500"
                                    }
                                )
                            ], style={"display": "flex", "alignItems": "center", "marginBottom": "15px"})
                        ], style={"marginBottom": "10px"}),
                        dcc.Graph(
                            id='reinvestment-risk-chart',
                            config={'displayModeBar': False}
                        )
                    ], className="summary-container", style={"flex": "1", "marginRight": "15px", "minWidth": "400px"}),
                    
                    # Container 2: Concentration Risk
                    html.Div([
                        html.H4("Concentration Risk", className="chart-title"),
                        dcc.Graph(
                            id='concentration-risk-chart',
                            config={'displayModeBar': False}
                        )
                    ], className="summary-container", style={"flex": "1", "marginRight": "15px", "minWidth": "400px"}),
                    
                    # Container 3: Currency Risk
                    html.Div([
                        html.H4("Currency Risk", className="chart-title"),
                        dcc.Graph(
                            id='currency-risk-chart',
                            config={'displayModeBar': False}
                        )
                    ], className="summary-container", style={"flex": "1", "minWidth": "400px"})
                ], className="summary-row", style={"display": "flex", "marginBottom": "20px", "gap": "15px"}),
                
                # Row 2 - Three containers side by side
                html.Div([
                    # Container 4: Credit Risk
                    html.Div([
                        html.H4("Credit Risk", className="chart-title"),
                        dcc.Graph(
                            id='credit-risk-chart',
                            config={'displayModeBar': False}
                        )
                    ], className="summary-container", style={"flex": "1", "marginRight": "15px", "minWidth": "400px"}),
                    
                    # Container 5: Liquidity Risk
                    html.Div([
                        html.H4("Liquidity Risk", className="chart-title"),
                        html.P("Liquidity risk analysis will be implemented here", 
                               className="chart-subtitle", 
                               style={"color": "#9ca3af", "fontSize": "14px", "textAlign": "center", "marginTop": "50px"})
                    ], className="summary-container", style={"flex": "1", "marginRight": "15px", "minWidth": "400px"}),
                    
                    # Container 6: Market Risk
                    html.Div([
                        html.H4("Market Risk", className="chart-title"),
                        html.P("Market risk analysis will be implemented here", 
                               className="chart-subtitle", 
                               style={"color": "#9ca3af", "fontSize": "14px", "textAlign": "center", "marginTop": "50px"})
                    ], className="summary-container", style={"flex": "1", "minWidth": "400px"})
                ], className="summary-row", style={"display": "flex", "gap": "15px"})
            ], className="summary-grid", style={"maxWidth": "1400px", "margin": "0 auto"})
        ], className="page")
    ])


# Callback for reinvestment risk chart
@app.callback(
    [Output('reinvestment-risk-chart', 'figure'),
     Output('reinvestment-filter-bank', 'style'),
     Output('reinvestment-filter-currency', 'style'),
     Output('reinvestment-filter-label', 'style')],
    [Input('risk-btn', 'n_clicks'),
     Input('reinvestment-filter-bank', 'n_clicks'),
     Input('reinvestment-filter-currency', 'n_clicks'),
     Input('reinvestment-filter-label', 'n_clicks')]
)
def update_reinvestment_risk_chart(risk_clicks, bank_clicks, currency_clicks, label_clicks):
    """Update reinvestment risk chart on Risk Analysis page load or filter change"""
    try:
        # Determine which button was clicked
        ctx = dash.callback_context
        if not ctx.triggered:
            filter_by = 'Bank'
        else:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if 'bank' in button_id:
                filter_by = 'Bank'
            elif 'currency' in button_id:
                filter_by = 'Currency'
            elif 'label' in button_id:
                filter_by = 'Label'
            else:
                filter_by = 'Bank'
        
        # Create the bonds stacked chart with filter
        chart = create_reinvestment_risk_chart(data['portfolio'], filter_by)
        if chart is None:
            chart = go.Figure().add_annotation(
                text="No bonds data available for reinvestment risk analysis",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color='#e5e5e5')
            ).update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e5e5e5',
                height=500
            )
        
        # Define button styles
        active_style = {
            "marginRight": "8px",
            "padding": "8px 16px",
            "backgroundColor": "#3b82f6",
            "color": "white",
            "border": "none",
            "borderRadius": "6px",
            "cursor": "pointer",
            "fontSize": "14px",
            "fontWeight": "500"
        }
        
        inactive_style = {
            "marginRight": "8px",
            "padding": "8px 16px",
            "backgroundColor": "#374151",
            "color": "#e5e5e5",
            "border": "1px solid #4b5563",
            "borderRadius": "6px",
            "cursor": "pointer",
            "fontSize": "14px",
            "fontWeight": "500"
        }
        
        # Set button styles based on active filter
        if filter_by == 'Bank':
            bank_style = active_style.copy()
            currency_style = inactive_style.copy()
            label_style = inactive_style.copy()
        elif filter_by == 'Currency':
            bank_style = inactive_style.copy()
            currency_style = active_style.copy()
            label_style = inactive_style.copy()
        else:  # Label
            bank_style = inactive_style.copy()
            currency_style = inactive_style.copy()
            label_style = active_style.copy()
        
        # Remove margin from last button
        if filter_by == 'Label':
            label_style.pop('marginRight', None)
        elif filter_by == 'Currency':
            currency_style.pop('marginRight', None)
        else:
            bank_style.pop('marginRight', None)
        
        return chart, bank_style, currency_style, label_style
        
    except Exception as e:
        print(f"Error updating reinvestment risk chart: {e}")
        error_chart = go.Figure().add_annotation(
            text=f"Error loading reinvestment risk data: {str(e)}",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color='#e5e5e5')
        ).update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#e5e5e5',
            height=500
        )
        
        # Return error chart with default button styles
        inactive_style = {
            "marginRight": "8px",
            "padding": "8px 16px",
            "backgroundColor": "#374151",
            "color": "#e5e5e5",
            "border": "1px solid #4b5563",
            "borderRadius": "6px",
            "cursor": "pointer",
            "fontSize": "14px",
            "fontWeight": "500"
        }
        
        return error_chart, inactive_style, inactive_style, inactive_style


# Callback for Currency Risk chart (USD Index with portfolio USD percentage)
@app.callback(
    Output('currency-risk-chart', 'figure'),
    [Input('risk-btn', 'n_clicks')]
)
def update_currency_risk_chart(risk_clicks):
    """Update Currency Risk chart on Risk Analysis page load"""
    try:
        # Create the USD Index chart with portfolio data
        chart = create_usd_index_chart(data['portfolio'])
        if chart is None:
            return go.Figure().add_annotation(
                text="No USD Index data available for currency risk analysis",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color='#e5e5e5')
            ).update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e5e5e5',
                height=400
            )
        
        return chart
        
    except Exception as e:
        print(f"Error updating Currency Risk chart: {e}")
        return go.Figure().add_annotation(
            text=f"Error loading Currency Risk data: {str(e)}",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color='#e5e5e5')
        ).update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#e5e5e5',
            height=400
        )


# Callback for concentration risk chart
@app.callback(
    Output('concentration-risk-chart', 'figure'),
    [Input('risk-btn', 'n_clicks')]
)
def update_concentration_risk_chart(risk_clicks):
    """Update concentration risk chart on Risk Analysis page load"""
    try:
        print(f"Concentration risk callback triggered with risk_clicks: {risk_clicks}")
        print(f"Data keys: {data.keys() if data else 'No data'}")
        
        if 'portfolio' not in data:
            print("No portfolio data found")
            return go.Figure().add_annotation(
                text="No portfolio data available",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color='#e5e5e5')
            ).update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e5e5e5',
                height=500
            )
        
        # Create the concentration risk chart
        chart = create_concentration_risk_chart(data['portfolio'])
        if chart is None:
            print("Chart creation returned None")
            return go.Figure().add_annotation(
                text="No data available for concentration risk analysis",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color='#e5e5e5')
            ).update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e5e5e5',
                height=500
            )
        
        print("Chart created successfully, returning to dashboard")
        return chart
        
    except Exception as e:
        print(f"Error updating concentration risk chart: {e}")
        import traceback
        traceback.print_exc()
        return go.Figure().add_annotation(
            text=f"Error loading concentration risk data: {str(e)}",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color='#e5e5e5')
        ).update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#e5e5e5',
            height=500
        )


# Callback for credit risk chart
@app.callback(
    Output('credit-risk-chart', 'figure'),
    [Input('risk-btn', 'n_clicks')]
)
def update_credit_risk_chart(risk_clicks):
    """Update credit risk chart on Risk Analysis page load"""
    try:
        # Create the credit risk chart
        chart = create_credit_risk_chart(data['portfolio'])
        if chart is None:
            return go.Figure().add_annotation(
                text="No bonds data available for credit risk analysis",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color='#e5e5e5')
            ).update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e5e5e5',
                height=400
            )
        
        return chart
        
    except Exception as e:
        print(f"Error updating credit risk chart: {e}")
        return go.Figure().add_annotation(
            text=f"Error loading credit risk data: {str(e)}",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color='#e5e5e5')
        ).update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#e5e5e5',
            height=400
        )


# Expose the server for Gunicorn
server = app.server

if __name__ == '__main__':
    # Railway-ready configuration
    debug_mode = os.environ.get('DEBUG', 'False').lower() == 'true'
    port = int(os.environ.get('PORT', 8050))
    
    app.run_server(
        debug=debug_mode,
        host='0.0.0.0',
        port=port
    )
