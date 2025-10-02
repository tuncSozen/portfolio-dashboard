"""
Data utilities for portfolio dashboard.
Contains data loading, processing, and chart creation functions.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys

# Handle distutils compatibility for Python 3.12+
if sys.version_info >= (3, 12):
    import setuptools
    sys.modules['distutils'] = setuptools
    sys.modules['distutils.util'] = setuptools.util
    sys.modules['distutils.version'] = setuptools.version
    sys.modules['distutils.errors'] = setuptools.errors


def load_data():
    """Load and process data from Excel files"""
    try:
        # Load portfolio data
        portfolio_df = pd.read_excel('portfolio.xlsx')
        print("Portfolio data shape:", portfolio_df.shape)
        print("Portfolio data columns:", portfolio_df.columns.tolist())
        
        # Filter by banks
        bbva_df = portfolio_df[portfolio_df['Bank'] == 'BBVA'].copy()
        ubs_df = portfolio_df[portfolio_df['Bank'] == 'UBS'].copy()
        lo_df = portfolio_df[portfolio_df['Bank'] == 'LO'].copy()
        
        print(f"BBVA assets: {len(bbva_df)}")
        print(f"UBS assets: {len(ubs_df)}")
        print(f"LO assets: {len(lo_df)}")
        
        # Load TWRR data
        twrr_bbva = pd.read_excel('TWRR.xlsx', sheet_name='BBVA')
        twrr_ubs = pd.read_excel('TWRR.xlsx', sheet_name='UBS')
        twrr_lo = pd.read_excel('TWRR.xlsx', sheet_name='LO')
        twrr_master = pd.read_excel('TWRR.xlsx', sheet_name='Master')
        
        # Load evolutionPL data
        evolution_df = pd.read_excel('evolutionPL.xlsx')
        
        # Create dataframes for each bank using P&L as index
        evolution_bbva = evolution_df[['P&L', 'BBVA']].copy()
        evolution_bbva = evolution_bbva.set_index('P&L')
        evolution_bbva.columns = ['Value']
        
        evolution_ubs = evolution_df[['P&L', 'UBS']].copy()
        evolution_ubs = evolution_ubs.set_index('P&L')
        evolution_ubs.columns = ['Value']
        
        evolution_lo = evolution_df[['P&L', 'LO']].copy()
        evolution_lo = evolution_lo.set_index('P&L')
        evolution_lo.columns = ['Value']
        
        evolution_master = evolution_df[['P&L', 'Master']].copy()
        evolution_master = evolution_master.set_index('P&L')
        evolution_master.columns = ['Value']
        
        return {
            'portfolio': {
                'bbva': bbva_df,
                'ubs': ubs_df,
                'lo': lo_df
            },
            'twrr': {
                'bbva': twrr_bbva,
                'ubs': twrr_ubs,
                'lo': twrr_lo,
                'master': twrr_master
            },
            'evolution': {
                'bbva': evolution_bbva,
                'ubs': evolution_ubs,
                'lo': evolution_lo,
                'master': evolution_master
            }
        }
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def load_benchmarks():
    """Load and process benchmarks data from Excel file"""
    try:
        # Load benchmarks data
        benchmarks_df = pd.read_excel('benchmarks.xlsx')
        print("Benchmarks data shape:", benchmarks_df.shape)
        print("Benchmarks data columns:", benchmarks_df.columns.tolist())
        
        # Define ticker mapping for better readability
        ticker_mapping = {
            '^GSPC': 'SP500',
            '^IXIC': 'Nasdaq', 
            '^TNX': 'US10Y',
            'LQD': 'LQD Bond Index',
            'BIGPX': 'Invesco MSCI World ETF',
            'MXWO.L': 'World US Dollar Index'
        }
        
        # Rename columns if they match the ticker mapping
        for old_name, new_name in ticker_mapping.items():
            if old_name in benchmarks_df.columns:
                benchmarks_df = benchmarks_df.rename(columns={old_name: new_name})
        
        print("Benchmarks data loaded successfully")
        return benchmarks_df
        
    except Exception as e:
        print(f"Error loading benchmarks data: {e}")
        return None


def create_benchmark_ytd(benchmarks_df):
    """Create benchmark YTD dataframes showing monthly returns and cumulative returns for the year of the final date"""
    if benchmarks_df is None or benchmarks_df.empty:
        print("No benchmarks data available")
        return None, None
    
    try:
        # Ensure we have a Date column and it's in datetime format
        if 'Date' not in benchmarks_df.columns:
            print("No Date column found in benchmarks data")
            return None, None
        
        # Convert Date column to datetime
        benchmarks_df = benchmarks_df.copy()
        benchmarks_df['Date'] = pd.to_datetime(benchmarks_df['Date'])
        
        # Get the final date and extract the year
        final_date = benchmarks_df['Date'].max()
        final_year = final_date.year
        previous_year = final_year - 1
        
        print(f"Creating YTD data for year: {final_year}")
        
        # Filter data for the final year AND previous December (to calculate January returns)
        ytd_data = benchmarks_df[
            (benchmarks_df['Date'].dt.year == final_year) | 
            ((benchmarks_df['Date'].dt.year == previous_year) & (benchmarks_df['Date'].dt.month == 12))
        ].copy()
        
        if ytd_data.empty:
            print(f"No data found for year {final_year}")
            return None, None
        
        # Sort by date to ensure proper order
        ytd_data = ytd_data.sort_values('Date')
        
        # Set Date as index for resampling
        ytd_data = ytd_data.set_index('Date')
        
        # Get all numeric columns (excluding Date)
        numeric_columns = ytd_data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Resample to monthly frequency using end-of-month values
        monthly_data = ytd_data.resample('M').last()
        
        # Reset index to get Date column back
        monthly_data = monthly_data.reset_index()
        
        print(f"Monthly data after resampling: {len(monthly_data)} months")
        print(f"Date range: {monthly_data['Date'].min()} to {monthly_data['Date'].max()}")
        print(f"Months included: {monthly_data['Date'].dt.strftime('%Y-%m').tolist()}")
        
        # Create monthly returns dataframe (keep all months for calculation)
        monthly_returns = pd.DataFrame()
        monthly_returns['Date'] = monthly_data['Date']
        
        # Create cumulative returns dataframe
        cumulative_returns = pd.DataFrame()
        cumulative_returns['Date'] = monthly_data['Date']
        
        for col in numeric_columns:
            if col != 'Date':
                # Calculate percentage change from previous month (specify fill_method=None to avoid deprecation warning)
                monthly_return = monthly_data[col].pct_change(fill_method=None) * 100
                monthly_returns[col] = monthly_return
                
                # Calculate cumulative YTD return from monthly returns
                cumulative_return = ((1 + monthly_return / 100).cumprod() - 1) * 100
                cumulative_returns[col] = cumulative_return
        
        # Now filter to only include months from the target year (exclude previous December)
        monthly_returns = monthly_returns[monthly_returns['Date'].dt.year == final_year].copy()
        cumulative_returns = cumulative_returns[cumulative_returns['Date'].dt.year == final_year].copy()
        
        print(f"After filtering to target year: {len(monthly_returns)} months")
        print(f"Filtered months: {monthly_returns['Date'].dt.strftime('%Y-%m').tolist()}")
        
        # Round to 2 decimal places
        numeric_cols_monthly = monthly_returns.select_dtypes(include=[np.number]).columns
        monthly_returns[numeric_cols_monthly] = monthly_returns[numeric_cols_monthly].round(2)
        
        numeric_cols_cumulative = cumulative_returns.select_dtypes(include=[np.number]).columns
        cumulative_returns[numeric_cols_cumulative] = cumulative_returns[numeric_cols_cumulative].round(2)
        
        print(f"Benchmark YTD data created with {len(monthly_returns)} months of data")
        print(f"Monthly returns columns: {[col for col in monthly_returns.columns if col != 'Date']}")
        print(f"Cumulative returns columns: {[col for col in cumulative_returns.columns if col != 'Date']}")
        
        return monthly_returns, cumulative_returns
        
    except Exception as e:
        print(f"Error creating benchmark YTD data: {e}")
        return None, None


def process_data(data):
    """Process dataframes to format TWRR and evolution data"""
    if data is None:
        return None
    
    try:
        # Process TWRR data - multiply TWRR and Cum. TWRR columns by 100
        for bank in ['bbva', 'ubs', 'lo', 'master']:
            if 'TWRR' in data['twrr'][bank].columns:
                data['twrr'][bank]['TWRR'] = data['twrr'][bank]['TWRR'] * 100
            if 'Cum. TWRR' in data['twrr'][bank].columns:
                data['twrr'][bank]['Cum. TWRR'] = data['twrr'][bank]['Cum. TWRR'] * 100
        
        # Process evolution data - convert values to integers
        for bank in ['bbva', 'ubs', 'lo', 'master']:
            if 'Value' in data['evolution'][bank].columns:
                data['evolution'][bank]['Value'] = data['evolution'][bank]['Value'].astype(int)
        
        print("Data processing completed successfully")
        return data
        
    except Exception as e:
        print(f"Error processing data: {e}")
        return data


def get_color_mapping(category_type):
    """Get color mapping for different category types"""
    if category_type == 'Type':
        return {
            'Fixed Income': '#1f77b4',  # Blue
            'Equity': '#ff7f0e',        # Orange
            'Cash': '#2ca02c',          # Green
            'Other': '#d62728'          # Red
        }
    elif category_type == 'Subtype':
        return {
            'Bonds': '#1f77b4',         # Blue
            'Shares': '#ff7f0e',        # Orange
            'Cash': '#2ca02c',          # Green
            'Equity Fund': '#9467bd',   # Purple
            'Private Equity': '#8c564b', # Brown
            'Crypto': '#e377c2',        # Pink
            'Precious Metals': '#7f7f7f', # Gray
            'Private Credit': '#bcbd22', # Olive
            'Bond Fund': '#17becf'      # Cyan
        }
    elif category_type == 'Currency':
        return {
            'USD': '#1f77b4',          # Blue
            'EUR': '#ff7f0e',          # Orange
            'GBP': '#2ca02c',          # Green
            'CHF': '#d62728'           # Red
        }
    elif category_type == 'Label':
        return {
            'IG': '#1f77b4',           # Blue
            'Tech': '#ff7f0e',         # Orange
            'TUR': '#2ca02c',          # Green
            'Other': '#d62728',        # Red
            'HY': '#9467bd',           # Purple
            'Dividend': '#8c564b',     # Brown
            'Index': '#e377c2',        # Pink
            'PE': '#7f7f7f',           # Gray
            'Crypto': '#bcbd22',       # Olive
            'Precious Metals': '#17becf', # Cyan
            'PC': '#aec7e8'            # Light Blue
        }
    return {}


def create_pie_chart(df, bank_name, groupby_value):
    """Create pie chart for asset allocation"""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#e5e5e5',
            height=320,
            annotations=[dict(text="No data available", x=0.5, y=0.5, showarrow=False, font=dict(size=16, color='#e5e5e5'))]
        )
        return fig
    
    # Group by selected column and sum values
    grouped = df.groupby(groupby_value)['Value_USD'].sum().reset_index()
    
    # Get color mapping for this category type
    color_mapping = get_color_mapping(groupby_value)
    
    # Create pie chart with consistent colors
    fig = px.pie(
        grouped, 
        values='Value_USD', 
        names=groupby_value, 
        hole=0.6,
        color=groupby_value,
        color_discrete_map=color_mapping
    ).update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#e5e5e5',
        height=320,
        showlegend=False
    ).update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(width=2, color='#0a0a0a')))
    
    return fig


def create_twrr_chart(twrr_data, bank_name, color):
    """Create TWRR performance chart"""
    return go.Figure().add_trace(
        go.Scatter(
            x=twrr_data['Date'],
            y=twrr_data['Cum. TWRR'],
            mode='lines+markers',
            name='Cumulative TWRR',
            line=dict(color=color, width=3),
            marker=dict(size=6)
        )
    ).add_trace(
        go.Scatter(
            x=[twrr_data['Date'].iloc[-1]],
            y=[twrr_data['Cum. TWRR'].iloc[-1]],
            mode='markers+text',
            marker=dict(size=12, color=color, symbol='diamond'),
            text=[f"{twrr_data['Cum. TWRR'].iloc[-1]:.2f}%"],
            textposition='top center',
            textfont=dict(size=12, color=color),
            showlegend=False,
            name='Final Value'
        )
    ).update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#e5e5e5',
        height=300,
        xaxis_title="Date",
        yaxis_title="Cumulative TWRR (%)",
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40)
    )


def create_bank_comparison_chart(twrr_data):
    """Create bank performance comparison chart"""
    fig = go.Figure()
    
    colors = {'bbva': '#9ca3af', 'ubs': '#3b82f6', 'lo': '#93c5fd', 'master': '#f59e0b'}
    bank_names = {'bbva': 'BBVA', 'ubs': 'UBS', 'lo': 'LO', 'master': 'Master'}
    
    # Add traces for individual banks with reduced opacity
    for bank in ['bbva', 'ubs', 'lo']:
        fig.add_trace(go.Scatter(
            x=twrr_data[bank]['Date'],
            y=twrr_data[bank]['Cum. TWRR'],
            mode='lines+markers',
            name=bank_names[bank],
            line=dict(color=colors[bank], width=2),
            marker=dict(size=4),
            opacity=0.6
        ))
    
    # Add Master trace with full opacity
    master_data = twrr_data['master']
    master_last_value = master_data['Cum. TWRR'].iloc[-1]
    
    fig.add_trace(go.Scatter(
        x=master_data['Date'],
        y=master_data['Cum. TWRR'],
        mode='lines+markers',
        name=bank_names['master'],
        line=dict(color=colors['master'], width=3),
        marker=dict(size=6)
    ))
    
    # Add final value text in same style as Banks page TWRR charts
    fig.add_trace(go.Scatter(
        x=[master_data['Date'].iloc[-1]],
        y=[master_data['Cum. TWRR'].iloc[-1]],
        mode='markers+text',
        marker=dict(size=12, color=colors['master'], symbol='diamond'),
        text=[f"{master_last_value:.2f}%"],
        textposition='top center',
        textfont=dict(size=12, color=colors['master']),
        showlegend=False,
        name='Master Final Value'
    ))
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#e5e5e5',
        height=300,
        xaxis_title="Date",
        yaxis_title="Cumulative TWRR (%)",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=40, r=40, t=40, b=60)
    )
    
    return fig


def create_portfolio_evolution_chart(evolution_data):
    """Create portfolio evolution chart"""
    fig = go.Figure()
    
    # Calculate total portfolio value over time
    dates = evolution_data['bbva'].index.tolist()
    total_values = []
    
    for date in dates:
        total_value = sum(evolution_data[bank].loc[date, 'Value'] for bank in ['bbva', 'ubs', 'lo', 'master'])
        total_values.append(total_value)
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=total_values,
        mode='lines+markers',
        name='Total Portfolio Value',
        line=dict(color='#3b82f6', width=3),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#e5e5e5',
        height=300,
        xaxis_title="Period",
        yaxis_title="Portfolio Value ($)",
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig


def create_portfolio_waterfall_chart(evolution_data):
    """Create portfolio evolution waterfall chart"""
    # Get the evolution data for Master (overall portfolio)
    master_data = evolution_data['master']
    
    # Get the periods and values
    periods = master_data.index.tolist()
    values = master_data['Value'].tolist()
    
    # Calculate the changes between periods
    changes = []
    for i in range(len(values)):
        if i == 0:
            changes.append(0)  # Starting value
        else:
            changes.append(values[i] - values[i-1])
    
    # Create waterfall chart data
    fig = go.Figure()
    
    # Add starting value
    fig.add_trace(go.Bar(
        x=[periods[0]],
        y=[values[0]],
        name='Starting Value',
        marker_color='#3b82f6',
        text=[f"${values[0]:,.0f}"],
        textposition='outside',
        showlegend=False
    ))
    
    # Add changes (positive and negative)
    for i in range(1, len(periods)):
        change = changes[i]
        color = '#22c55e' if change >= 0 else '#ef4444'
        
        fig.add_trace(go.Bar(
            x=[periods[i]],
            y=[abs(change)],
            name=f'Change to {periods[i]}',
            marker_color=color,
            text=[f"{'+' if change >= 0 else ''}${change:,.0f}"],
            textposition='outside',
            showlegend=False
        ))
    
    # Add final value
    fig.add_trace(go.Bar(
        x=[periods[-1]],
        y=[values[-1]],
        name='Final Value',
        marker_color='#f59e0b',
        text=[f"${values[-1]:,.0f}"],
        textposition='outside',
        showlegend=False
    ))
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#e5e5e5',
        height=300,
        xaxis_title="Period",
        yaxis_title="Portfolio Value ($)",
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40),
        barmode='group'
    )
    
    return fig


def get_top_holdings(bank_data, top_n=10):
    """Get top holdings across all banks"""
    all_holdings = []
    
    for bank in ['bbva', 'ubs', 'lo']:
        bank_name = bank.upper()
        holdings = bank_data[bank][bank_data[bank]['Type'] != 'Cash'].copy()
        holdings['Bank'] = bank_name
        all_holdings.append(holdings[['Issuer', 'Bank', 'Value_USD']])
    
    combined_holdings = pd.concat(all_holdings, ignore_index=True)
    top_holdings = combined_holdings.nlargest(top_n, 'Value_USD')
    
    return top_holdings.to_dict('records')


def process_portfolio_data(df):
    """Process portfolio data with percentage conversion and formatting"""
    processed_df = df.copy()
    
    # Convert percentage values to percentage format (multiply by 100)
    if 'Change_pct' in processed_df.columns:
        processed_df['Change_pct'] = (processed_df['Change_pct'] * 100).round(2)
    if 'Change_pct_USD' in processed_df.columns:
        processed_df['Change_pct_USD'] = (processed_df['Change_pct_USD'] * 100).round(2)
    
    # Format Value_USD
    if 'Value_USD' in processed_df.columns:
        processed_df['Value_USD'] = processed_df['Value_USD'].round(2)
    
    # Convert Maturity column to datetime and format as date only
    if 'Maturity' in processed_df.columns:
        processed_df['Maturity'] = pd.to_datetime(processed_df['Maturity'], errors='coerce')
        processed_df['Maturity'] = processed_df['Maturity'].dt.date
    
    # Format numeric columns - remove decimal points for portfolio holdings
    numeric_columns = ['Change_pct', 'Change_pct_USD', 'Value_USD']
    for col in numeric_columns:
        if col in processed_df.columns:
            # Handle NaN values before converting to integer
            processed_df[col] = processed_df[col].fillna(0).round(0).astype(int)
    
    return processed_df


def get_portfolio_holdings_data(bank_data):
    """Get processed portfolio holdings data for the holdings page"""
    # Combine all portfolio data from all banks
    all_holdings = []
    for bank in ['bbva', 'ubs', 'lo']:
        bank_name = bank.upper()
        holdings = bank_data[bank].copy()
        holdings['Bank'] = bank_name
        all_holdings.append(holdings)
    
    combined_holdings = pd.concat(all_holdings, ignore_index=True)
    
    # Filter to only show the specified columns
    filtered_columns = ['Type', 'Subtype', 'Issuer', 'Currency', 'Bank', 'Maturity', 'Change_pct', 'Change_pct_USD', 'Label', 'Value_USD']
    filtered_data = combined_holdings[filtered_columns].copy()
    
    # Process the data with percentage conversion and formatting
    return process_portfolio_data(filtered_data)


def get_bank_portfolio_data(bank_data, bank_name):
    """Get processed portfolio data for a specific bank"""
    # Get the bank data and filter out Cash assets
    bank_df = bank_data[bank_name][bank_data[bank_name]['Type'] != 'Cash'].copy()
    
    # Process the data with percentage conversion and formatting
    processed_df = process_portfolio_data(bank_df)
    
    return processed_df[['Issuer', 'Subtype', 'Change_pct', 'Value_USD']]


def get_evolution_data(evolution_data, bank_name):
    """Get processed evolution data for a specific bank"""
    # Get the evolution data and reset index
    evolution_df = evolution_data[bank_name].reset_index().copy()
    
    # Remove decimal points from Value column, handling NaN values
    if 'Value' in evolution_df.columns:
        evolution_df['Value'] = evolution_df['Value'].fillna(0).round(0).astype(int)
    
    return evolution_df


def create_currency_risk_chart(portfolio_data):
    """
    Create a donut chart showing portfolio breakdown by currency
    
    Parameters:
    portfolio_data (dict): Dictionary containing portfolio data for each bank
    
    Returns:
    plotly.graph_objects.Figure: Donut chart figure
    """
    try:
        # Combine all portfolio data from all banks
        all_holdings = []
        for bank in ['bbva', 'ubs', 'lo']:
            bank_name = bank.upper()
            holdings = portfolio_data[bank].copy()
            holdings['Bank'] = bank_name
            all_holdings.append(holdings)
        
        combined_holdings = pd.concat(all_holdings, ignore_index=True)
        
        if combined_holdings.empty:
            print("No portfolio data found")
            return None
        
        # Group by currency and sum values
        currency_data = combined_holdings.groupby('Currency')['Value_USD'].sum().reset_index()
        
        if currency_data.empty:
            print("No currency data found")
            return None
        
        # Get color mapping for currencies
        color_mapping = get_color_mapping('Currency')
        
        # Create donut chart with same style as portfolio allocation
        fig = px.pie(
            currency_data, 
            values='Value_USD', 
            names='Currency', 
            hole=0.6,
            color='Currency',
            color_discrete_map=color_mapping
        ).update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#e5e5e5',
            height=400,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.1,
                xanchor="center",
                x=0.5
            )
        ).update_traces(
            textposition='inside', 
            textinfo='percent+label', 
            marker=dict(line=dict(width=2, color='#0a0a0a'))
        )
        
        print(f"Currency risk chart created with {len(currency_data)} currencies")
        print(f"Currencies: {currency_data['Currency'].tolist()}")
        print(f"Total portfolio value: ${currency_data['Value_USD'].sum():,.0f}")
        
        return fig
        
    except Exception as e:
        print(f"Error creating currency risk chart: {e}")
        return None


def create_bonds_stacked_chart(portfolio_data):
    """
    Create a stacked bar chart of bonds quantity by year
    
    Parameters:
    portfolio_data (dict): Dictionary containing portfolio data for each bank
    
    Returns:
    plotly.graph_objects.Figure: Stacked bar chart figure
    """
    try:
        # Combine all portfolio data from all banks
        all_holdings = []
        for bank in ['bbva', 'ubs', 'lo']:
            bank_name = bank.upper()
            holdings = portfolio_data[bank].copy()
            holdings['Bank'] = bank_name
            all_holdings.append(holdings)
        
        combined_holdings = pd.concat(all_holdings, ignore_index=True)
        
        # Filter for bonds only
        bonds_data = combined_holdings[combined_holdings['Type'] == 'Fixed Income'].copy()
        
        if bonds_data.empty:
            print("No bonds data found")
            return None
        
        # Convert Maturity to datetime and extract year
        bonds_data['Maturity'] = pd.to_datetime(bonds_data['Maturity'], errors='coerce')
        bonds_data['Maturity_Year'] = bonds_data['Maturity'].dt.year
        
        # Remove rows where we couldn't extract year
        bonds_data = bonds_data.dropna(subset=['Maturity_Year'])
        
        if bonds_data.empty:
            print("No valid maturity dates found for bonds")
            return None
        
        # Group by year and currency, sum the Value_USD
        yearly_bonds = bonds_data.groupby(['Maturity_Year', 'Currency'])['Value_USD'].sum().reset_index()
        
        # Pivot to get currencies as columns
        pivot_data = yearly_bonds.pivot(index='Maturity_Year', columns='Currency', values='Value_USD').fillna(0)
        
        # Create stacked bar chart
        fig = go.Figure()
        
        # Define colors for each currency
        currency_colors = {
            'USD': '#3b82f6',      # Blue
            'EUR': '#10b981',      # Green  
            'GBP': '#f59e0b',      # Orange
            'CHF': '#ef4444',      # Red
            'CAD': '#8b5cf6',      # Purple
            'JPY': '#06b6d4'       # Cyan
        }
        
        # Add traces for each currency
        for currency in pivot_data.columns:
            fig.add_trace(go.Bar(
                name=currency,
                x=pivot_data.index,
                y=pivot_data[currency],
                marker_color=currency_colors.get(currency, '#6b7280'),
                hovertemplate=f'<b>{currency}</b><br>Year: %{{x}}<br>Value: $%{{y:,.0f}}<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Bonds Portfolio by Maturity Year',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16, 'color': '#e5e5e5'}
            },
            xaxis_title="Maturity Year",
            yaxis_title="Value (USD)",
            barmode='stack',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#e5e5e5',
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=60, r=40, t=80, b=60)
        )
        
        # Format y-axis to show currency
        fig.update_yaxes(tickformat='$,.0f')
        
        print(f"Bonds stacked chart created with {len(pivot_data)} years of data")
        print(f"Years covered: {sorted(pivot_data.index.tolist())}")
        print(f"Total bonds value: ${pivot_data.sum().sum():,.0f}")
        
        return fig
        
    except Exception as e:
        print(f"Error creating bonds stacked chart: {e}")
        return None


def create_benchmark_summary_chart(cumulative_returns, twrr_data):
    """Create bar chart for Summary Page Benchmark Container showing final cumulative returns"""
    if cumulative_returns is None or cumulative_returns.empty:
        print("No cumulative returns data available")
        return None
    
    if twrr_data is None or not isinstance(twrr_data, dict) or 'master' not in twrr_data:
        print("No TWRR data available")
        return None
    
    try:
        # Get the final cumulative returns for each benchmark
        final_returns = {}
        
        # Define benchmarks to exclude
        excluded_benchmarks = ['US10Y', 'USD Index', 'Nasdaq']
        
        # Get benchmark returns (last row of cumulative returns)
        for col in cumulative_returns.columns:
            if col != 'Date' and col not in excluded_benchmarks:
                final_value = cumulative_returns[col].iloc[-1]
                if not pd.isna(final_value):
                    final_returns[col] = final_value
        
        # Get portfolio TWRR (last value from master TWRR data)
        if 'master' in twrr_data and 'Cum. TWRR' in twrr_data['master'].columns:
            portfolio_twr = twrr_data['master']['Cum. TWRR'].iloc[-1]
            if not pd.isna(portfolio_twr):
                final_returns['Portfolio'] = portfolio_twr
        
        if not final_returns:
            print("No valid return data found")
            return None
        
        # Create dataframe and sort by value (highest to lowest)
        summary_df = pd.DataFrame(list(final_returns.items()), columns=['Benchmark', 'Return'])
        summary_df = summary_df.sort_values('Return', ascending=False).reset_index(drop=True)
        
        # Define colors for different benchmarks
        colors = []
        for benchmark in summary_df['Benchmark']:
            if benchmark == 'Portfolio':
                colors.append('#FF8C00')  # Bloomberg orange for portfolio
            else:
                colors.append('#3b82f6')  # Blue for all other benchmarks
        
        # Create scatter plot
        fig = go.Figure(data=[
            go.Scatter(
                x=summary_df['Benchmark'],
                y=summary_df['Return'],
                mode='markers+text',
                marker=dict(
                    color=colors,
                    size=20,
                    line=dict(width=2, color='#0a0a0a')
                ),
                text=[f"{val:.2f}%" for val in summary_df['Return']],
                textposition='top center',
                textfont=dict(size=12, color='#e5e5e5'),
                hovertemplate='<b>%{x}</b><br>Return: %{y:.2f}%<extra></extra>'
            )
        ])
        
        # Update layout
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#e5e5e5',
            height=400,
            xaxis_title="Benchmark",
            yaxis_title="Cumulative Return (%)",
            title={
                'text': 'YTD Performance Comparison',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16, 'color': '#e5e5e5'}
            },
            xaxis=dict(
                tickangle=45,
                tickfont=dict(size=12),
                showgrid=True,
                gridcolor='#374151',
                gridwidth=1
            ),
            yaxis=dict(
                tickformat='.1f',
                tickfont=dict(size=12),
                showgrid=True,
                gridcolor='#374151',
                gridwidth=1
            ),
            margin=dict(l=60, r=40, t=60, b=80),
            showlegend=False
        )
        
        print(f"Benchmark summary chart created with {len(summary_df)} benchmarks")
        return fig
        
    except Exception as e:
        print(f"Error creating benchmark summary chart: {e}")
        return None


def create_credit_risk_chart(portfolio_data):
    """Create a donut chart showing bonds breakdown by credit rating (Label)"""
    try:
        # Combine all holdings from all banks
        all_holdings = []
        for bank in ['bbva', 'ubs', 'lo']:
            holdings = portfolio_data[bank].copy()
            holdings['Bank'] = bank.upper()
            all_holdings.append(holdings)
        
        combined_holdings = pd.concat(all_holdings, ignore_index=True)
        
        # Filter for bonds only
        bonds_data = combined_holdings[combined_holdings['Type'] == 'Fixed Income'].copy()
        
        if bonds_data.empty:
            return None
        
        # Group by Label (credit rating) and sum values
        credit_data = bonds_data.groupby('Label')['Value_USD'].sum().reset_index()
        credit_data = credit_data.sort_values('Value_USD', ascending=False)
        
        # Calculate percentages
        total_value = credit_data['Value_USD'].sum()
        credit_data['Percentage'] = (credit_data['Value_USD'] / total_value * 100).round(1)
        
        # Create donut chart
        fig = go.Figure(data=[go.Pie(
            labels=credit_data['Label'],
            values=credit_data['Value_USD'],
            hole=0.6,
            textinfo='label+percent',
            textposition='inside',
            textfont=dict(size=12, color='white'),
            marker=dict(
                colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
                line=dict(color='#000000', width=2)
            ),
            hovertemplate='<b>%{label}</b><br>Value: $%{value:,.0f}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title={'text': 'Bonds Credit Rating Breakdown', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 16, 'color': '#e5e5e5'}},
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#e5e5e5',
            height=400,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.1,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=0, r=0, t=60, b=60)
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating credit risk chart: {e}")
        return None


def create_reinvestment_risk_chart(portfolio_data, filter_by='Bank'):
    """Create a stacked bar chart of bonds by maturity year with interactive filtering"""
    try:
        # Combine all holdings from all banks
        all_holdings = []
        for bank in ['bbva', 'ubs', 'lo']:
            holdings = portfolio_data[bank].copy()
            holdings['Bank'] = bank.upper()
            all_holdings.append(holdings)
        
        combined_holdings = pd.concat(all_holdings, ignore_index=True)
        
        # Filter for bonds only
        bonds_data = combined_holdings[combined_holdings['Type'] == 'Fixed Income'].copy()
        
        if bonds_data.empty:
            return None
        
        # Convert maturity date and extract year
        bonds_data['Maturity'] = pd.to_datetime(bonds_data['Maturity'], errors='coerce')
        bonds_data['Maturity_Year'] = bonds_data['Maturity'].dt.year
        bonds_data = bonds_data.dropna(subset=['Maturity_Year'])
        
        if bonds_data.empty:
            return None
        
        # Group by maturity year and the selected filter column
        if filter_by == 'Bank':
            group_cols = ['Maturity_Year', 'Bank']
            color_col = 'Bank'
        elif filter_by == 'Currency':
            group_cols = ['Maturity_Year', 'Currency']
            color_col = 'Currency'
        elif filter_by == 'Label':
            group_cols = ['Maturity_Year', 'Label']
            color_col = 'Label'
        else:
            group_cols = ['Maturity_Year', 'Bank']
            color_col = 'Bank'
        
        yearly_bonds = bonds_data.groupby(group_cols)['Value_USD'].sum().reset_index()
        
        # Create pivot table for stacked bar chart
        pivot_data = yearly_bonds.pivot(index='Maturity_Year', columns=color_col, values='Value_USD').fillna(0)
        
        # Create the chart
        fig = go.Figure()
        
        # Define color mapping based on filter type
        if filter_by == 'Bank':
            colors = {'BBVA': '#3b82f6', 'UBS': '#10b981', 'LO': '#f59e0b'}
        elif filter_by == 'Currency':
            colors = {'USD': '#3b82f6', 'EUR': '#10b981', 'GBP': '#f59e0b', 
                     'CHF': '#ef4444', 'CAD': '#8b5cf6', 'JPY': '#06b6d4'}
        else:  # Label - create dynamic color mapping
            # Get unique labels from the data
            unique_labels = pivot_data.columns.tolist()
            print(f"Unique labels found: {unique_labels}")
            
            # Define a comprehensive color palette
            color_palette = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', 
                           '#06b6d4', '#f97316', '#84cc16', '#ec4899', '#6366f1', 
                           '#14b8a6', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4']
            
            # Create color mapping for each unique label
            colors = {}
            for i, label in enumerate(unique_labels):
                colors[label] = color_palette[i % len(color_palette)]
            
            print(f"Color mapping for labels: {colors}")
        
        for col in pivot_data.columns:
            fig.add_trace(go.Bar(
                name=col,
                x=pivot_data.index,
                y=pivot_data[col],
                marker_color=colors.get(col, '#6b7280'),
                hovertemplate=f'<b>{col}</b><br>Year: %{{x}}<br>Value: $%{{y:,.0f}}<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title={'text': f'Bonds Portfolio by Maturity Year ({filter_by})', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 16, 'color': '#e5e5e5'}},
            xaxis_title="Maturity Year",
            yaxis_title="Value (USD)",
            barmode='stack',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#e5e5e5',
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=60, r=40, t=80, b=60)
        )
        
        fig.update_yaxes(tickformat='$,.0f')
        
        return fig
        
    except Exception as e:
        print(f"Error creating reinvestment risk chart: {e}")
        return None


def create_concentration_risk_chart(portfolio_data):
    """Create a treemap chart showing portfolio concentration by Label (excluding Fixed Income and Cash)"""
    try:
        print("Starting concentration risk chart creation...")
        
        # Combine all holdings from all banks
        all_holdings = []
        for bank in ['bbva', 'ubs', 'lo']:
            if bank in portfolio_data:
                holdings = portfolio_data[bank].copy()
                holdings['Bank'] = bank.upper()
                all_holdings.append(holdings)
                print(f"Added {len(holdings)} holdings from {bank}")
            else:
                print(f"No data found for bank: {bank}")
        
        if not all_holdings:
            print("No holdings data found")
            return None
            
        combined_holdings = pd.concat(all_holdings, ignore_index=True)
        print(f"Combined holdings: {len(combined_holdings)} rows")
        print(f"Available columns: {combined_holdings.columns.tolist()}")
        print(f"Available types: {combined_holdings['Type'].unique()}")
        
        # Filter out Fixed Income and Cash
        filtered_data = combined_holdings[
            (combined_holdings['Type'] != 'Fixed Income') & 
            (combined_holdings['Type'] != 'Cash')
        ].copy()
        
        print(f"After filtering (excluding Fixed Income and Cash): {len(filtered_data)} rows")
        print(f"Remaining types: {filtered_data['Type'].unique()}")
        
        if filtered_data.empty:
            print("No data after filtering")
            return None
        
        # Check if Label column exists and has data
        if 'Label' not in filtered_data.columns:
            print("Label column not found")
            return None
            
        # Remove rows with missing labels
        filtered_data = filtered_data.dropna(subset=['Label'])
        print(f"After removing missing labels: {len(filtered_data)} rows")
        print(f"Unique labels: {filtered_data['Label'].unique()}")
        
        if filtered_data.empty:
            print("No data after removing missing labels")
            return None
        
        # Group by Label and sum values
        label_data = filtered_data.groupby('Label')['Value_USD'].sum().reset_index()
        label_data = label_data.sort_values('Value_USD', ascending=False)
        
        print(f"Label data: {len(label_data)} unique labels")
        print(f"Label data:\n{label_data}")
        
        if label_data.empty:
            print("No label data after grouping")
            return None
        
        # Create treemap chart using plotly express
        print(f"Creating treemap with data:\n{label_data}")
        
        try:
            fig = px.treemap(
                label_data,
                path=['Label'],
                values='Value_USD',
                color='Value_USD',
                color_continuous_scale='Viridis',
                title='Portfolio Concentration by Label'
            )
            
            print("Treemap created, updating traces...")
            
            # Update traces for better styling
            fig.update_traces(
                textinfo="label+value+percent parent",
                textfont=dict(size=14, color='white'),
                hovertemplate='<b>%{label}</b><br>Value: $%{value:,.0f}<br>Percentage: %{percentParent:.1f}%<extra></extra>'
            )
            
        except Exception as treemap_error:
            print(f"Treemap creation failed: {treemap_error}")
            print("Creating fallback bar chart...")
            
            # Fallback to bar chart
            fig = px.bar(
                label_data,
                x='Label',
                y='Value_USD',
                color='Value_USD',
                color_continuous_scale='Viridis',
                title='Portfolio Concentration by Label (Bar Chart)'
            )
            
            fig.update_traces(
                hovertemplate='<b>%{x}</b><br>Value: $%{y:,.0f}<extra></extra>'
            )
        
        # Update layout
        fig.update_layout(
            title={'text': 'Portfolio Concentration by Label', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 16, 'color': '#e5e5e5'}},
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#e5e5e5',
            height=500,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        print("Concentration risk chart created successfully")
        return fig
        
    except Exception as e:
        print(f"Error creating concentration risk chart: {e}")
        import traceback
        traceback.print_exc()
        return None
