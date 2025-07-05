# dashboard_application/modes/flow_mode_display_v2_5.py
# EOTS v2.5 - S-GRADE, AUTHORITATIVE FLOW MODE DISPLAY
# Enhanced with SGDHP, IVSDH, and UGCH Heatmaps

import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html
import dash_bootstrap_components as dbc
import numpy as np
from typing import Optional
from dash.development.base_component import Component

from data_models import FinalAnalysisBundleV2_5 # Updated import
from utils.config_manager_v2_5 import ConfigManagerV2_5
from dashboard_application.utils_dashboard_v2_5 import (
    create_empty_figure, apply_dark_theme_template, 
    add_price_line, add_timestamp_annotation, PLOTLY_TEMPLATE
)
import logging

logger = logging.getLogger(__name__)

# --- Helper Functions for Chart Generation ---

def _about_section(text, component_id):
    # Collapsible about section with toggle button
    return html.Div([
        dbc.Button(
            "About",
            id={"type": "about-toggle-btn", "section": component_id},
            color="info",
            size="sm",
            className="mb-2",
            n_clicks=0,
        ),
        dbc.Collapse(
            html.Div(text, className="about-section mb-2"),
            id={"type": "about-collapse", "section": component_id},
            is_open=False,
        ),
        # Note: Wire up a Dash callback to toggle is_open on button click for each card.
        # Example callback signature:
        # @app.callback(
        #   Output({"type": "about-collapse", "section": MATCH}, "is_open"),
        #   Input({"type": "about-toggle-btn", "section": MATCH}, "n_clicks"),
        #   State({"type": "about-collapse", "section": MATCH}, "is_open"),
        # )
        # def toggle_about(n, is_open):
        #     if n:
        #         return not is_open
        #     return is_open
    ], id=f"about-{component_id}", className="about-section-container mb-2")

def _wrap_chart_in_card(chart_component, about_text, component_id):
    return dbc.Card([
        dbc.CardHeader(html.H5(component_id.replace('-', ' ').title())),
        dbc.CardBody([
            _about_section(about_text, component_id),
            chart_component
        ])
    ], className="mb-4", id=f"card-{component_id}")

def _generate_net_value_heatmap(bundle: FinalAnalysisBundleV2_5, config: ConfigManagerV2_5) -> Component:
    """Generates a heatmap of net value pressure by strike and option type."""
    chart_name = "Net Value by Strike"
    # Renamed for clarity, used in empty/error states
    default_fig_height_for_empty = getattr(config, 'flow_mode_settings', None) and getattr(config.flow_mode_settings, 'net_value_heatmap_height', 500) or 500
    about_text = (
        "💡 Net Value by Strike: Visualizes the net value pressure (call/put) across strikes. "
        "Red = net selling/pressure, Green = net buying/support. "
        "Use this to spot where the largest value flows are concentrated and potential support/resistance zones."
    )
    
    try:
        options_data = getattr(bundle.processed_data_bundle, 'options_data_with_metrics', None)
        if options_data is None:
            logger.error(f"[{chart_name}] options_data_with_metrics missing from bundle.")
            return _wrap_chart_in_card(dbc.Alert("Options data unavailable.", color="danger"), about_text, "net-value-heatmap")
        df_chain = pd.DataFrame([c.model_dump() for c in options_data])
        if df_chain.empty or 'value_bs' not in df_chain.columns:
            return _wrap_chart_in_card(dcc.Graph(
                figure=create_empty_figure(chart_name, default_fig_height_for_empty, "Per-contract 'value_bs' data not available."),
                config={'displayModeBar': False, 'displaylogo': False}
            ), about_text, "net-value-heatmap")

        df_plot = df_chain.dropna(subset=['strike', 'opt_kind', 'value_bs'])
        pivot_df = df_plot.pivot_table(index='strike', columns='opt_kind', values='value_bs', aggfunc='sum').fillna(0)
        
        if 'call' not in pivot_df: pivot_df['call'] = 0
        if 'put' not in pivot_df: pivot_df['put'] = 0
        pivot_df = pivot_df[['put', 'call']].sort_index(ascending=True)
        
        if pivot_df.empty:
            return _wrap_chart_in_card(dcc.Graph(
                figure=create_empty_figure(chart_name, default_fig_height_for_empty, "No data to pivot."),
                config={'displayModeBar': False, 'displaylogo': False}
            ), about_text, "net-value-heatmap")

        # Adaptive height calculation
        num_strikes = len(pivot_df.index.unique())
        flow_settings = getattr(config, 'flow_mode_settings', None)
        row_height_px = getattr(flow_settings, 'net_value_heatmap_row_height_px', 20) if flow_settings else 20
        header_footer_px = getattr(flow_settings, 'net_value_heatmap_header_footer_px', 150) if flow_settings else 150
        min_fig_height = getattr(flow_settings, 'net_value_heatmap_min_height', 400) if flow_settings else 400
        max_fig_height = getattr(flow_settings, 'net_value_heatmap_max_height', 1500) if flow_settings else 1500

        adaptive_height = (num_strikes * row_height_px) + header_footer_px
        dynamic_height = max(min_fig_height, min(adaptive_height, max_fig_height))

        fig = go.Figure(data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns.str.capitalize(),
            y=pivot_df.index.astype(str),
            colorscale='RdYlGn',
            zmid=0,
            hoverongaps=False,
            hovertemplate='Strike: %{y}<br>Type: %{x}<br>Net Value: %{z:$,.0f}<extra></extra>',
            ygap=1 # Added vertical gap between heatmap rows
        ))
        
        fig.update_layout(
            title_text=f"<b>{bundle.target_symbol}</b> - {chart_name}",
            template=PLOTLY_TEMPLATE,
            margin=dict(l=80, r=80, t=100, b=80),
            yaxis=dict(
                type='category', # Essential for strike prices as categories
                title="Strike Price",
                tickmode='auto', # Changed from 'linear'
                autorange=True,  # Added
                showgrid=True,   # Added
                gridcolor='rgba(128,128,128,0.2)', # Subtle grid color
                tickfont=dict(size=10), # Optional: consistent tick label size
                domain=[0, 1] # Ensure y-axis uses full vertical space
                # automargin=True was removed as it can conflict with explicit height
            ),
            xaxis=dict(title="Option Type"),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            uirevision='constant',
            height=dynamic_height # Use calculated dynamic height
        )
        
        apply_dark_theme_template(fig)
        
        # CANONICAL FIX: Draw the current price line and annotation separately.
        current_price = getattr(bundle.processed_data_bundle.underlying_data_enriched, 'price', None)
        if current_price is not None:
            y_values_numeric = pivot_df.index.to_series().astype(float)
            if not y_values_numeric.empty: # Check if y_values_numeric is empty
                closest_strike_val = y_values_numeric.iloc[(y_values_numeric - current_price).abs().argmin()]

                # Step 1: Add the horizontal line using the string representation of the closest strike.
                fig.add_hline(
                y=str(closest_strike_val), 
                line_dash="dash", 
                line_color="white", 
                line_width=2
            )
            
            # Step 2: Add the annotation separately for full control and to avoid the TypeError.
            fig.add_annotation(
                x=1.02,
                y=str(closest_strike_val),
                xref="paper",
                yref="y",
                text=f"ATM ≈ {current_price:.2f}",
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font=dict(color="white", size=10),
                bgcolor="rgba(0,0,0,0.5)"
            )
            
        add_timestamp_annotation(fig, bundle.bundle_timestamp)

    except Exception as e:
        logger.error(f"Error creating {chart_name}: {e}", exc_info=True)
        # Use default_fig_height_for_empty for error figure
        fig = create_empty_figure(chart_name, default_fig_height_for_empty, f"Error: {e}")
        # Ensure dynamic_height is not used for the error graph's style
        graph_to_return = dcc.Graph(
            id=f"net-value-heatmap-{bundle.target_symbol}-error", # Different ID for error graph
            figure=fig,
            config={'displayModeBar': False, 'displaylogo': False},
            style={'width': '100%', 'height': f'{default_fig_height_for_empty}px'}
        )
        return _wrap_chart_in_card(graph_to_return, about_text, "net-value-heatmap")

    # If successful, use dynamic_height for the graph style
    success_graph = dcc.Graph(
        id=f"net-value-heatmap-{bundle.target_symbol}",
        figure=fig,
        config={'displayModeBar': False, 'displaylogo': False},
        style={'width': '100%', 'height': f'{dynamic_height}px'}
    )
    return _wrap_chart_in_card(success_graph, about_text, "net-value-heatmap")

def _generate_greek_flow_chart(bundle: FinalAnalysisBundleV2_5, metric: str, title: str, color: str) -> Component:
    """Generic helper to create a bar chart for a net customer Greek flow metric."""
    chart_name = f"{title} by Strike"
    about_text = (
        f"💡 {title} by Strike: Shows net customer {title.lower()} flow at each strike. "
        f"Blue/Green/Orange bars = net {title.lower()} buying/selling. "
        f"Use this to identify where the largest directional or volatility bets are being placed."
    )
    fig = go.Figure()
    
    try:
        strike_data = getattr(bundle.processed_data_bundle, 'strike_level_data_with_metrics', None)
        if strike_data is None:
            logger.error(f"[{chart_name}] strike_level_data_with_metrics missing from bundle.")
            return _wrap_chart_in_card(dbc.Alert(f"Strike data unavailable for {title}.", color="danger"), about_text, f"{title.lower()}-flow-chart")
        df_strike = pd.DataFrame([s.model_dump() for s in strike_data])
        
        logger.info(f"[{chart_name}] Strike data shape: {df_strike.shape}")
        logger.info(f"[{chart_name}] Available columns: {df_strike.columns.tolist()}")
        logger.info(f"[{chart_name}] Looking for metric: {metric}")
        
        if not df_strike.empty:
            if metric in df_strike.columns:
                logger.info(f"[{chart_name}] Metric column sample values: {df_strike[metric].head(10).tolist()}")
                logger.info(f"[{chart_name}] Metric column null count: {df_strike[metric].isnull().sum()}")
                logger.info(f"[{chart_name}] Metric column non-null count: {df_strike[metric].notnull().sum()}")
                
                df_plot = df_strike.dropna(subset=['strike', metric]).sort_values('strike')
                logger.info(f"[{chart_name}] Plot data shape after filtering: {df_plot.shape}")
                
                if df_plot.empty:
                    logger.info(f"[{chart_name}] Trying fallback: filling NaN values with 0")
                    df_fallback = df_strike.dropna(subset=['strike']).copy()
                    df_fallback[metric] = df_fallback[metric].fillna(0)
                    df_plot = df_fallback.sort_values('strike')
                    logger.info(f"[{chart_name}] Fallback data shape: {df_plot.shape}")
                
                if not df_plot.empty:
                    non_zero_count = (df_plot[metric] != 0).sum()
                    logger.info(f"[{chart_name}] Non-zero values: {non_zero_count}")
                    
                    fig.add_trace(go.Bar(
                        x=df_plot['strike'],
                        y=df_plot[metric],
                        name=title,
                        marker_color=color,
                        hovertemplate=f'Strike: %{{x}}<br>{title}: %{{y:.4f}}<extra></extra>'
                    ))
                    fig.update_layout(
                        title_text=f"<b>{bundle.target_symbol}</b> - {chart_name}",
                        height=400,
                        template=PLOTLY_TEMPLATE,
                        showlegend=False,
                        xaxis_title="Strike Price",
                        yaxis_title=title,
                        margin=dict(l=60, r=60, t=80, b=60)
                    )
                    apply_dark_theme_template(fig)
                    current_price = getattr(bundle.processed_data_bundle.underlying_data_enriched, 'price', None)
                    if current_price is not None:
                        add_price_line(fig, current_price)
                    add_timestamp_annotation(fig, bundle.bundle_timestamp)
                else:
                    logger.warning(f"[{chart_name}] No valid data after filtering")
                    fig = create_empty_figure(chart_name, 400, "No valid data after filtering.")
            else:
                logger.warning(f"[{chart_name}] Metric '{metric}' not found in columns: {df_strike.columns.tolist()}")
                fig = _try_calculate_greek_flow_fallback(df_strike, bundle, metric, title, color, chart_name)
        else:
            logger.warning(f"[{chart_name}] Strike data is empty")
            fig = create_empty_figure(chart_name, 400, "Strike data is empty.")
    except Exception as e:
        logger.error(f"Error creating {chart_name}: {e}", exc_info=True)
        fig = create_empty_figure(chart_name, 400, f"Error: {e}")

    return _wrap_chart_in_card(dcc.Graph(
        figure=fig,
        config={'displayModeBar': False, 'displaylogo': False},
        style={'height': '400px', 'width': '100%'}
    ), about_text, f"{title.lower()}-flow-chart")

def _generate_sgdhp_heatmap(bundle: FinalAnalysisBundleV2_5, config: ConfigManagerV2_5) -> Component:
    """Generates SGDHP (Strike-level Gamma Delta Hedging Pressure) heatmap."""
    chart_name = "SGDHP - Strike Gamma Delta Hedging Pressure"
    fig_height = getattr(config, 'flow_mode_settings', None) and getattr(config.flow_mode_settings, 'sgdhp_heatmap_height', 500) or 500
    about_text = (
        "🧮 SGDHP Heatmap: Shows strike-level gamma-delta hedging pressure. "
        "High values = areas where dealers must hedge aggressively. "
        "Use this to spot potential volatility clusters and risk zones."
    )

    try:
        strike_data = getattr(bundle.processed_data_bundle, 'strike_level_data_with_metrics', None)
        options_data = getattr(bundle.processed_data_bundle, 'options_data_with_metrics', None)
        if strike_data is None or options_data is None:
            logger.error(f"[{chart_name}] Required data missing from bundle.")
            return _wrap_chart_in_card(dbc.Alert("SGDHP data unavailable.", color="danger"), about_text, "sgdhp-heatmap")
        df_strike = pd.DataFrame([s.model_dump() for s in strike_data])
        df_options = pd.DataFrame([c.model_dump() for c in options_data])
        
        if df_strike.empty or 'sgdhp_score_strike' not in df_strike.columns:
            if not df_options.empty:
                current_price = getattr(bundle.processed_data_bundle.underlying_data_enriched, 'price', None)
                if current_price is not None:
                    sgdhp_data = _calculate_sgdhp_fallback(df_options, current_price)
                    if sgdhp_data is not None:
                        df_strike = sgdhp_data
                    else:
                        return _wrap_chart_in_card(dcc.Graph(
                            figure=create_empty_figure(chart_name, fig_height, "SGDHP data not available."),
                            config={'displayModeBar': False, 'displaylogo': False}
                        ), about_text, "sgdhp-heatmap")
                else:
                    return _wrap_chart_in_card(dcc.Graph(
                        figure=create_empty_figure(chart_name, fig_height, "Current price not available for SGDHP calculation."),
                        config={'displayModeBar': False, 'displaylogo': False}
                    ), about_text, "sgdhp-heatmap")
            else:
                return _wrap_chart_in_card(dcc.Graph(
                    figure=create_empty_figure(chart_name, fig_height, "No options data available for SGDHP calculation."),
                    config={'displayModeBar': False, 'displaylogo': False}
                ), about_text, "sgdhp-heatmap")

        df_plot = df_strike.dropna(subset=['strike', 'sgdhp_score_strike']).sort_values('strike')
        
        if df_plot.empty:
            return _wrap_chart_in_card(dcc.Graph(
                figure=create_empty_figure(chart_name, fig_height, "No valid SGDHP data to display."),
                config={'displayModeBar': False, 'displaylogo': False}
            ), about_text, "sgdhp-heatmap")

        strikes = df_plot['strike'].values
        sgdhp_values = np.array(df_plot['sgdhp_score_strike'].values)
        z_data = sgdhp_values.reshape(1, -1)
        
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=strikes,
            y=['SGDHP'],
            colorscale='RdYlGn_r',
            zmid=0,
            hoverongaps=False,
            hovertemplate='Strike: %{x}<br>SGDHP Score: %{z:.3f}<extra></extra>',
            colorbar=dict(title="SGDHP Score", x=1.02)
        ))
        
        fig.update_layout(
            title_text=f"<b>{bundle.target_symbol}</b> - {chart_name}",
            height=fig_height,
            template=PLOTLY_TEMPLATE,
            xaxis_title="Strike Price",
            yaxis_title=""
        )
        
        apply_dark_theme_template(fig)
        current_price = getattr(bundle.processed_data_bundle.underlying_data_enriched, 'price', None)
        if current_price is not None:
            add_price_line(fig, current_price)
        add_timestamp_annotation(fig, bundle.bundle_timestamp)

    except Exception as e:
        logger.error(f"Error creating {chart_name}: {e}", exc_info=True)
        fig = create_empty_figure(chart_name, fig_height, f"Error: {e}")

    return _wrap_chart_in_card(dcc.Graph(
        figure=fig,
        config={'displayModeBar': False, 'displaylogo': False},
        style={'height': f'{fig_height}px', 'width': '100%'}
    ), about_text, "sgdhp-heatmap")

def _generate_ivsdh_heatmap(bundle: FinalAnalysisBundleV2_5, config: ConfigManagerV2_5) -> Component:
    """Generates IVSDH (Implied Volatility Surface Delta Hedging) heatmap."""
    chart_name = "IVSDH - IV Surface Delta Hedging"
    fig_height = getattr(config, 'flow_mode_settings', None) and getattr(config.flow_mode_settings, 'ivsdh_heatmap_height', 500) or 500
    about_text = (
        "🌈 IVSDH Heatmap: Visualizes the implied volatility surface delta hedging. "
        "Shows how volatility and delta interact across strikes and expiries. "
        "Use this to identify where volatility is most sensitive to price moves."
    )

    try:
        und_data = getattr(bundle.processed_data_bundle, 'underlying_data_enriched', None)
        options_data = getattr(bundle.processed_data_bundle, 'options_data_with_metrics', None)
        if und_data is None or options_data is None:
            logger.error(f"[{chart_name}] Required data missing from bundle.")
            return _wrap_chart_in_card(dbc.Alert("IVSDH data unavailable.", color="danger"), about_text, "ivsdh-heatmap")
        ivsdh_surface = getattr(und_data, 'ivsdh_surface_data', None)
        df_options = pd.DataFrame([c.model_dump() for c in options_data])
        
        if ivsdh_surface is None:
            if not df_options.empty:
                ivsdh_data = _calculate_ivsdh_fallback(df_options)
                if ivsdh_data is not None:
                    ivsdh_surface = ivsdh_data
                else:
                    return _wrap_chart_in_card(dcc.Graph(
                        figure=create_empty_figure(chart_name, fig_height, "IVSDH surface data not available."),
                        config={'displayModeBar': False, 'displaylogo': False},
                        style={'height': f'{fig_height}px', 'width': '100%'}
                    ), about_text, "ivsdh-heatmap")
            else:
                return _wrap_chart_in_card(dcc.Graph(
                    figure=create_empty_figure(chart_name, fig_height, "No options data available for IVSDH calculation."),
                    config={'displayModeBar': False, 'displaylogo': False},
                    style={'height': f'{fig_height}px', 'width': '100%'}
                ), about_text, "ivsdh-heatmap")

        if not isinstance(ivsdh_surface, pd.DataFrame):
            return _wrap_chart_in_card(dcc.Graph(
                figure=create_empty_figure(chart_name, fig_height, "IVSDH surface data format not supported."),
                config={'displayModeBar': False, 'displaylogo': False},
                style={'height': f'{fig_height}px', 'width': '100%'}
            ), about_text, "ivsdh-heatmap")

        if 'strike' in ivsdh_surface.columns and 'dte_calc' in ivsdh_surface.columns and 'ivsdh_score' in ivsdh_surface.columns:
            pivot_df = ivsdh_surface.pivot_table(
                index='dte_calc', 
                columns='strike', 
                values='ivsdh_score', 
                aggfunc='mean'
            ).fillna(0)
        else:
            return _wrap_chart_in_card(dcc.Graph(
                figure=create_empty_figure(chart_name, fig_height, "Required columns not found in IVSDH surface data."),
                config={'displayModeBar': False, 'displaylogo': False},
                style={'height': f'{fig_height}px', 'width': '100%'}
            ), about_text, "ivsdh-heatmap")

        if pivot_df.empty:
            return _wrap_chart_in_card(dcc.Graph(
                figure=create_empty_figure(chart_name, fig_height, "No valid IVSDH surface data to display."),
                config={'displayModeBar': False, 'displaylogo': False},
                style={'height': f'{fig_height}px', 'width': '100%'}
            ), about_text, "ivsdh-heatmap")

        fig = go.Figure(data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale='RdYlBu',
            zmid=0,
            hoverongaps=False,
            hovertemplate='Strike: %{x}<br>DTE: %{y}<br>IVSDH Score: %{z:.3f}<extra></extra>',
            colorbar=dict(title="IVSDH Score", x=1.02)
        ))
        
        fig.update_layout(
            title_text=f"<b>{bundle.target_symbol}</b> - {chart_name}",
            height=fig_height,
            template=PLOTLY_TEMPLATE,
            xaxis_title="Strike Price",
            yaxis_title="Days to Expiration"
        )
        
        apply_dark_theme_template(fig)
        current_price = getattr(bundle.processed_data_bundle.underlying_data_enriched, 'price', None)
        if current_price is not None:
            add_price_line(fig, current_price)
        add_timestamp_annotation(fig, bundle.bundle_timestamp)

    except Exception as e:
        logger.error(f"Error creating {chart_name}: {e}", exc_info=True)
        fig = create_empty_figure(chart_name, fig_height, f"Error: {e}")

    return _wrap_chart_in_card(dcc.Graph(
        figure=fig,
        config={'displayModeBar': False, 'displaylogo': False},
        style={'height': f'{fig_height}px', 'width': '100%'}
    ), about_text, "ivsdh-heatmap")

def _generate_ugch_heatmap(bundle: FinalAnalysisBundleV2_5, config: ConfigManagerV2_5) -> Component:
    """Generates UGCH (Unified Gamma Charm Hedging) heatmap."""
    chart_name = "UGCH - Unified Gamma Charm Hedging"
    fig_height = getattr(config, 'flow_mode_settings', None) and getattr(config.flow_mode_settings, 'ugch_heatmap_height', 500) or 500
    about_text = (
        "🟣 UGCH Heatmap: Shows unified gamma-charm hedging intensity. "
        "Highlights where both gamma and charm are driving dealer hedging. "
        "Use this to spot high-risk, high-volatility strike zones."
    )

    try:
        strike_data = getattr(bundle.processed_data_bundle, 'strike_level_data_with_metrics', None)
        options_data = getattr(bundle.processed_data_bundle, 'options_data_with_metrics', None)
        if strike_data is None or options_data is None:
            logger.error(f"[{chart_name}] Required data missing from bundle.")
            return _wrap_chart_in_card(dbc.Alert("UGCH data unavailable.", color="danger"), about_text, "ugch-heatmap")
        df_strike = pd.DataFrame([s.model_dump() for s in strike_data])
        df_options = pd.DataFrame([c.model_dump() for c in options_data])
        
        if df_strike.empty or 'ugch_score_strike' not in df_strike.columns:
            if not df_options.empty:
                current_price = getattr(bundle.processed_data_bundle.underlying_data_enriched, 'price', None)
                if current_price is not None:
                    ugch_data = _calculate_ugch_fallback(df_options, current_price)
                    if ugch_data is not None:
                        df_strike = ugch_data
                    else:
                        return _wrap_chart_in_card(dcc.Graph(
                            figure=create_empty_figure(chart_name, fig_height, "UGCH data not available."),
                            config={'displayModeBar': False, 'displaylogo': False},
                            style={'height': f'{fig_height}px', 'width': '100%'}
                        ), about_text, "ugch-heatmap")
                else:
                    return _wrap_chart_in_card(dcc.Graph(
                        figure=create_empty_figure(chart_name, fig_height, "Current price not available for UGCH calculation."),
                        config={'displayModeBar': False, 'displaylogo': False},
                        style={'height': f'{fig_height}px', 'width': '100%'}
                    ), about_text, "ugch-heatmap")
            else:
                return _wrap_chart_in_card(dcc.Graph(
                    figure=create_empty_figure(chart_name, fig_height, "No options data available for UGCH calculation."),
                    config={'displayModeBar': False, 'displaylogo': False},
                    style={'height': f'{fig_height}px', 'width': '100%'}
                ), about_text, "ugch-heatmap")

        df_plot = df_strike.dropna(subset=['strike', 'ugch_score_strike']).sort_values('strike')
        
        if df_plot.empty:
            return _wrap_chart_in_card(dcc.Graph(
                figure=create_empty_figure(chart_name, fig_height, "No valid UGCH data to display."),
                config={'displayModeBar': False, 'displaylogo': False},
                style={'height': f'{fig_height}px', 'width': '100%'}
            ), about_text, "ugch-heatmap")

        strikes = df_plot['strike'].values
        ugch_values = df_plot['ugch_score_strike'].values
        intensity_bands = ['Low', 'Medium', 'High']
        z_data = np.tile(np.array(ugch_values), (len(intensity_bands), 1))
        
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=strikes,
            y=intensity_bands,
            colorscale='Viridis',
            zmid=0,
            hoverongaps=False,
            hovertemplate='Strike: %{x}<br>Intensity: %{y}<br>UGCH Score: %{z:.3f}<extra></extra>',
            colorbar=dict(title="UGCH Score", x=1.02)
        ))
        
        fig.update_layout(
            title_text=f"<b>{bundle.target_symbol}</b> - {chart_name}",
            height=fig_height,
            template=PLOTLY_TEMPLATE,
            xaxis_title="Strike Price",
            yaxis_title="Hedging Intensity"
        )
        
        apply_dark_theme_template(fig)
        current_price = getattr(bundle.processed_data_bundle.underlying_data_enriched, 'price', None)
        if current_price is not None:
            add_price_line(fig, current_price)
        add_timestamp_annotation(fig, bundle.bundle_timestamp)

    except Exception as e:
        logger.error(f"Error creating {chart_name}: {e}", exc_info=True)
        fig = create_empty_figure(chart_name, fig_height, f"Error: {e}")

    return _wrap_chart_in_card(dcc.Graph(
        figure=fig,
        config={'displayModeBar': False, 'displaylogo': False},
        style={'height': f'{fig_height}px', 'width': '100%'}
    ), about_text, "ugch-heatmap")

# --- Fallback Calculation Functions ---

def _calculate_sgdhp_fallback(df_options: pd.DataFrame, underlying_price: float) -> Optional[pd.DataFrame]:
    """Fallback calculation for SGDHP when not available in processed data."""
    try:
        if 'strike' not in df_options.columns or 'gxoi' not in df_options.columns or 'dxoi' not in df_options.columns:
            return None
            
        strike_groups = df_options.groupby('strike').agg({
            'gxoi': 'sum',
            'dxoi': 'sum',
            'open_interest': 'sum'
        }).reset_index()
        
        strike_groups['distance_from_atm'] = abs(strike_groups['strike'] - underlying_price) / underlying_price
        
        strike_groups['sgdhp_score_strike'] = (
            strike_groups['gxoi'] * strike_groups['dxoi'] * 
            np.exp(-strike_groups['distance_from_atm'] * 2)
        ) / (strike_groups['open_interest'] + 1)
        
        return strike_groups[['strike', 'sgdhp_score_strike']]
        
    except Exception as e:
        logger.error(f"Error in SGDHP fallback calculation: {e}")
        return None

def _calculate_ivsdh_fallback(df_options: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Fallback calculation for IVSDH when not available in processed data."""
    try:
        required_cols = ['strike', 'dte_calc', 'iv', 'delta_contract', 'vega_contract']
        if not all(col in df_options.columns for col in required_cols):
            return None
            
        df_calc = df_options.copy()
        df_calc['ivsdh_score'] = (
            df_calc['iv'] * abs(df_calc['delta_contract']) * df_calc['vega_contract']
        ) / (df_calc['dte_calc'] + 1)
        
        return df_calc[['strike', 'dte_calc', 'ivsdh_score']]
        
    except Exception as e:
        logger.error(f"Error in IVSDH fallback calculation: {e}")
        return None

def _calculate_ugch_fallback(df_options: pd.DataFrame, underlying_price: float) -> Optional[pd.DataFrame]:
    """Fallback calculation for UGCH when not available in processed data."""
    try:
        required_cols = ['strike', 'gamma_contract', 'charm_contract']
        if not all(col in df_options.columns for col in required_cols):
            return None
            
        strike_groups = df_options.groupby('strike').agg({
            'gamma_contract': 'sum',
            'charm_contract': 'sum'
        }).reset_index()
        
        strike_groups['ugch_score_strike'] = (
            abs(strike_groups['gamma_contract']) * abs(strike_groups['charm_contract'])
        )
        
        return strike_groups[['strike', 'ugch_score_strike']]
        
    except Exception as e:
        logger.error(f"Error in UGCH fallback calculation: {e}")
        return None


def _try_calculate_greek_flow_fallback(df_strike: pd.DataFrame, bundle: FinalAnalysisBundleV2_5, metric: str, title: str, color: str, chart_name: str) -> go.Figure:
    """Try to calculate Greek flow metrics from available data when they're missing."""
    try:
        logger.info(f"[{chart_name}] Attempting fallback calculation for {metric}")
        
        metric_calculations = {
            'net_customer_delta_flow': lambda df: _calculate_net_flow(df, 'delta_contract', 'customer'),
            'net_customer_gamma_flow': lambda df: _calculate_net_flow(df, 'gamma_contract', 'customer'),
            'net_customer_vega_flow': lambda df: _calculate_net_flow(df, 'vega_contract', 'customer'),
            'net_customer_theta_flow': lambda df: _calculate_net_flow(df, 'theta_contract', 'customer'),
            'net_customer_charm_flow': lambda df: _calculate_net_flow(df, 'charm_contract', 'customer')
        }
        
        if metric in metric_calculations:
            df_calc = metric_calculations[metric](df_strike)
            
            if not df_calc.empty and metric in df_calc.columns:
                logger.info(f"[{chart_name}] Successfully calculated {metric} using fallback")
                
                fig = go.Figure()
                df_plot = df_calc.dropna(subset=['strike', metric]).sort_values('strike')
                
                if not df_plot.empty:
                    fig.add_trace(go.Bar(
                        x=df_plot['strike'],
                        y=df_plot[metric],
                        name=title,
                        marker_color=color,
                        hovertemplate=f'Strike: %{{x}}<br>{title}: %{{y:.4f}}<extra></extra>'
                    ))
                    fig.update_layout(
                        title_text=f"<b>{bundle.target_symbol}</b> - {chart_name} (Calculated)",
                        height=400,
                        template=PLOTLY_TEMPLATE,
                        showlegend=False,
                        xaxis_title="Strike Price",
                        yaxis_title=title,
                        margin=dict(l=60, r=60, t=80, b=60)
                    )
                    apply_dark_theme_template(fig)
                    current_price = getattr(bundle.processed_data_bundle.underlying_data_enriched, 'price', None)
                    if current_price is not None:
                        add_price_line(fig, current_price)
                    add_timestamp_annotation(fig, bundle.bundle_timestamp)
                    return fig
                else:
                    logger.warning(f"[{chart_name}] Calculated data is empty after filtering")
            else:
                logger.warning(f"[{chart_name}] Failed to calculate {metric} using fallback")
        else:
            logger.warning(f"[{chart_name}] No fallback calculation available for {metric}")
        
        return create_empty_figure(chart_name, 400, f"Metric '{metric}' not available and cannot be calculated.")
        
    except Exception as e:
        logger.error(f"Error in Greek flow fallback calculation for {metric}: {e}")
        return create_empty_figure(chart_name, 400, f"Error calculating {metric}: {e}")


def _calculate_net_flow(df_strike: pd.DataFrame, greek_col: str, flow_type: str) -> pd.DataFrame:
    """Calculate net flow for a specific Greek and flow type."""
    try:
        required_cols = ['strike', greek_col, 'open_interest', 'volume']
        available_cols = [col for col in required_cols if col in df_strike.columns]
        
        if len(available_cols) < 2:
            logger.warning(f"Insufficient columns for {greek_col} flow calculation. Available: {available_cols}")
            return pd.DataFrame()
        
        df_calc = df_strike[available_cols].copy()
        
        if greek_col in df_calc.columns:
            if 'volume' in df_calc.columns:
                df_calc[f'net_{flow_type}_{greek_col.replace("_contract", "")}_flow'] = df_calc[greek_col] * df_calc['volume']
            elif 'open_interest' in df_calc.columns:
                df_calc[f'net_{flow_type}_{greek_col.replace("_contract", "")}_flow'] = df_calc[greek_col] * df_calc['open_interest'] * 0.1
            else:
                df_calc[f'net_{flow_type}_{greek_col.replace("_contract", "")}_flow'] = df_calc[greek_col]
            
            return df_calc
        else:
            logger.warning(f"Greek column {greek_col} not found in data")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error calculating net flow for {greek_col}: {e}")
        return pd.DataFrame()

# --- Main Layout Function ---

def create_layout(bundle: FinalAnalysisBundleV2_5, config: ConfigManagerV2_5) -> html.Div:
    """
    Creates the complete layout for the "Flow Breakdown" mode.
    Enforces Pydantic validation at the UI boundary.
    """
    if not isinstance(bundle, FinalAnalysisBundleV2_5):
        raise ValueError("Input bundle is not a FinalAnalysisBundleV2_5 Pydantic model.")
    if not bundle or not bundle.processed_data_bundle:
        return html.Div([
            dbc.Alert("Flow data is not available. Cannot render Flow Mode.", color="danger")
        ])
    if not hasattr(bundle.processed_data_bundle, 'underlying_data_enriched') or not bundle.processed_data_bundle.underlying_data_enriched:
        return html.Div([
            dbc.Alert("Underlying data is not available. Cannot render Flow Mode.", color="danger")
        ])

    try:
        net_value_heatmap = _generate_net_value_heatmap(bundle, config)
        delta_flow_chart = _generate_greek_flow_chart(bundle, 'net_cust_delta_flow_at_strike', 'Delta Flow', '#4A9EFF')
        gamma_flow_chart = _generate_greek_flow_chart(bundle, 'net_cust_gamma_flow_at_strike', 'Gamma Flow', '#10B981')
        vega_flow_chart = _generate_greek_flow_chart(bundle, 'net_cust_vega_flow_at_strike', 'Vega Flow', '#FFB84A')
        
        sgdhp_heatmap = _generate_sgdhp_heatmap(bundle, config)
        ivsdh_heatmap = _generate_ivsdh_heatmap(bundle, config)
        ugch_heatmap = _generate_ugch_heatmap(bundle, config)
        
        layout = html.Div([
            dbc.Row([
                dbc.Col(net_value_heatmap, width=12)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col(delta_flow_chart, width=4),
                dbc.Col(gamma_flow_chart, width=4),
                dbc.Col(vega_flow_chart, width=4)
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col(sgdhp_heatmap, width=12)
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col(ivsdh_heatmap, width=6),
                dbc.Col(ugch_heatmap, width=6)
            ], className="mb-3")
        ])
        
        return layout
        
    except Exception as e:
        logger.error(f"Error creating flow mode layout: {e}", exc_info=True)
        return html.Div([
            dbc.Alert(
                f"Error loading flow mode: {str(e)}",
                color="danger",
                className="m-3"
            )
        ])