# dashboard_application/modes/structure_mode_display_v2_5.py
# EOTS v2.5 - S-GRADE, AUTHORITATIVE STRUCTURE MODE DISPLAY

import logging
import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
from dash.development.base_component import Component
import numpy as np
import math

from dashboard_application.utils_dashboard_v2_5 import create_empty_figure, add_price_line, PLOTLY_TEMPLATE, add_bottom_right_timestamp_annotation, apply_dark_theme_template
from data_models import (
    FinalAnalysisBundleV2_5,
    ProcessedStrikeLevelMetricsV2_5,
    ProcessedUnderlyingAggregatesV2_5,
    KeyLevelsDataV2_5
)
from utils.config_manager_v2_5 import ConfigManagerV2_5

logger = logging.getLogger(__name__)

# --- Helper Function for Chart Generation ---

def _log_chart_diagnostics(chart_name, config, price_range_percent, current_price, strikes, x_min, x_max, extra_msg=None):
    logger.info(f"[ChartDiag] {chart_name} | config_id={id(config)} | price_range_percent={price_range_percent} | current_price={current_price}")
    logger.info(f"[ChartDiag] {chart_name} | strikes count={len(strikes)} | min_strike={min(strikes) if len(strikes) else None} | max_strike={max(strikes) if len(strikes) else None}")
    logger.info(f"[ChartDiag] {chart_name} | x_min={x_min} | x_max={x_max}")
    if extra_msg:
        logger.warning(f"[ChartDiag] {chart_name} | {extra_msg}")

def _generate_a_mspi_profile_chart(bundle, config):
    """
    Generates the primary visualization for Structure Mode: the Adaptive MSPI profile.
    This chart shows the synthesized structural pressure across different strikes.
    Now filters strikes to the canonical price_range_percent window around ATM, as set in the control panel.
    """
    chart_name = "Adaptive MSPI Profile"
    struct_settings = getattr(config, 'structure_mode_settings', None)
    fig_height = getattr(struct_settings, 'mspi_chart_height', 600) if struct_settings else 600
    try:
        strike_data = bundle.processed_data_bundle.strike_level_data_with_metrics
        if not strike_data:
            return dcc.Graph(
                figure=create_empty_figure(chart_name, fig_height, "Strike level data not available."),
                config={
                    'displayModeBar': False,
                    'displaylogo': False
                }
            )
        df_strike = pd.DataFrame([s.model_dump() for s in strike_data])
        metric_to_plot = 'a_dag_strike'
        if df_strike.empty or metric_to_plot not in df_strike.columns:
            return dcc.Graph(
                figure=create_empty_figure(chart_name, fig_height, f"'{metric_to_plot}' data not found."),
                config={
                    'displayModeBar': False,
                    'displaylogo': False
                }
            )
        price_range_percent = getattr(config, 'price_range_percent', 20) or 20
        current_price = bundle.processed_data_bundle.underlying_data_enriched.price
        price_range_percent = price_range_percent if price_range_percent is not None else 20

        # CRITICAL FIX: Fail-fast for invalid current_price instead of using fake data
        if current_price is None or math.isnan(current_price):
            logger.error(f"[{chart_name}] Invalid current price - cannot generate chart")
            return dcc.Graph(
                figure=create_empty_figure(chart_name, fig_height, "Invalid current price data."),
                config={'displayModeBar': False, 'displaylogo': False}
            )
        df_plot = df_strike.dropna(subset=['strike', metric_to_plot]).copy()
        df_plot['strike'] = pd.to_numeric(df_plot['strike'], errors='coerce')
        df_plot = df_plot.dropna(subset=['strike'])
        df_plot = df_plot.sort_values('strike')
        price_min = current_price * (1 - price_range_percent / 100.0)
        price_max = current_price * (1 + price_range_percent / 100.0)
        df_plot = df_plot[(df_plot['strike'] >= price_min) & (df_plot['strike'] <= price_max)]
        # Defensive: ensure strikes is always defined as a float array
        strikes_clean = df_plot['strike'].dropna()
        strikes = np.array([float(s) for s in strikes_clean if s is not None and not math.isnan(float(s))])
        strikes_valid = [s for s in strikes if isinstance(s, float) and not math.isnan(s)]
        # Final check: guarantee current_price is a valid float for add_vline
        if current_price is None or (isinstance(current_price, float) and math.isnan(current_price)):
            current_price = 0.0
        # Only add ATM line if within min/max strike
        if len(strikes_valid) > 0 and min(strikes_valid) <= current_price <= max(strikes_valid):
            fig = go.Figure(data=[go.Bar(
                x=df_plot['strike'],
                y=df_plot[metric_to_plot],
                name='A-DAG',
                marker_color=['#d62728' if x < 0 else '#2ca02c' for x in df_plot[metric_to_plot]],
                hovertemplate='Strike: %{x}<br>A-DAG: %{y:,.2f}<extra></extra>'
            )])
            fig.add_vline(x=current_price, line_dash="dash", line_color="red", annotation_text="ATM", annotation_position="top right")
        else:
            logger.warning(f"[ChartDiag] ATM/current price {current_price} is outside plotted strikes [{min(strikes_valid) if len(strikes_valid) else 'N/A'}, {max(strikes_valid) if len(strikes_valid) else 'N/A'}]; skipping ATM line.")
            fig = go.Figure(data=[go.Bar(
                x=df_plot['strike'],
                y=df_plot[metric_to_plot],
                name='A-DAG',
                marker_color=['#d62728' if x < 0 else '#2ca02c' for x in df_plot[metric_to_plot]],
                hovertemplate='Strike: %{x}<br>A-DAG: %{y:,.2f}<extra></extra>'
            )])
        # Let Plotly auto-scale axes
        fig.update_layout(
            title_text=f"<b>{bundle.target_symbol}</b> - {chart_name} (using A-DAG)",
            height=fig_height,
            plot_bgcolor="#181A1B",
            paper_bgcolor="#181A1B",
            font=dict(color="#E1E1E1"),
            showlegend=False,
            barmode='relative',
            xaxis_autorange=True,
            yaxis_autorange=True,
            margin=dict(l=60, r=30, t=60, b=60)
        )
        apply_dark_theme_template(fig)
        add_bottom_right_timestamp_annotation(fig, bundle.bundle_timestamp)
    except Exception as e:
        logger.error(f"Error creating {chart_name}: {e}", exc_info=True)
        fig = create_empty_figure(chart_name, fig_height, f"Error: {e}")
    return dcc.Graph(
        figure=fig,
        config={
            'displayModeBar': False,
            'displaylogo': False
        }
    )

def _about_section(text, component_id):
    return [
        dbc.Button(
            "ℹ️ About", 
            id={"type": "about-toggle", "index": component_id}, 
            color="link", 
            size="sm", 
            className="p-0 text-elite-secondary mb-2 elite-focus-visible", 
            style={'font-size': '0.75em'}
        ),
        dbc.Collapse(
            html.Small(text, className="text-elite-secondary d-block mb-2", style={'font-size': '0.75em'}),
            id={"type": "about-collapse", "index": component_id},
            is_open=False
        )
    ]

def _wrap_chart_in_card(chart_component, about_text, component_id):
    """Wraps a chart component in a card with about section, following main dashboard pattern."""
    about_button = dbc.Button(
        "ℹ️ About", 
        id={"type": "about-toggle", "index": component_id}, 
        color="link", 
        size="sm", 
        className="p-0 text-elite-secondary mb-2 elite-focus-visible",
        style={'font-size': '0.75em'}
    )
    
    about_collapse = dbc.Collapse(
        html.Small(about_text, className="text-elite-secondary d-block mb-2", style={'font-size': '0.75em'}),
        id={"type": "about-collapse", "index": component_id},
        is_open=False
    )
    
    return dbc.Card(
        dbc.CardBody([
            about_button,
            about_collapse,
            chart_component
        ], className="elite-card-body"),
        className="elite-card fade-in-up"
    )

# --- Chart Stubs ---
def _generate_amspi_heatmap(bundle, config):
    chart_name = "A-MSPI Heatmap (SGDHP Score)"
    struct_settings = getattr(config, 'structure_mode_settings', None)
    fig_height = getattr(struct_settings, 'sgdhp_chart_height', 400) if struct_settings else 400
    about_text = (
        "🎯 A-MSPI Heatmap (Adaptive Market Structure Pressure Index): This is your PRIMARY structural view showing where dealers are positioned to provide support or resistance. "
        "GREEN bars = SUPPORT zones where dealers will likely buy to hedge (price floors). "
        "RED bars = RESISTANCE zones where dealers will likely sell to hedge (price ceilings). "
        "INTENSITY shows strength - darker colors mean stronger levels. "
        "CURRENT PRICE LINE shows where we are relative to these zones. "
        "💡 TRADING INSIGHT: Look for price to 'bounce' off strong green support or 'reject' at red resistance. "
        "When price breaks through a strong level with volume, expect acceleration in that direction as dealers are forced to re-hedge. "
        "Multiple green levels below = bullish structure. Multiple red levels above = bearish structure. "
        "This chart updates in real-time as options flow changes dealer positioning!"
    )
    
    try:
        strike_data = bundle.processed_data_bundle.strike_level_data_with_metrics
        if not strike_data:
            logger.warning("[A-MSPI Heatmap] No strike data available.")
            chart_component = html.Div([
                dbc.Alert("No strike level data available.", color="warning", className="mb-2"),
                dcc.Graph(
                    figure=create_empty_figure(chart_name, fig_height, "Strike level data not available."),
                    config={
                        'displayModeBar': False,
                        'displaylogo': False
                    }
                )
            ])
            return _wrap_chart_in_card(chart_component, about_text, "amspi-heatmap")
        df_strike = pd.DataFrame([m.model_dump() for m in strike_data])
        metric_to_plot = 'sgdhp_score_strike'
        if df_strike.empty or metric_to_plot not in df_strike.columns:
            logger.warning(f"[A-MSPI Heatmap] '{metric_to_plot}' data not found in DataFrame columns: {df_strike.columns}")
            chart_component = html.Div([
                dbc.Alert(f"'{metric_to_plot}' data not found.", color="warning", className="mb-2"),
                dcc.Graph(
                    figure=create_empty_figure(chart_name, fig_height, f"'{metric_to_plot}' data not found."),
                    config={
                        'displayModeBar': False,
                        'displaylogo': False
                    }
                )
            ])
            return _wrap_chart_in_card(chart_component, about_text, "amspi-heatmap")
        df_plot = df_strike.dropna(subset=['strike', metric_to_plot]).sort_values('strike')
        colors = df_plot[metric_to_plot].apply(lambda x: '#2ca02c' if x > 0 else '#d62728')
        fig = go.Figure(data=[go.Bar(
            x=df_plot['strike'],
            y=df_plot[metric_to_plot],
            marker_color=colors,
            name='SGDHP',
            hovertemplate='Strike: %{x}<br>SGDHP Score: %{y:,.2f}<extra></extra>'
        )])
        current_price = bundle.processed_data_bundle.underlying_data_enriched.price
        price_range_percent = getattr(config, 'price_range_percent', 20) if getattr(config, 'price_range_percent', None) is not None else 20

        # CRITICAL FIX: Fail-fast for invalid current_price instead of using fake data
        if current_price is None or math.isnan(current_price):
            logger.error(f"[{chart_name}] Invalid current price - cannot generate chart")
            return dcc.Graph(
                figure=create_empty_figure(chart_name, fig_height, "Invalid current price data."),
                config={'displayModeBar': False, 'displaylogo': False}
            )
        fig.update_layout(
            title_text=f"<b>{bundle.target_symbol}</b> - {chart_name}",
            height=fig_height,
            template=PLOTLY_TEMPLATE,
            xaxis_title="Strike Price",
            yaxis_title="SGDHP Score (Support/Resistance Intensity)",
            showlegend=False,
            xaxis={'type': 'linear', 'automargin': True},
            margin=dict(l=60, r=30, t=60, b=60)
        )
        # Apply custom dark theme styling
        apply_dark_theme_template(fig)
        add_price_line(fig, current_price, orientation='vertical', width=2, color='white')
        add_bottom_right_timestamp_annotation(fig, bundle.bundle_timestamp)
        
        chart_component = dcc.Graph(
            figure=fig,
            config={
                'displayModeBar': False,
                'displaylogo': False
            }
        )
        return _wrap_chart_in_card(chart_component, about_text, "amspi-heatmap")
    except Exception as e:
        logger.error(f"Error creating {chart_name}: {e}", exc_info=True)
        chart_component = html.Div([
            dbc.Alert(f"Error: {e}", color="danger", className="mb-2"),
            dcc.Graph(
                figure=create_empty_figure(chart_name, fig_height, f"Error: {e}"),
                config={
                    'displayModeBar': False,
                    'displaylogo': False
                }
            )
        ])
        return _wrap_chart_in_card(chart_component, about_text, "amspi-heatmap")

def _generate_esdag_charts(bundle, config):
    chart_name = "E-SDAG Methodology Components"
    struct_settings = getattr(config, 'structure_mode_settings', None)
    fig_height = getattr(struct_settings, 'esdag_chart_height', 350) if struct_settings else 350
    
    about_text = (
        "📊 E-SDAG Components (Enhanced Structural Delta-Adjusted Gamma): These 4 lines show DIFFERENT METHODS of calculating dealer hedging pressure, each capturing unique market dynamics. "
        "MULTIPLICATIVE (Blue) = Traditional gamma * delta interaction - shows basic hedging needs. "
        "DIRECTIONAL (Orange) = Factors in whether dealers are long/short gamma - shows hedging direction. "
        "WEIGHTED (Green) = Adjusts for volume and open interest - shows 'real' vs theoretical pressure. "
        "VOL FLOW (Red) = Incorporates volatility and flow - shows dynamic hedging adjustments. "
        "💡 TRADING INSIGHT: When ALL 4 lines AGREE (all positive or all negative) at a strike = VERY HIGH CONVICTION level. "
        "DIVERGENCE between lines = uncertainty, potential for volatility. "
        "Watch for lines CROSSING zero = hedging flip points where dealer behavior changes. "
        "The line with the LARGEST magnitude often leads price action. "
        "Use the legend to toggle lines and focus on specific methodologies!"
    )
    
    try:
        strike_data = bundle.processed_data_bundle.strike_level_data_with_metrics
        if not strike_data:
            logger.warning("[E-SDAG] No strike data available.")
            chart_component = html.Div([
                dbc.Alert("No strike level data available.", color="warning", className="mb-2"),
                dcc.Graph(
                    figure=create_empty_figure(chart_name, fig_height, "Strike level data not available."),
                    config={
                        'displayModeBar': False,
                        'displaylogo': False
                    }
                )
            ])
            return _wrap_chart_in_card(chart_component, about_text, "esdag-charts")
        df_strike = pd.DataFrame([m.model_dump() for m in strike_data])
        components = [
            ("Multiplicative", 'e_sdag_mult_strike', '#1f77b4'),
            ("Directional", 'e_sdag_dir_strike', '#ff7f0e'),
            ("Weighted", 'e_sdag_w_strike', '#2ca02c'),
            ("Vol Flow", 'e_sdag_vf_strike', '#d62728')
        ]
        fig = go.Figure()
        for label, col, color in components:
            if col in df_strike.columns:
                fig.add_trace(go.Scatter(
                    x=df_strike['strike'],
                    y=df_strike[col],
                    mode='lines+markers',
                    name=label,
                    line=dict(color=color),
                    hovertemplate=f'Strike: %{{x}}<br>{label}: %{{y:,.2f}}<extra></extra>'
                ))
        current_price = bundle.processed_data_bundle.underlying_data_enriched.price
        price_range_percent = getattr(config, 'price_range_percent', 20) if getattr(config, 'price_range_percent', None) is not None else 20

        # CRITICAL FIX: Fail-fast for invalid current_price instead of using fake data
        if current_price is None or math.isnan(current_price):
            logger.error(f"[{chart_name}] Invalid current price - cannot generate chart")
            return dcc.Graph(
                figure=create_empty_figure(chart_name, fig_height, "Invalid current price data."),
                config={'displayModeBar': False, 'displaylogo': False}
            )
        fig.update_layout(
            title_text=f"<b>{bundle.target_symbol}</b> - {chart_name}",
            height=fig_height,
            template=PLOTLY_TEMPLATE,
            xaxis_title="Strike Price",
            yaxis_title="E-SDAG Component Value",
            showlegend=True,
            xaxis={'type': 'linear', 'automargin': True},
            margin=dict(l=60, r=30, t=60, b=60)
        )
        # Apply custom dark theme styling
        apply_dark_theme_template(fig)
        add_price_line(fig, current_price, orientation='vertical', width=2, color='white')
        add_bottom_right_timestamp_annotation(fig, bundle.bundle_timestamp)
        
        chart_component = dcc.Graph(
            figure=fig,
            config={
                'displayModeBar': False,
                'displaylogo': False
            }
        )
        return _wrap_chart_in_card(chart_component, about_text, "esdag-charts")
    except Exception as e:
        logger.error(f"Error creating {chart_name}: {e}", exc_info=True)
        chart_component = html.Div([
            dbc.Alert(f"Error: {e}", color="danger", className="mb-2"),
            dcc.Graph(
                figure=create_empty_figure(chart_name, fig_height, f"Error: {e}"),
                config={
                    'displayModeBar': False,
                    'displaylogo': False
                }
            )
        ])
        return _wrap_chart_in_card(chart_component, about_text, "esdag-charts")

def _generate_adag_strike_chart(bundle, config):
    chart_name = "A-DAG by Strike"
    struct_settings = getattr(config, 'structure_mode_settings', None)
    fig_height = getattr(struct_settings, 'adag_chart_height', 350) if struct_settings else 350
    
    about_text = (
        "📈 A-DAG by Strike (Adaptive Delta-Adjusted Gamma): This shows the NET DIRECTIONAL PRESSURE at each strike after adapting to current market conditions. "
        "GREEN bars = NET BUYING pressure expected (support). "
        "RED bars = NET SELLING pressure expected (resistance). "
        "BAR HEIGHT = Magnitude of expected dealer hedging activity. "
        "This is MORE ADVANCED than regular gamma exposure because it: "
        "1) Adapts to market regime (trending vs ranging), "
        "2) Incorporates recent flow alignment, "
        "3) Adjusts for time decay effects. "
        "💡 TRADING INSIGHT: The LARGEST bars (positive or negative) are your KEY LEVELS for the day. "
        "Price tends to 'MAGNETIZE' toward large positive A-DAG strikes (dealer buying). "
        "Price tends to 'REJECT' from large negative A-DAG strikes (dealer selling). "
        "When A-DAG flips from positive to negative (or vice versa) = MAJOR INFLECTION POINT. "
        "In trending markets, trade WITH the A-DAG direction. In ranging markets, FADE extreme A-DAG levels."
    )
    
    try:
        strike_data = bundle.processed_data_bundle.strike_level_data_with_metrics
        if not strike_data:
            logger.warning("[A-DAG] No strike data available.")
            chart_component = html.Div([
                dbc.Alert("No strike level data available.", color="warning", className="mb-2"),
                dcc.Graph(
                    figure=create_empty_figure(chart_name, fig_height, "Strike level data not available."),
                    config={
                        'displayModeBar': False,
                        'displaylogo': False
                    }
                )
            ])
            return _wrap_chart_in_card(chart_component, about_text, "adag-strike-chart")
        df_strike = pd.DataFrame([m.model_dump() for m in strike_data])
        metric_to_plot = 'a_dag_strike'
        if df_strike.empty or metric_to_plot not in df_strike.columns:
            logger.warning(f"[A-DAG] '{metric_to_plot}' data not found in DataFrame columns: {df_strike.columns}")
            chart_component = html.Div([
                dbc.Alert(f"'{metric_to_plot}' data not found.", color="warning", className="mb-2"),
                dcc.Graph(
                    figure=create_empty_figure(chart_name, fig_height, f"'{metric_to_plot}' data not found."),
                    config={
                        'displayModeBar': False,
                        'displaylogo': False
                    }
                )
            ])
            return _wrap_chart_in_card(chart_component, about_text, "adag-strike-chart")
        df_plot = df_strike.dropna(subset=['strike', metric_to_plot]).sort_values('strike')
        # logger.info(f"Processed {len(df_plot)} strikes for {bundle.target_symbol}")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Processed {len(df_plot)} strikes for {bundle.target_symbol}")
        if df_plot.empty or df_plot[metric_to_plot].isnull().all():
            logger.warning("[A-DAG] All A-DAG values are missing or None.")
            chart_component = html.Div([
                dbc.Alert("All A-DAG values are missing or None.", color="warning", className="mb-2"),
                dcc.Graph(
                    figure=create_empty_figure(chart_name, fig_height, "All A-DAG values are missing or None."),
                    config={
                        'displayModeBar': False,
                        'displaylogo': False
                    }
                )
            ])
            about_text = (
                "📈 A-DAG by Strike (Adaptive Delta-Adjusted Gamma): This shows the NET DIRECTIONAL PRESSURE at each strike after adapting to current market conditions. "
                "GREEN bars = NET BUYING pressure expected (support). "
                "RED bars = NET SELLING pressure expected (resistance). "
                "BAR HEIGHT = Magnitude of expected dealer hedging activity. "
                "This is MORE ADVANCED than regular gamma exposure because it: "
                "1) Adapts to market regime (trending vs ranging), "
                "2) Incorporates recent flow alignment, "
                "3) Adjusts for time decay effects. "
                "💡 TRADING INSIGHT: The LARGEST bars (positive or negative) are your KEY LEVELS for the day. "
                "Price tends to 'MAGNETIZE' toward large positive A-DAG strikes (dealer buying). "
                "Price tends to 'REJECT' from large negative A-DAG strikes (dealer selling). "
                "When A-DAG flips from positive to negative (or vice versa) = MAJOR INFLECTION POINT. "
                "In trending markets, trade WITH the A-DAG direction. In ranging markets, FADE extreme A-DAG levels."
            )
            return _wrap_chart_in_card(chart_component, about_text, "adag-strike-chart")
        colors = ['#d62728' if x < 0 else '#2ca02c' for x in df_plot[metric_to_plot]]
        fig = go.Figure(data=[go.Bar(
            x=df_plot['strike'],
            y=df_plot[metric_to_plot],
            name='A-DAG',
            marker_color=colors,
            hovertemplate='Strike: %{x}<br>A-DAG: %{y:,.2f}<extra></extra>'
        )])
        current_price = bundle.processed_data_bundle.underlying_data_enriched.price
        price_range_percent = getattr(config, 'price_range_percent', 20) if getattr(config, 'price_range_percent', None) is not None else 20

        # CRITICAL FIX: Fail-fast for invalid current_price instead of using fake data
        if current_price is None or math.isnan(current_price):
            logger.error(f"[{chart_name}] Invalid current price - cannot generate chart")
            return dcc.Graph(
                figure=create_empty_figure(chart_name, fig_height, "Invalid current price data."),
                config={'displayModeBar': False, 'displaylogo': False}
            )
        # Defensive: ensure strikes is not empty and contains only valid floats
        strikes = df_plot['strike'].tolist()  # FIX: define strikes from df_plot
        strikes_valid = [s for s in strikes if isinstance(s, float) and not math.isnan(s)]
        if len(strikes_valid) == 0:
            # CRITICAL FIX: Fail-fast instead of using fake data - no valid strikes means no chart
            logger.error(f"[{chart_name}] No valid strikes available - cannot generate chart")
            return dcc.Graph(
                figure=create_empty_figure(chart_name, fig_height, "No valid strike data available."),
                config={'displayModeBar': False, 'displaylogo': False}
            )

        # CRITICAL FIX: Use real data only - fail-fast if NaN values detected
        min_strike_raw = np.nanmin(strikes_valid)
        max_strike_raw = np.nanmax(strikes_valid)

        if math.isnan(min_strike_raw) or math.isnan(max_strike_raw):
            logger.error(f"[{chart_name}] Invalid strike range detected - cannot generate chart")
            return dcc.Graph(
                figure=create_empty_figure(chart_name, fig_height, "Invalid strike range data."),
                config={'displayModeBar': False, 'displaylogo': False}
            )

        min_strike = float(min_strike_raw)
        max_strike = float(max_strike_raw)
        pad = (strikes_valid[1] - strikes_valid[0]) if len(strikes_valid) > 1 else 5.0

        # CRITICAL FIX: Fail-fast for invalid current_price instead of using fake data
        if current_price is None or math.isnan(current_price):
            logger.error(f"[{chart_name}] Invalid current price - cannot generate chart")
            return dcc.Graph(
                figure=create_empty_figure(chart_name, fig_height, "Invalid current price data."),
                config={'displayModeBar': False, 'displaylogo': False}
            )

        valid_for_minmax = [float(v) for v in [min_strike, current_price] if v is not None and not math.isnan(v)]
        x_min = min(valid_for_minmax) - pad
        valid_for_max = [float(v) for v in [max_strike, current_price] if v is not None and not math.isnan(v)]
        x_max = max(valid_for_max) + pad
        fig.update_layout(
            title_text=f"<b>{bundle.target_symbol}</b> - {chart_name}",
            height=fig_height,
            template=PLOTLY_TEMPLATE,
            xaxis_title="Strike Price",
            yaxis_title="A-DAG Value",
            showlegend=False,
            xaxis={'type': 'linear', 'automargin': True},
            margin=dict(l=60, r=30, t=60, b=60)
        )
        # Apply custom dark theme styling
        apply_dark_theme_template(fig)
        add_price_line(fig, current_price, orientation='vertical', width=2, color='white')
        add_bottom_right_timestamp_annotation(fig, bundle.bundle_timestamp)
        chart_component = dcc.Graph(
            figure=fig,
            config={
                'displayModeBar': False,
                'displaylogo': False
            }
        )
        about_text = (
            "📈 A-DAG by Strike (Adaptive Delta-Adjusted Gamma): This shows the NET DIRECTIONAL PRESSURE at each strike after adapting to current market conditions. "
            "GREEN bars = NET BUYING pressure expected (support). "
            "RED bars = NET SELLING pressure expected (resistance). "
            "BAR HEIGHT = Magnitude of expected dealer hedging activity. "
            "This is MORE ADVANCED than regular gamma exposure because it: "
            "1) Adapts to market regime (trending vs ranging), "
            "2) Incorporates recent flow alignment, "
            "3) Adjusts for time decay effects. "
            "💡 TRADING INSIGHT: The LARGEST bars (positive or negative) are your KEY LEVELS for the day. "
            "Price tends to 'MAGNETIZE' toward large positive A-DAG strikes (dealer buying). "
            "Price tends to 'REJECT' from large negative A-DAG strikes (dealer selling). "
            "When A-DAG flips from positive to negative (or vice versa) = MAJOR INFLECTION POINT. "
            "In trending markets, trade WITH the A-DAG direction. In ranging markets, FADE extreme A-DAG levels."
        )
        return _wrap_chart_in_card(chart_component, about_text, "adag-strike-chart")
    except Exception as e:
        logger.error(f"Error creating {chart_name}: {e}", exc_info=True)
        chart_component = html.Div([
            dbc.Alert(f"Error: {e}", color="danger", className="mb-2"),
            dcc.Graph(
                figure=create_empty_figure(chart_name, fig_height, f"Error: {e}"),
                config={
                    'displayModeBar': False,
                    'displaylogo': False
                }
            )
        ])
        about_text = (
            "📈 A-DAG by Strike (Adaptive Delta-Adjusted Gamma): This shows the NET DIRECTIONAL PRESSURE at each strike after adapting to current market conditions. "
            "GREEN bars = NET BUYING pressure expected (support). "
            "RED bars = NET SELLING pressure expected (resistance). "
            "BAR HEIGHT = Magnitude of expected dealer hedging activity. "
            "This is MORE ADVANCED than regular gamma exposure because it: "
            "1) Adapts to market regime (trending vs ranging), "
            "2) Incorporates recent flow alignment, "
            "3) Adjusts for time decay effects. "
            "💡 TRADING INSIGHT: The LARGEST bars (positive or negative) are your KEY LEVELS for the day. "
            "Price tends to 'MAGNETIZE' toward large positive A-DAG strikes (dealer buying). "
            "Price tends to 'REJECT' from large negative A-DAG strikes (dealer selling). "
            "When A-DAG flips from positive to negative (or vice versa) = MAJOR INFLECTION POINT. "
            "In trending markets, trade WITH the A-DAG direction. In ranging markets, FADE extreme A-DAG levels."
        )
        return _wrap_chart_in_card(chart_component, about_text, "adag-strike-chart")

def _generate_asai_assi_charts(bundle, config):
    chart_name = "A-SAI & A-SSI (Aggregate Structural Indexes)"
    fig_height = getattr(config, 'asai_assi_chart_height', 350)
    try:
        und_data = bundle.processed_data_bundle.underlying_data_enriched
        # [A-SAI/A-SSI] underlying_data_enriched: ...
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[A-SAI/A-SSI] underlying_data_enriched: {und_data}")
        # logger.info(f"[A-SAI/A-SSI] underlying_data_enriched received for {getattr(und_data, 'symbol', 'N/A')}")
        asai = getattr(und_data, 'a_sai_und_avg', None)
        assi = getattr(und_data, 'a_ssi_und_avg', None)
        # logger.info(f"[A-SAI/A-SSI] a_sai_und_avg: {asai}, a_ssi_und_avg: {assi}")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[A-SAI/A-SSI] a_sai_und_avg: {asai}, a_ssi_und_avg: {assi}")
        fig = go.Figure()
        missing = []
        if asai is not None:
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=asai,
                title={"text": "A-SAI (Support Aggregate Index)"},
                gauge={"axis": {"range": [-1, 1]}, "bar": {"color": "#2ca02c"}},
                domain={"row": 0, "column": 0}
            ))
        else:
            missing.append("A-SAI (Support Aggregate Index)")
        if assi is not None:
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=assi,
                title={"text": "A-SSI (Resistance Aggregate Index)"},
                gauge={"axis": {"range": [-1, 1]}, "bar": {"color": "#d62728"}},
                domain={"row": 0, "column": 1}
            ))
        else:
            missing.append("A-SSI (Resistance Aggregate Index)")
        fig.update_layout(
            title_text=f"<b>{bundle.target_symbol}</b> - {chart_name}",
            height=fig_height,
            template=PLOTLY_TEMPLATE,
            grid={"rows": 1, "columns": 2, "pattern": "independent"},
            margin=dict(l=60, r=30, t=60, b=60),
            hovermode='x',        # Use x-axis hover to reduce obstruction
            hoverdistance=20      # Reduced hover distance to minimize obstruction
        )
        add_bottom_right_timestamp_annotation(fig, bundle.bundle_timestamp)
        warning = dbc.Alert(f"Missing: {', '.join(missing)}" if missing else None, color="warning", className="mb-2") if missing else None
        chart_component = html.Div(
            ([warning] if warning else []) + [dcc.Graph(
                figure=fig,
                config={
                    'displayModeBar': False,
                    'displaylogo': False
                }
            )]
        )
        about_text = (
            "🎚️ A-SAI & A-SSI Gauges (Adaptive Support/Resistance Aggregate Indexes): These gauges show the OVERALL MARKET STRUCTURE at a glance. "
            "A-SAI (GREEN gauge) = Aggregate SUPPORT strength index (-1 to +1). "
            "A-SSI (RED gauge) = Aggregate RESISTANCE strength index (-1 to +1). "
            "POSITIVE A-SAI = Strong support structure below current price (bullish). "
            "NEGATIVE A-SSI = Strong resistance structure above current price (bearish). "
            "BOTH NEAR ZERO = Balanced/neutral structure. "
            "💡 TRADING INSIGHT: These are your 'MARKET STRUCTURE COMPASS'. "
            "A-SAI > 0.5 = VERY BULLISH structure, dips are buying opportunities. "
            "A-SSI < -0.5 = VERY BEARISH structure, rallies are selling opportunities. "
            "When BOTH are extreme (SAI > 0.7, SSI < -0.7) = RANGE-BOUND market, fade extremes. "
            "When they FLIP (SAI goes negative or SSI goes positive) = MAJOR STRUCTURE CHANGE, potential trend reversal! "
            "These update throughout the day as options flow shifts the structural balance."
        )
        return _wrap_chart_in_card(chart_component, about_text, "asai-assi-charts")
    except Exception as e:
        logger.error(f"Error creating {chart_name}: {e}", exc_info=True)
        chart_component = html.Div([
            dbc.Alert(f"Error: {e}", color="danger", className="mb-2"),
            dcc.Graph(
                figure=create_empty_figure(chart_name, fig_height, f"Error: {e}"),
                config={
                    'displayModeBar': False,
                    'displaylogo': False
                }
            )
        ])
        about_text = (
            "🎚️ A-SAI & A-SSI Gauges (Adaptive Support/Resistance Aggregate Indexes): These gauges show the OVERALL MARKET STRUCTURE at a glance. "
            "A-SAI (GREEN gauge) = Aggregate SUPPORT strength index (-1 to +1). "
            "A-SSI (RED gauge) = Aggregate RESISTANCE strength index (-1 to +1). "
            "POSITIVE A-SAI = Strong support structure below current price (bullish). "
            "NEGATIVE A-SSI = Strong resistance structure above current price (bearish). "
            "BOTH NEAR ZERO = Balanced/neutral structure. "
            "💡 TRADING INSIGHT: These are your 'MARKET STRUCTURE COMPASS'. "
            "A-SAI > 0.5 = VERY BULLISH structure, dips are buying opportunities. "
            "A-SSI < -0.5 = VERY BEARISH structure, rallies are selling opportunities. "
            "When BOTH are extreme (SAI > 0.7, SSI < -0.7) = RANGE-BOUND market, fade extremes. "
            "When they FLIP (SAI goes negative or SSI goes positive) = MAJOR STRUCTURE CHANGE, potential trend reversal! "
            "These update throughout the day as options flow shifts the structural balance."
        )
        return _wrap_chart_in_card(chart_component, about_text, "asai-assi-charts")

def _generate_key_level_table(bundle, config):
    """
    Generates the Key Level Identifier Table using only the canonical Pydantic model (KeyLevelsDataV2_5).
    Flattens all key level lists, adds a level_type column, sorts by conviction_score, and provides robust error/empty handling.
    """
    chart_name = "Key Level Identifier Table"
    try:
        key_levels_data = getattr(bundle, 'key_levels_data_v2_5', None)
        if not key_levels_data:
            raise ValueError("No key level data present in bundle.")

        # Flatten all key levels into a single list, tagging with type
        all_levels = []
        for attr in ['supports', 'resistances', 'pin_zones', 'vol_triggers', 'major_walls']:
            levels = getattr(key_levels_data, attr, [])
            for lvl in levels:
                d = lvl.model_dump()
                # CRITICAL FIX: Use getattr instead of dictionary .get() method
                d['level_type'] = getattr(lvl, 'level_type', attr[:-1].capitalize())
                all_levels.append(d)

        if not all_levels:
            return html.Div(_about_section(
                "🎯 Key Level Identifier Table: This table shows ALL CRITICAL PRICE LEVELS identified by the system's advanced algorithms. "
                "TYPES: Support (price floor), Resistance (price ceiling), Pin Zone (magnetic price), Vol Trigger (volatility expansion), Major Wall (massive OI). "
                "CONVICTION SCORE (0-1): Higher = stronger level. Above 0.7 = VERY HIGH conviction. "
                "METRICS: Shows which calculations identified this level (technical, options-based, flow-based). "
                "💡 TRADING INSIGHT: These are your BATTLE LINES for the day. "
                "SUPPORT levels = Where to buy/cover shorts, place stops below. "
                "RESISTANCE levels = Where to sell/short, place stops above. "
                "PIN ZONES = Price will gravitate here, especially near expiration. Great for iron condors/butterflies. "
                "VOL TRIGGERS = Breakout/breakdown points. Break = volatility expansion. Use for straddle/strangle entries. "
                "MAJOR WALLS = Extreme levels that rarely break. Use for credit spread strikes. "
                "Sort by CONVICTION to focus on the strongest levels. Filter by TYPE for specific strategies. "
                "These levels are DYNAMIC and update as market structure evolves!",
                "key-level-table"
            ) + [
                dbc.Alert("No key levels identified.", color="warning", className="mb-2")
            ])

        df = pd.DataFrame(all_levels)
        # Optional: sort by conviction_score descending
        if 'conviction_score' in df.columns:
            df = df.sort_values("conviction_score", ascending=False)

        # Render table with robust columns
        columns = [
            {"name": "Level Price", "id": "level_price"},
            {"name": "Type", "id": "level_type"},
            {"name": "Conviction Score", "id": "conviction_score"},
            {"name": "Contributing Metrics", "id": "contributing_metrics"},
            {"name": "Source", "id": "source_identifier"}
        ]
        # Ensure all columns exist in DataFrame
        for col in [c["id"] for c in columns]:
            if col not in df.columns:
                df[col] = None
        # Format contributing_metrics as comma-separated string
        if 'contributing_metrics' in df.columns:
            df['contributing_metrics'] = df['contributing_metrics'].apply(lambda x: ', '.join(x) if isinstance(x, list) else (x or ''))

        table = dash_table.DataTable(
            columns=columns,
            data=df.to_dict('records'),
            style_cell={'textAlign': 'left', 'padding': '5px', 'minWidth': '80px', 'width': 'auto', 'maxWidth': '200px'},
            style_header={'backgroundColor': 'rgb(30, 30, 30)', 'fontWeight': 'bold', 'color': 'white'},
            style_data={'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white'},
            style_as_list_view=True,
            page_size=10,
            sort_action="native",
            filter_action="native",
        )
        return html.Div(_about_section(
            "🎯 Key Level Identifier Table: This table shows ALL CRITICAL PRICE LEVELS identified by the system's advanced algorithms. "
            "TYPES: Support (price floor), Resistance (price ceiling), Pin Zone (magnetic price), Vol Trigger (volatility expansion), Major Wall (massive OI). "
            "CONVICTION SCORE (0-1): Higher = stronger level. Above 0.7 = VERY HIGH conviction. "
            "METRICS: Shows which calculations identified this level (technical, options-based, flow-based). "
            "💡 TRADING INSIGHT: These are your BATTLE LINES for the day. "
            "SUPPORT levels = Where to buy/cover shorts, place stops below. "
            "RESISTANCE levels = Where to sell/short, place stops above. "
            "PIN ZONES = Price will gravitate here, especially near expiration. Great for iron condors/butterflies. "
            "VOL TRIGGERS = Breakout/breakdown points. Break = volatility expansion. Use for straddle/strangle entries. "
            "MAJOR WALLS = Extreme levels that rarely break. Use for credit spread strikes. "
            "Sort by CONVICTION to focus on the strongest levels. Filter by TYPE for specific strategies. "
            "These levels are DYNAMIC and update as market structure evolves!",
            "key-level-table"
        ) + [table])
    except Exception as e:
        logger.error(f"[Key Level Table] Error: {e}", exc_info=True)
        return dbc.Alert("Key level data unavailable.", color="danger")

# --- Main Layout Function ---
def create_layout(bundle, config):
    """
    Creates the complete layout for the Structure Mode Display.
    Enforces Pydantic validation at the UI boundary.
    """
    if not isinstance(bundle, FinalAnalysisBundleV2_5):
        raise ValueError("Input bundle is not a FinalAnalysisBundleV2_5 Pydantic model.")

    # --- DIAGNOSTIC LOGGING ---
    try:
        logger.info(f"[StructureMode] Bundle type: {type(bundle)}")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[StructureMode] Bundle keys: {list(bundle.__dict__.keys()) if hasattr(bundle, '__dict__') else 'N/A'}")
        pdb = getattr(bundle, 'processed_data_bundle', None)
        if pdb:
            contracts = getattr(pdb, 'options_data_with_metrics', [])
            strikes = getattr(pdb, 'strike_level_data_with_metrics', [])
            # logger.info(f"Processed {len(contracts)} contracts, {len(strikes)} strikes for {bundle.target_symbol}")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Processed {len(contracts)} contracts, {len(strikes)} strikes for {bundle.target_symbol}")
            if strikes and logger.isEnabledFor(logging.DEBUG):
                first_row = strikes[0].model_dump() if hasattr(strikes[0], 'model_dump') else str(strikes[0])
                logger.debug(f"[StructureMode] First strike row: {first_row}")
            elif not strikes:
                logger.info("[StructureMode] No strike data present.")
        else:
            logger.info("[StructureMode] No processed_data_bundle in bundle.")
    except Exception as e:
        logger.error(f"[StructureMode] Diagnostic logging failed: {e}", exc_info=True)

    warnings = []
    if not bundle or not getattr(bundle, 'processed_data_bundle', None):
        warnings.append("Structural data is not available. Cannot render Structure Mode.")
    else:
        strike_data = getattr(bundle.processed_data_bundle, 'strike_level_data_with_metrics', None)
        if not strike_data:
            warnings.append("Strike level data not available.")

    chart_blocks = [
        _generate_amspi_heatmap(bundle, config),
        _generate_esdag_charts(bundle, config),
        _generate_adag_strike_chart(bundle, config),
        _generate_asai_assi_charts(bundle, config),
        _generate_key_level_table(bundle, config)
    ]

    return html.Div([
        dbc.Container([
            html.H2("Structure & Dealer Positioning", className="mb-4 mt-2"),
            *([dbc.Alert(w, color="warning", className="mb-2")] for w in warnings),
            dbc.Row([
                dbc.Col(block, width=12, className="mb-4") for block in chart_blocks
            ])
        ], fluid=True)
    ])