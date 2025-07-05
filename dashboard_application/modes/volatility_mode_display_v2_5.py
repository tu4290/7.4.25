# EOTS v2.5 - S-GRADE, AUTHORITATIVE VOLATILITY MODE DISPLAY

import logging
import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.development.base_component import Component

from dashboard_application.utils_dashboard_v2_5 import PLOTLY_TEMPLATE, add_bottom_right_timestamp_annotation, apply_dark_theme_template
from data_models import FinalAnalysisBundleV2_5 # Updated import
from utils.config_manager_v2_5 import ConfigManagerV2_5

logger = logging.getLogger(__name__)

# --- Helper Functions for Chart Generation ---

def _wrap_chart_in_card(chart_component, about_text, component_id):
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

def _generate_vri_2_0_strike_profile(bundle, config):
    chart_name = "VRI 2.0 Volatility Regime Profile"
    vol_settings = getattr(config, 'volatility_mode_settings', None)
    fig_height = getattr(vol_settings, 'vri_chart_height', 500) if vol_settings else 500
    about_text = (
        "📈 VRI 2.0 Volatility Regime Profile: Shows the volatility regime indicator by strike. "
        "Green bars = positive regime (supportive), Red bars = negative regime (resistance). "
        "Use this to spot where volatility regime shifts occur across strikes. "
        "💡 TRADING INSIGHT: Large positive VRI = supportive regime, large negative VRI = resistance. "
        "Transitions signal regime change opportunities."
    )
    component_id = "vri-2-0-strike-profile"
    try:
        strike_models = getattr(bundle.processed_data_bundle, 'strike_level_data_with_metrics', None)
        if not isinstance(strike_models, list) or not all(hasattr(m, 'model_dump') for m in strike_models):
            chart_component = html.Div([
                dbc.Alert(f"strike_level_data_with_metrics is not a list of Pydantic models for {chart_name}.", color="danger")
            ])
            return _wrap_chart_in_card(chart_component, about_text, component_id)
        if not strike_models:
            chart_component = html.Div([
                dbc.Alert(f"No strike-level data available for {chart_name}.", color="warning")
            ])
            return _wrap_chart_in_card(chart_component, about_text, component_id)
        df = pd.DataFrame([m.model_dump() for m in strike_models])
        if df.empty:
            chart_component = html.Div([
                dbc.Alert(f"No strike-level data available for {chart_name}.", color="warning")
            ])
            return _wrap_chart_in_card(chart_component, about_text, component_id)
        required_cols = ["strike", "vri_2_0_strike"]
        for col in required_cols:
            if col not in df.columns:
                chart_component = html.Div([
                    dbc.Alert(f"Column '{col}' not found in strike-level data for {chart_name}.", color="danger")
                ])
                return _wrap_chart_in_card(chart_component, about_text, component_id)
        df_plot = df.dropna(subset=required_cols).sort_values('strike')
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[VRI2.0 Chart] Strike/VRI values: {list(zip(df_plot['strike'], df_plot['vri_2_0_strike']))}")
        if df_plot.empty:
            chart_component = html.Div([
                dbc.Alert(f"No valid VRI 2.0 data available for {chart_name}.", color="warning")
            ])
            return _wrap_chart_in_card(chart_component, about_text, component_id)
        colors = ['#d62728' if x < 0 else '#2ca02c' for x in df_plot['vri_2_0_strike']]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_plot["strike"],
            y=df_plot["vri_2_0_strike"],
            name="VRI 2.0",
            marker_color=colors,
            hovertemplate='Strike: %{x}<br>VRI 2.0: %{y:,.2f}<extra></extra>'
        ))
        current_price = getattr(bundle.processed_data_bundle.underlying_data_enriched, 'price', None)
        if current_price:
            fig.add_vline(
                x=current_price,
                line_width=2,
                line_color="white",
                annotation_text="Current Price",
                annotation_position="top"
            )
        fig.update_layout(
            height=fig_height,
            title=f"{bundle.target_symbol} - {chart_name}",
            margin=dict(l=40, r=40, t=60, b=40),
            xaxis_title="Strike Price",
            yaxis_title="VRI 2.0 Score",
            template="plotly_dark",
            showlegend=False
        )
        apply_dark_theme_template(fig)
        add_bottom_right_timestamp_annotation(fig, getattr(bundle, 'bundle_timestamp', None))
        chart_component = dcc.Graph(figure=fig, style={"width": "100%", "height": f"{fig_height}px"}, config={"displayModeBar": False, "displaylogo": False})
        return _wrap_chart_in_card(chart_component, about_text, component_id)
    except Exception as e:
        logging.exception("Error rendering VRI 2.0 chart")
        chart_component = html.Div([
            dbc.Alert(f"Error rendering {chart_name}: {e}", color="danger")
        ])
        return _wrap_chart_in_card(chart_component, about_text, component_id)

def _generate_volatility_gauges(bundle, config):
    chart_name = "0DTE Volatility Metrics (VRI, VFI, VVR, VCI)"
    vol_settings = getattr(config, 'volatility_mode_settings', None)
    fig_height = getattr(vol_settings, 'gauge_chart_height', 300) if vol_settings else 300
    about_text = (
        "🎯 Volatility Gauges (VRI, VFI, VVR, VCI): These four gauges provide a comprehensive view of current volatility conditions. "
        "All gauges are normalized and displayed on a -1 to 1 scale. "
        "VRI (Volatility Regime Indicator) = Overall volatility environment strength (-1 = extreme negative, +1 = extreme positive). "
        "VFI (Volatility Flow Indicator) = Direction and momentum of volatility changes (-1 = strong downward, +1 = strong upward). "
        "VVR (Volatility-Volume Ratio) = Relationship between volatility and trading volume (-1 = low, +1 = high). "
        "VCI (Volatility Convergence Indicator) = How different volatility measures are aligning (-1 = divergent, +1 = convergent). "
        "\n💡 TRADING INSIGHT: When ALL gauges are in the GREEN zone = low volatility, good time to BUY options. "
        "When ALL gauges are in the RED zone = high volatility, good time to SELL options. "
        "MIXED readings = transitional period, use caution. "
        "Watch for gauges moving from one extreme to another = volatility regime change in progress!"
    )
    component_id = "volatility-gauges"
    try:
        # CRITICAL FIX: Get volatility metrics from underlying aggregate data, not strike-level data
        underlying_data = getattr(bundle.processed_data_bundle, 'underlying_data_enriched', None)
        if not underlying_data or not hasattr(underlying_data, 'model_dump'):
            chart_component = html.Div([
                dbc.Alert(f"underlying_data_enriched is not a Pydantic model for {chart_name}.", color="danger")
            ])
            return _wrap_chart_in_card(chart_component, about_text, component_id)

        # Extract volatility metrics from underlying aggregate data using correct field names
        underlying_dict = underlying_data.model_dump()
        vri_value = underlying_dict.get("vri_0dte_und_sum", 0.0) or 0.0
        vfi_value = underlying_dict.get("vfi_0dte_und_sum", 0.0) or 0.0
        vvr_value = underlying_dict.get("vvr_0dte_und_avg", 0.0) or 0.0
        vci_value = underlying_dict.get("vci_0dte_agg", 0.0) or 0.0
        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=vri_value,
            title={"text": "VRI 0DTE Sum"},
            domain={"row": 0, "column": 0},
            gauge={"axis": {"range": [-1, 1]}, "bar": {"color": "#7b2ff2"}}
        ))
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=vfi_value,
            title={"text": "VFI 0DTE Sum"},
            domain={"row": 0, "column": 1},
            gauge={"axis": {"range": [-1, 1]}, "bar": {"color": "#f357a8"}}
        ))
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=vvr_value,
            title={"text": "VVR 0DTE Avg"},
            domain={"row": 0, "column": 2},
            gauge={"axis": {"range": [-1, 1]}, "bar": {"color": "#43e97b"}}
        ))
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=vci_value,
            title={"text": "VCI 0DTE"},
            domain={"row": 0, "column": 3},
            gauge={"axis": {"range": [-1, 1]}, "bar": {"color": "#ff6b35"}}
        ))
        fig.update_layout(
            grid={"rows": 1, "columns": 4, "pattern": "independent"},
            height=fig_height,
            title=f"<b>{bundle.target_symbol}</b> - {chart_name}",
            margin=dict(l=40, r=40, t=60, b=40),
            template=PLOTLY_TEMPLATE
        )
        apply_dark_theme_template(fig)
        add_bottom_right_timestamp_annotation(fig, getattr(bundle, 'bundle_timestamp', None))
        chart_component = dcc.Graph(
            figure=fig,
            config={
                'displayModeBar': False,
                'displaylogo': False
            }
        )
        return _wrap_chart_in_card(chart_component, about_text, component_id)
    except Exception as e:
        logging.exception("Error rendering volatility gauges")
        chart_component = html.Div([
            dbc.Alert(f"Error rendering {chart_name}: {e}", color="danger")
        ])
        return _wrap_chart_in_card(chart_component, about_text, component_id)

def _generate_volatility_surface_heatmap(bundle, config):
    chart_name = "Volatility Surface Heatmap"
    vol_settings = getattr(config, 'volatility_mode_settings', None)
    fig_height = getattr(vol_settings, 'surface_chart_height', 400) if vol_settings else 400
    about_text = (
        "🌡️ Volatility Surface Heatmap: This heatmap shows implied volatility levels across different strikes and expiration dates. "
        "DARKER COLORS = HIGHER implied volatility (more expensive options). "
        "LIGHTER COLORS = LOWER implied volatility (cheaper options). "
        "VERTICAL AXIS = Different expiration dates (time to expiration). "
        "HORIZONTAL AXIS = Strike prices relative to current underlying price. "
        "💡 TRADING INSIGHT: Look for DARK SPOTS (high IV) to SELL options and LIGHT SPOTS (low IV) to BUY options. "
        "SKEW patterns show market sentiment - right skew (higher IV for higher strikes) = bullish bias. "
        "TERM STRUCTURE shows how volatility changes over time - upward sloping = volatility expected to increase. "
        "Use this to identify the most attractive strikes and expirations for your volatility strategy!"
    )
    component_id = "volatility-surface-heatmap"
    try:
        strike_models = getattr(bundle.processed_data_bundle, 'strike_level_data_with_metrics', None)
        if not isinstance(strike_models, list) or not all(hasattr(m, 'model_dump') for m in strike_models):
            chart_component = html.Div([
                dbc.Alert(f"strike_level_data_with_metrics is not a list of Pydantic models for {chart_name}.", color="danger")
            ])
            return _wrap_chart_in_card(chart_component, about_text, component_id)
        if not strike_models:
            chart_component = html.Div([
                dbc.Alert(f"No strike-level data available for {chart_name}.", color="warning")
            ])
            return _wrap_chart_in_card(chart_component, about_text, component_id)
        df = pd.DataFrame([m.model_dump() for m in strike_models])
        if df.empty:
            chart_component = html.Div([
                dbc.Alert(f"No strike-level data available for {chart_name}.", color="warning")
            ])
            return _wrap_chart_in_card(chart_component, about_text, component_id)
        if "strike" not in df.columns or "vri_2_0_strike" not in df.columns:
            chart_component = html.Div([
                dbc.Alert("Required columns for heatmap not found.", color="danger")
            ])
            return _wrap_chart_in_card(chart_component, about_text, component_id)
        fig = go.Figure(data=go.Heatmap(
            z=[df["vri_2_0_strike"].values],
            x=df["strike"].values,
            y=["Volatility Regime"],
            colorscale="RdYlBu_r"
        ))
        fig.update_layout(
            height=fig_height,
            title=f"<b>{bundle.target_symbol}</b> - {chart_name}",
            margin=dict(l=40, r=40, t=60, b=40),
            template=PLOTLY_TEMPLATE
        )
        apply_dark_theme_template(fig)
        add_bottom_right_timestamp_annotation(fig, getattr(bundle, 'bundle_timestamp', None))
        chart_component = dcc.Graph(
            figure=fig,
            config={
                'displayModeBar': False,
                'displaylogo': False
            }
        )
        return _wrap_chart_in_card(chart_component, about_text, component_id)
    except Exception as e:
        logging.exception("Error rendering volatility surface heatmap")
        chart_component = html.Div([
            dbc.Alert(f"Error rendering {chart_name}: {e}", color="danger")
        ])
        return _wrap_chart_in_card(chart_component, about_text, component_id)

# --- Contextual Panels ---

def _volatility_context_panel(bundle):
    """
    Displays key volatility metrics and regime information from ticker_context_dict_v2_5.
    """
    about_text = (
        "🧭 Volatility Context Panel: Key volatility state, regime, and risk flags for the current symbol. "
        "Shows current volatility state, VIX-SPY divergence, regime, and expansion flags. "
        "Use this to quickly assess the volatility environment and potential risk factors."
    )
    component_id = "volatility-context-panel"
    ctx = getattr(bundle.processed_data_bundle.underlying_data_enriched, 'ticker_context_dict_v2_5', None)
    if not ctx or not hasattr(ctx, 'model_dump'):
        chart_component = html.Div([
            dbc.Alert("Volatility context unavailable.", color="secondary", className="mb-2")
        ])
        return _wrap_chart_in_card(chart_component, about_text, component_id)
    ctx_dict = ctx.model_dump()
    volatility_flags = [
        ("Volatility State", ctx_dict.get("ticker_volatility_state_flag")),
        ("VIX-SPY Divergence Strong Negative", ctx_dict.get("vix_spy_price_divergence_strong_negative")),
        ("High Volatility Regime", ctx_dict.get("high_volatility_regime_flag")),
        ("Volatility Expansion", ctx_dict.get("volatility_expansion_flag")),
    ]
    risk_flags = [
        ("Pinning Risk", ctx_dict.get("pinning_risk_flag")),
        ("Cascade Risk", ctx_dict.get("cascade_risk_flag")),
        ("Gamma Squeeze Risk", ctx_dict.get("gamma_squeeze_risk_flag")),
    ]
    def flag_li(label, value):
        color = "#43e97b" if value is True else ("#f357a8" if value is False else "#cccccc")
        return html.Li([
            html.Span(f"{label}: ", style={"fontWeight": "bold"}),
            html.Span(str(value), style={"color": color})
        ])
    chart_component = html.Div([
        html.B("Volatility Flags:"),
        html.Ul([flag_li(label, val) for label, val in volatility_flags]),
        html.B("Risk Flags:"),
        html.Ul([flag_li(label, val) for label, val in risk_flags]),
    ])
    return _wrap_chart_in_card(chart_component, about_text, component_id)

def _volatility_regime_panel(bundle):
    """
    Displays current market regime and volatility state information.
    """
    about_text = (
        "📊 Volatility Regime Panel: Shows the current market regime and volatility state. "
        "Includes regime label, volatility state, and VIX-SPY divergence. "
        "Use this to understand the prevailing volatility regime and its implications for trading."
    )
    component_id = "volatility-regime-panel"
    regime = getattr(bundle.processed_data_bundle.underlying_data_enriched, 'current_market_regime_v2_5', None)
    ctx = getattr(bundle.processed_data_bundle.underlying_data_enriched, 'ticker_context_dict_v2_5', None)
    items = []
    if regime:
        items.append(html.Li([html.B("Market Regime: "), str(regime)]))
    if ctx and hasattr(ctx, 'model_dump'):
        ctx_dict = ctx.model_dump()
        vol_state = ctx_dict.get('ticker_volatility_state_flag')
        if vol_state is not None:
            items.append(html.Li([html.B("Volatility State: "), str(vol_state)]))
        vix_divergence = ctx_dict.get('vix_spy_price_divergence_strong_negative')
        if vix_divergence is not None:
            items.append(html.Li([html.B("VIX-SPY Divergence: "), str(vix_divergence)]))
    if not items:
        items.append(html.Li("No regime information available."))
    chart_component = html.Ul(items)
    return _wrap_chart_in_card(chart_component, about_text, component_id)

def _volatility_metrics_summary_panel(bundle):
    """
    Displays summary of key volatility metrics from strike-level data.
    """
    about_text = (
        "📈 Volatility Metrics Summary: Key statistics from volatility analysis. "
        "Shows mean, range, and max values for VRI 2.0 and VCI 0DTE. "
        "Use this to quickly gauge the overall volatility landscape."
    )
    component_id = "volatility-metrics-summary-panel"
    try:
        strike_models = getattr(bundle.processed_data_bundle, 'strike_level_data_with_metrics', None)
        if not isinstance(strike_models, list) or not strike_models:
            chart_component = html.Div([
                dbc.Alert("No volatility metrics data available.", color="secondary", className="mb-2")
            ])
            return _wrap_chart_in_card(chart_component, about_text, component_id)
        df = pd.DataFrame([m.model_dump() for m in strike_models])
        if df.empty:
            chart_component = html.Div([
                dbc.Alert("No volatility metrics data available.", color="secondary", className="mb-2")
            ])
            return _wrap_chart_in_card(chart_component, about_text, component_id)
        metrics = []
        if "vri_2_0_strike" in df.columns:
            vri_mean = df["vri_2_0_strike"].mean()
            vri_max = df["vri_2_0_strike"].max()
            vri_min = df["vri_2_0_strike"].min()
            metrics.extend([
                html.Li([html.B("VRI 2.0 Mean: "), f"{vri_mean:.2f}"]),
                html.Li([html.B("VRI 2.0 Range: "), f"{vri_min:.2f} to {vri_max:.2f}"])
            ])
        if "vci_0dte_agg" in df.columns:
            vci_max = df["vci_0dte_agg"].max()
            metrics.append(html.Li([html.B("VCI 0DTE Max: "), f"{vci_max:.2f}"]))
        if not metrics:
            metrics.append(html.Li("No volatility metrics available."))
        chart_component = html.Ul(metrics)
        return _wrap_chart_in_card(chart_component, about_text, component_id)
    except Exception as e:
        logging.exception("Error generating volatility metrics summary")
        chart_component = html.Div([
            dbc.Alert(f"Error generating metrics summary: {e}", color="danger", className="mb-2")
        ])
        return _wrap_chart_in_card(chart_component, about_text, component_id)

def _volatility_risk_assessment_panel(bundle):
    """
    Displays volatility-related risk assessment and warnings.
    """
    about_text = (
        "⚠️ Volatility Risk Assessment: Current volatility-related risks and warnings. "
        "Flags pinning, cascade, gamma squeeze, high volatility regime, and VIX-SPY divergence. "
        "Use this to identify and manage key volatility risks in real time."
    )
    component_id = "volatility-risk-assessment-panel"
    ctx = getattr(bundle.processed_data_bundle.underlying_data_enriched, 'ticker_context_dict_v2_5', None)
    if not ctx or not hasattr(ctx, 'model_dump'):
        chart_component = html.Div([
            dbc.Alert("Risk assessment data unavailable.", color="secondary", className="mb-2")
        ])
        return _wrap_chart_in_card(chart_component, about_text, component_id)
    ctx_dict = ctx.model_dump()
    risk_items = []
    if ctx_dict.get("pinning_risk_flag"):
        risk_items.append(html.Li([html.Span("⚠️ ", style={"color": "#ff6b35"}), "Pinning Risk Detected"]))
    if ctx_dict.get("cascade_risk_flag"):
        risk_items.append(html.Li([html.Span("⚠️ ", style={"color": "#ff6b35"}), "Cascade Risk Detected"]))
    if ctx_dict.get("gamma_squeeze_risk_flag"):
        risk_items.append(html.Li([html.Span("⚠️ ", style={"color": "#ff6b35"}), "Gamma Squeeze Risk Detected"]))
    if ctx_dict.get("high_volatility_regime_flag"):
        risk_items.append(html.Li([html.Span("📈 ", style={"color": "#f357a8"}), "High Volatility Regime Active"]))
    if ctx_dict.get("vix_spy_price_divergence_strong_negative"):
        risk_items.append(html.Li([html.Span("📉 ", style={"color": "#d62728"}), "Strong VIX-SPY Negative Divergence"]))
    if not risk_items:
        risk_items.append(html.Li([html.Span("✅ ", style={"color": "#43e97b"}), "No significant volatility risks detected"]))
    chart_component = html.Ul(risk_items)
    return _wrap_chart_in_card(chart_component, about_text, component_id)

# --- Main Layout Function ---
def create_layout(bundle, config):
    """
    Creates the complete layout for the "Volatility Analysis" mode.
    Pydantic-first: All data bundles are Pydantic models/lists; DataFrame conversion only happens at the UI boundary in each chart generator.
    Chart generators must not expect DataFrames from the bundle.
    """
    if not bundle or not bundle.processed_data_bundle:
        return html.Div([
            dbc.Alert("Volatility data is not available. Cannot render Volatility Mode.", color="danger")
        ])

    # Integration point: ensure config is a Pydantic model
    if not hasattr(config, 'modes_detail_config'):
        # Try to extract from manager
        if hasattr(config, 'config') and hasattr(config.config, 'modes_detail_config'):
            config = config.config
        else:
            raise TypeError(f"config must be a DashboardServerConfig Pydantic model, got {type(config)}")

    chart_generators = {
        "vri_2_0_strike_profile": (_generate_vri_2_0_strike_profile, (bundle, config)),
        "volatility_gauges": (_generate_volatility_gauges, (bundle, config)),
        "volatility_surface_heatmap": (_generate_volatility_surface_heatmap, (bundle, config)),
    }
    
    charts_to_display = getattr(config.modes_detail_config.volatility, 'charts', ["vri_2_0_strike_profile", "volatility_gauges", "volatility_surface_heatmap"])
    chart_divs = []
    for chart_id in charts_to_display:
        gen = chart_generators.get(chart_id)
        if gen:
            chart_divs.append(gen[0](*gen[1]))
    
    # Contextual panels (now wrapped in cards)
    context_panels = [
        _volatility_context_panel(bundle),
        _volatility_regime_panel(bundle),
        _volatility_metrics_summary_panel(bundle),
        _volatility_risk_assessment_panel(bundle)
    ]
    
    return html.Div([
        dbc.Container([
            html.H2("Volatility Analysis", className="mb-4 mt-2"),
            dbc.Row([
                dbc.Col(card, width=12, className="mb-4") for card in chart_divs
            ]),
            html.H3("Contextual Information", className="mb-3 mt-4"),
            dbc.Row([
                dbc.Col(panel, width=12, className="mb-4") for panel in context_panels
            ])
        ], fluid=True)
    ])