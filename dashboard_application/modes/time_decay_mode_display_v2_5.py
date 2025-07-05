# dashboard_application/modes/time_decay_mode_display_v2_5.py
# EOTS v2.5 - S-GRADE, AUTHORITATIVE TIME DECAY MODE DISPLAY

import logging
import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.development.base_component import Component

from data_models import FinalAnalysisBundleV2_5 # Updated import
from utils.config_manager_v2_5 import ConfigManagerV2_5

logger = logging.getLogger(__name__)

# --- Helper Functions for Chart Generation ---

def _generate_tdpi_ectr_etdfi_charts(bundle: FinalAnalysisBundleV2_5, config: ConfigManagerV2_5) -> Component:
    """
    Generates D-TDPI, E-CTR, and E-TDFI by strike as a multi-metric chart.
    Strict Pydantic-first: expects a list of ProcessedStrikeLevelMetricsV2_5 models, converts to DataFrame at UI boundary.
    """
    chart_name = "D-TDPI, E-CTR, E-TDFI by Strike"
    td_settings = getattr(config, 'time_decay_mode_settings', None)
    fig_height = getattr(td_settings, 'tdpi_chart_height', 500) if td_settings else 500
    about_text = (
        "📊 D-TDPI, E-CTR, E-TDFI by Strike: This chart visualizes the three core time decay and pinning metrics across all strikes. "
        "D-TDPI (Directional Time Decay Pinning Index) shows net directional pinning pressure. "
        "E-CTR (Effective Call/Put Trade Ratio) highlights call/put dominance. "
        "E-TDFI (Effective Time Decay Flow Index) quantifies net time decay flow. "
        "Use this chart to spot where pinning, call/put imbalances, and time decay flows cluster by strike. "
        "💡 TRADING INSIGHT: Large D-TDPI values = strong pinning. E-CTR extremes = call/put dominance. E-TDFI spikes = time decay inflection. "
        "Combine all three for a holistic view of intraday pinning and decay dynamics."
    )
    component_id = "tdpi-ectr-etdfi-charts"
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
        required_cols = ["strike", "d_tdpi_strike", "e_ctr_strike", "e_tdfi_strike"]
        for col in required_cols:
            if col not in df.columns:
                chart_component = html.Div([
                    dbc.Alert(f"Column '{col}' not found in strike-level data for {chart_name}.", color="danger")
                ])
                return _wrap_chart_in_card(chart_component, about_text, component_id)
        # Plot
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df["strike"],
            y=df["d_tdpi_strike"],
            name="D-TDPI",
            marker_color="#7b2ff2"
        ))
        fig.add_trace(go.Scatter(
            x=df["strike"],
            y=df["e_ctr_strike"],
            name="E-CTR",
            mode="lines+markers",
            marker_color="#f357a8"
        ))
        fig.add_trace(go.Scatter(
            x=df["strike"],
            y=df["e_tdfi_strike"],
            name="E-TDFI",
            mode="lines+markers",
            marker_color="#43e97b"
        ))
        fig.update_layout(
            height=fig_height,
            title=chart_name,
            margin=dict(l=40, r=40, t=60, b=40),
            xaxis_title="Strike",
            yaxis_title="Metric Value",
            template="plotly_dark"
        )
        chart_component = dcc.Graph(figure=fig, style={"width": "100%", "height": f"{fig_height}px"}, config={"displayModeBar": False, "displaylogo": False})
        return _wrap_chart_in_card(chart_component, about_text, component_id)
    except Exception as e:
        logging.exception("Error rendering TDPI chart")
        chart_component = html.Div([
            dbc.Alert(f"Error rendering {chart_name}: {e}", color="danger")
        ])
        return _wrap_chart_in_card(chart_component, about_text, component_id)

def _generate_vci_gci_dci_gauges(bundle: FinalAnalysisBundleV2_5, config: ConfigManagerV2_5) -> Component:
    """
    Generates VCI, GCI, DCI gauges for 0DTE.
    Strict Pydantic-first: expects a list of ProcessedStrikeLevelMetricsV2_5 models, converts to DataFrame at UI boundary.
    """
    chart_name = "0DTE Concentration Indices (VCI, GCI, DCI)"
    td_settings = getattr(config, 'time_decay_mode_settings', None)
    fig_height = getattr(td_settings, 'vci_chart_height', 250) if td_settings else 250
    about_text = (
        "🎯 0DTE Concentration Indices (VCI, GCI, DCI): These gauges show the concentration of volume (VCI), gamma (GCI), and delta (DCI) at 0DTE strikes. "
        "VCI = where most 0DTE volume is clustered. GCI = where gamma risk is highest. DCI = where delta risk is highest. "
        "Use these to spot where market makers are most exposed on 0DTE expiries. "
        "💡 TRADING INSIGHT: High VCI = liquidity magnet. High GCI = gamma flip zone. High DCI = directional risk. "
        "Watch for clustering of all three for major inflection points."
    )
    component_id = "vci-gci-dci-gauges"
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
        required_cols = ["strike", "vci_0dte", "gci_0dte", "dci_0dte"]
        for col in required_cols:
            if col not in df.columns:
                chart_component = html.Div([
                    dbc.Alert(f"Column '{col}' not found in strike-level data for {chart_name}.", color="danger")
                ])
                return _wrap_chart_in_card(chart_component, about_text, component_id)
        vci = df["vci_0dte"].max()
        gci = df["gci_0dte"].max()
        dci = df["dci_0dte"].max()
        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=vci,
            title={"text": "VCI 0DTE"},
            domain={"row": 0, "column": 0}
        ))
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=gci,
            title={"text": "GCI 0DTE"},
            domain={"row": 0, "column": 1}
        ))
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=dci,
            title={"text": "DCI 0DTE"},
            domain={"row": 0, "column": 2}
        ))
        fig.update_layout(
            grid={"rows": 1, "columns": 3, "pattern": "independent"},
            height=300,
            margin=dict(l=40, r=40, t=60, b=40),
            template="plotly_dark"
        )
        chart_component = dcc.Graph(figure=fig, style={"width": "100%", "height": "300px"}, config={"displayModeBar": False, "displaylogo": False})
        return _wrap_chart_in_card(chart_component, about_text, component_id)
    except Exception as e:
        logging.exception("Error rendering VCI/GCI/DCI gauges")
        chart_component = html.Div([
            dbc.Alert(f"Error rendering {chart_name}: {e}", color="danger")
        ])
        return _wrap_chart_in_card(chart_component, about_text, component_id)

# --- Contextual Panels ---

def _ticker_context_panel(bundle: FinalAnalysisBundleV2_5) -> html.Div:
    """
    Displays key session, expiry, and event flags for the current symbol from ticker_context_dict_v2_5.
    """
    ctx = getattr(bundle.processed_data_bundle.underlying_data_enriched, 'ticker_context_dict_v2_5', None)
    if not ctx or not hasattr(ctx, 'model_dump'):
        return html.Div([
            dbc.Alert("Ticker context unavailable.", color="secondary", className="mb-2")
        ])
    ctx_dict = ctx.model_dump()
    # Group related flags for better readability
    expiry_flags = [
        ("0DTE", ctx_dict.get("is_0dte")),
        ("1DTE", ctx_dict.get("is_1dte")),
        ("SPX MWF Expiry", ctx_dict.get("is_spx_mwf_expiry_type")),
        ("SPY EOM Expiry", ctx_dict.get("is_spy_eom_expiry")),
        ("Quad Witching Week", ctx_dict.get("is_quad_witching_week_flag")),
    ]
    event_flags = [
        ("FOMC Meeting Day", ctx_dict.get("is_fomc_meeting_day")),
        ("FOMC Announcement Imminent", ctx_dict.get("is_fomc_announcement_imminent")),
        ("Post-FOMC Drift", ctx_dict.get("post_fomc_drift_period_active")),
        ("Earnings Approaching", ctx_dict.get("earnings_approaching_flag")),
        ("Days to Earnings", ctx_dict.get("days_to_earnings")),
    ]
    session_flags = [
        ("Active Intraday Session", ctx_dict.get("active_intraday_session")),
        ("Near Auction Period", ctx_dict.get("is_near_auction_period")),
        ("Days to Nearest 0DTE", ctx_dict.get("days_to_nearest_0dte")),
        ("Days to Monthly OPEX", ctx_dict.get("days_to_monthly_opex")),
    ]
    liquidity_flags = [
        ("Liquidity Profile", ctx_dict.get("ticker_liquidity_profile_flag")),
        ("Volatility State", ctx_dict.get("ticker_volatility_state_flag")),
        ("VIX-SPY Price Divergence Strong Negative", ctx_dict.get("vix_spy_price_divergence_strong_negative")),
    ]
    def flag_li(label, value):
        color = "#43e97b" if value is True else ("#f357a8" if value is False else "#cccccc")
        return html.Li([
            html.Span(f"{label}: ", style={"fontWeight": "bold"}),
            html.Span(str(value), style={"color": color})
        ])
    return html.Div([
        dbc.Alert([
            html.B("Ticker Context Panel: "),
            "Key session, expiry, and event flags for the current symbol."
        ], color="info", dismissable=False, className="mb-2"),
        html.Div([
            html.B("Expiry Flags:"),
            html.Ul([flag_li(label, val) for label, val in expiry_flags]),
            html.B("Event Flags:"),
            html.Ul([flag_li(label, val) for label, val in event_flags]),
            html.B("Session Flags:"),
            html.Ul([flag_li(label, val) for label, val in session_flags]),
            html.B("Liquidity/Volatility Flags:"),
            html.Ul([flag_li(label, val) for label, val in liquidity_flags]),
        ])
    ])

def _expiration_calendar_panel(bundle: FinalAnalysisBundleV2_5) -> html.Div:
    """
    Displays a table of upcoming expiry dates, highlighting today and special expiries.
    """
    import datetime
    # Try to extract expiry dates from options_data_with_metrics
    options = getattr(bundle.processed_data_bundle, 'options_data_with_metrics', None)
    expiry_dates = set()
    if options and isinstance(options, list):
        for opt in options:
            exp = getattr(opt, 'expiration_str', None)
            if exp:
                try:
                    expiry_dates.add(datetime.datetime.strptime(exp, "%Y-%m-%d").date())
                except Exception:
                    continue
    expiry_dates = sorted(list(expiry_dates))
    today = datetime.date.today()
    if not expiry_dates:
        return html.Div([
            dbc.Alert("No expiry data available for this symbol.", color="secondary", className="mb-2")
        ])
    # Mark special expiries (monthly = third Friday, EOM = last trading day of month)
    def is_monthly(date):
        return date.weekday() == 4 and 15 <= date.day <= 21  # Third Friday
    def is_eom(date):
        next_day = date + datetime.timedelta(days=1)
        return next_day.month != date.month
    rows = []
    for d in expiry_dates[:5]:
        style = {}
        label = d.strftime("%Y-%m-%d")
        if d == today:
            style["fontWeight"] = "bold"
            style["color"] = "#43e97b"
            label += " (Today)"
        if is_monthly(d):
            label += " [Monthly]"
        if is_eom(d):
            label += " [EOM]"
        rows.append(html.Tr([html.Td(label, style=style)]))
    return html.Div([
        dbc.Alert([
            html.B("Expiration Calendar: "),
            "Upcoming expiry dates and 0DTE/1DTE focus (SPY/SPX)."
        ], color="info", dismissable=False, className="mb-2"),
        html.Table([
            html.Thead(html.Tr([html.Th("Expiry Date")])) ,
            html.Tbody(rows)
        ], className="table table-dark table-sm")
    ])

def _session_clock_panel(bundle: FinalAnalysisBundleV2_5, config: ConfigManagerV2_5) -> html.Div:
    """
    Displays the current session and time remaining to close using TimeOfDayDefinitions from config and system time.
    """
    import datetime
    # Use config object passed in
    time_defs = getattr(config, 'config', None)
    if time_defs and hasattr(time_defs, 'time_of_day_definitions'):
        tod = time_defs.time_of_day_definitions
    else:
        # Fallback defaults
        class Dummy:
            market_open = "09:30:00"
            market_close = "16:00:00"
            pre_market_start = "04:00:00"
            after_hours_end = "20:00:00"
            eod_pressure_calc_time = "15:00:00"
        tod = Dummy()
    now = datetime.datetime.now()
    def parse_time(tstr):
        return datetime.datetime.strptime(tstr, "%H:%M:%S").time()
    open_time = parse_time(getattr(tod, 'market_open', "09:30:00"))
    close_time = parse_time(getattr(tod, 'market_close', "16:00:00"))
    session = "Pre-market"
    if now.time() >= open_time and now.time() < close_time:
        session = "Regular Trading"
    elif now.time() >= close_time:
        session = "After-hours"
    # Time to close
    close_dt = now.replace(hour=close_time.hour, minute=close_time.minute, second=close_time.second, microsecond=0)
    if now > close_dt:
        close_dt += datetime.timedelta(days=1)
    time_left = close_dt - now
    return html.Div([
        dbc.Alert([
            html.B("Session Clock: "),
            f"Current session: {session}, time to close: {str(time_left).split('.')[0]}"
        ], color="info", dismissable=False, className="mb-2")
    ])

def _behavioral_patterns_panel(bundle: FinalAnalysisBundleV2_5) -> html.Div:
    """
    Summarizes pinning/cascade risk, regime, and session context if available.
    """
    # Try to extract regime and pinning/cascade risk from bundle
    regime = getattr(bundle.processed_data_bundle.underlying_data_enriched, 'current_market_regime_v2_5', None)
    ctx = getattr(bundle.processed_data_bundle.underlying_data_enriched, 'ticker_context_dict_v2_5', None)
    pinning = None
    cascade = None
    if ctx and hasattr(ctx, 'model_dump'):
        ctx_dict = ctx.model_dump()
        pinning = ctx_dict.get('pinning_risk_flag')
        cascade = ctx_dict.get('cascade_risk_flag')
    items = []
    if regime:
        items.append(html.Li([html.B("Market Regime: "), str(regime)]))
    if pinning is not None:
        items.append(html.Li([html.B("Pinning Risk: "), str(pinning)]))
    if cascade is not None:
        items.append(html.Li([html.B("Cascade Risk: "), str(cascade)]))
    if not items:
        items.append(html.Li("No special behavioral patterns detected."))
    return html.Div([
        dbc.Alert([
            html.B("Behavioral Patterns: "),
            "Summary of detected pinning/cascade risk, regime, and session context."
        ], color="info", dismissable=False, className="mb-2"),
        html.Ul(items)
    ])

# --- Mini Heatmap ---
def _mini_heatmap(bundle: FinalAnalysisBundleV2_5, config: ConfigManagerV2_5) -> Component:
    """
    Generates Pin Risk/Net Value Flow Mini Heatmap.
    Strict Pydantic-first: expects a list of ProcessedStrikeLevelMetricsV2_5 models, converts to DataFrame at UI boundary.
    """
    chart_name = "Pin Risk/Net Value Flow Mini Heatmap"
    fig_height = 300
    about_text = (
        "🔥 Pin Risk/Net Value Flow Mini Heatmap: Visualizes net value flow and pin risk intensity by strike. "
        "Red = net negative value (selling/pressure), Green = net positive value (buying/support). "
        "Use this to spot where price is likely to pin or cascade. "
        "💡 TRADING INSIGHT: Strong green = pin/support, strong red = cascade/resistance. "
        "Look for clusters and transitions for actionable trade zones."
    )
    component_id = "mini-heatmap"
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
        if "strike" not in df.columns or "nvp_at_strike" not in df.columns:
            chart_component = html.Div([
                dbc.Alert("Required columns for heatmap not found.", color="danger")
            ])
            return _wrap_chart_in_card(chart_component, about_text, component_id)
        fig = go.Figure(data=go.Heatmap(
            z=[df["nvp_at_strike"].values],
            x=df["strike"].values,
            y=["Net Value"],
            colorscale="RdYlGn"
        ))
        fig.update_layout(
            height=fig_height,
            title=chart_name,
            margin=dict(l=40, r=40, t=60, b=40),
            template="plotly_dark"
        )
        chart_component = dcc.Graph(figure=fig, style={"width": "100%", "height": "300px"}, config={"displayModeBar": False, "displaylogo": False})
        return dbc.Card(
            dbc.CardBody([
                dbc.Button(
                    "ℹ️ About",
                    id={"type": "about-toggle", "index": component_id},
                    color="link",
                    size="sm",
                    className="p-0 text-elite-secondary mb-2 elite-focus-visible",
                    style={'font-size': '0.75em'}
                ),
                dbc.Collapse(
                    html.Small(about_text, className="text-elite-secondary d-block mb-2", style={'font-size': '0.75em'}),
                    id={"type": "about-collapse", "index": component_id},
                    is_open=False
                ),
                chart_component
            ], className="elite-card-body", style={"height": "auto"}),
            className="elite-card fade-in-up",
            style={"height": "auto"}
        )
    except Exception as e:
        logging.exception("Error rendering Pin Risk/Net Value Flow Mini Heatmap")
        chart_component = html.Div([
            dbc.Alert(f"Error rendering {chart_name}: {e}", color="danger")
        ])
        return _wrap_chart_in_card(chart_component, about_text, component_id)

# --- Overlay Pin Zones/Key Levels on Main Chart ---
def _add_pin_zone_overlays(fig, bundle: FinalAnalysisBundleV2_5):
    """
    Adds pin zone overlays to a Plotly figure using key levels from the bundle.
    Expects bundle.key_levels_data_v2_5.pin_zones as a list of KeyLevelV2_5 Pydantic models.
    """
    pin_zones = getattr(bundle.key_levels_data_v2_5, 'pin_zones', None)
    if not isinstance(pin_zones, list) or not all(hasattr(pz, 'model_dump') for pz in pin_zones):
        return  # No overlays if not a list of Pydantic models
    for pin in pin_zones:
        pin_dict = pin.model_dump()
        level = pin_dict.get('level_price')
        if level is not None:
            fig.add_vrect(
                x0=level - 0.5, x1=level + 0.5,
                fillcolor="rgba(255, 255, 0, 0.15)",
                line_width=0,
                annotation_text="Pin Zone",
                annotation_position="top left"
            )

# --- Patch main chart to add overlays ---
_old_generate_tdpi = _generate_tdpi_ectr_etdfi_charts

def _generate_tdpi_ectr_etdfi_charts_with_overlays(bundle: FinalAnalysisBundleV2_5, config: ConfigManagerV2_5) -> Component:
    div = _old_generate_tdpi(bundle, config)
    # Patch: add overlays if possible
    try:
        # Find the dcc.Graph in the div and add overlays
        for c in getattr(div, 'children', []):
            if isinstance(c, dcc.Graph):
                fig = getattr(c, 'figure', None)
                if fig is not None:
                    _add_pin_zone_overlays(fig, bundle)
    except Exception as e:
        logger.warning(f"Could not add overlays to main chart: {e}")
    return div

# --- Card/Info Helper (mirroring structure mode) ---
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
    """Wraps a chart component in a card with about section, following structure mode pattern."""
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

# --- Main Layout Function (extended) ---
def create_layout(bundle: FinalAnalysisBundleV2_5, config: ConfigManagerV2_5) -> html.Div:
    """
    Creates the complete layout for the "Time Decay & Pinning" mode.
    Pydantic-first: All data bundles are Pydantic models/lists; DataFrame conversion only happens at the UI boundary in each chart generator.
    Chart generators must not expect DataFrames from the bundle.
    """
    if not bundle or not bundle.processed_data_bundle:
        return html.Div([
            dbc.Alert("Time Decay data is not available. Cannot render Time Decay Mode.", color="danger")
        ])

    # Main chart cards (each already wrapped in a card)
    chart_divs = [
        _generate_tdpi_ectr_etdfi_charts(bundle, config),
        _generate_vci_gci_dci_gauges(bundle, config),
        _mini_heatmap(bundle, config)
    ]

    # Contextual panels (not wrapped in cards)
    context_panels = [
        _ticker_context_panel(bundle),
        _expiration_calendar_panel(bundle),
        _session_clock_panel(bundle, config),
        _behavioral_patterns_panel(bundle)
    ]

    return html.Div([
        dbc.Container([
            html.H2("Time Decay & Pinning Metrics", className="mb-4 mt-2"),
            dbc.Row([
                dbc.Col(card, width=12, className="mb-4") for card in chart_divs
            ]),
            html.H3("Contextual Information", className="mb-3 mt-4"),
            *context_panels
        ], fluid=True)
    ])