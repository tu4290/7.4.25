import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

import dash_bootstrap_components as dbc
from dash import dcc, html

from data_models.processed_data import FinalAnalysisBundleV2_5, ProcessedUnderlyingAggregatesV2_5, ProcessedStrikeLevelMetricsV2_5
from core_analytics_engine.eots_metrics.elite_definitions import ConvexValueColumns, EliteImpactColumns
from utils.config_manager_v2_5 import ConfigManagerV2_5
from dashboard_application import ids

def create_layout(bundle: FinalAnalysisBundleV2_5, config: ConfigManagerV2_5) -> html.Div:
    """
    Creates the layout for the main dashboard display, incorporating elite metrics
    and visualizations.
    """
    und_data = bundle.processed_data_bundle.underlying_data_enriched
    strike_data_list = bundle.processed_data_bundle.strike_level_data_with_metrics
    
    # Convert list of Pydantic models to DataFrame for easier plotting
    strike_df = pd.DataFrame([s.model_dump() for s in strike_data_list]) if strike_data_list else pd.DataFrame()

    # Helper to safely get attribute value or default
    def get_attr_or_default(obj, attr, default="---", formatter=None):
        value = getattr(obj, attr, None)
        if value is None:
            return default
        if formatter:
            return formatter(value)
        return value

    # --- Elite Overview Card ---
    elite_overview_card = dbc.Card([
        dbc.CardHeader(html.H5("📊 Elite Overview", className="mb-0")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Small("Elite Impact Score:", className="text-muted d-block"),
                    html.Span(get_attr_or_default(und_data, 'elite_impact_score_und', formatter=lambda x: f'{x:.4f}'), id=ids.ID_ELITE_IMPACT_SCORE_DISPLAY, className="fw-bold text-primary")
                ], width=4),
                dbc.Col([
                    html.Small("Institutional Flow:", className="text-muted d-block"),
                    html.Span(get_attr_or_default(und_data, 'institutional_flow_score_und', formatter=lambda x: f'{x:.4f}'), id=ids.ID_INSTITUTIONAL_FLOW_SCORE_DISPLAY, className="fw-bold text-info")
                ], width=4),
                dbc.Col([
                    html.Small("Flow Momentum:", className="text-muted d-block"),
                    html.Span(get_attr_or_default(und_data, 'flow_momentum_index_und', formatter=lambda x: f'{x:.4f}'), id=ids.ID_FLOW_MOMENTUM_INDEX_DISPLAY, className="fw-bold text-success")
                ], width=4)
            ], className="g-2 mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Small("Elite Market Regime:", className="text-muted d-block"),
                    html.Span(get_attr_or_default(und_data, 'market_regime_elite'), id=ids.ID_ELITE_MARKET_REGIME_DISPLAY, className="fw-bold text-warning")
                ], width=6),
                dbc.Col([
                    html.Small("Elite Flow Type:", className="text-muted d-block"),
                    html.Span(get_attr_or_default(und_data, 'flow_type_elite'), id=ids.ID_ELITE_FLOW_TYPE_DISPLAY, className="fw-bold text-danger")
                ], width=6)
            ], className="g-2 mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Small("Elite Volatility Regime:", className="text-muted d-block"),
                    html.Span(get_attr_or_default(und_data, 'volatility_regime_elite'), id=ids.ID_ELITE_VOLATILITY_REGIME_DISPLAY, className="fw-bold text-info")
                ], width=12)
            ], className="g-2 mb-3")
        ])
    ], className="mb-4 elite-card")

    # --- Charts ---
    charts = []

    # 1. Elite Impact Score Distribution
    if EliteImpactColumns.ELITE_IMPACT_SCORE in strike_df.columns and not strike_df.empty:
        fig_impact_dist = px.histogram(strike_df, x=EliteImpactColumns.ELITE_IMPACT_SCORE, nbins=20,
                                       title='Elite Impact Score Distribution',
                                       template="plotly_dark")
        fig_impact_dist.update_layout(showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
        charts.append(dbc.Col(
            dbc.Card([
                dbc.CardHeader(html.H5("📈 Elite Impact Score Distribution", className="mb-0")),
                dbc.CardBody([
                    dcc.Graph(figure=fig_impact_dist, id='elite-impact-score-distribution-chart')
                ])
            ], className="mb-4 elite-card"),
            width=6
        ))
    else:
        charts.append(dbc.Col(dbc.Card(dbc.CardBody("Elite Impact Score Distribution data not available."), className="mb-4 elite-card"), width=6))

    # 2. SDAG Consensus vs Strike
    if EliteImpactColumns.SDAG_CONSENSUS in strike_df.columns and not strike_df.empty:
        fig_sdag = px.scatter(strike_df, x=ConvexValueColumns.STRIKE, y=EliteImpactColumns.SDAG_CONSENSUS,
                              color=EliteImpactColumns.PREDICTION_CONFIDENCE, size=EliteImpactColumns.SIGNAL_STRENGTH,
                              hover_data=[ConvexValueColumns.OPT_KIND, ConvexValueColumns.EXPIRATION],
                              title='SDAG Consensus vs Strike',
                              template="plotly_dark")
        fig_sdag.add_vline(x=und_data.price, line_width=2, line_dash="dash", line_color="yellow", annotation_text="Current Price")
        fig_sdag.update_layout(showlegend=True, margin=dict(l=20, r=20, t=40, b=20))
        charts.append(dbc.Col(
            dbc.Card([
                dbc.CardHeader(html.H5("🎯 SDAG Consensus vs Strike", className="mb-0")),
                dbc.CardBody([
                    dcc.Graph(figure=fig_sdag, id='sdag-consensus-chart')
                ])
            ], className="mb-4 elite-card"),
            width=6
        ))
    else:
        charts.append(dbc.Col(dbc.Card(dbc.CardBody("SDAG Consensus data not available."), className="mb-4 elite-card"), width=6))

    # 3. Strike Magnetism Index
    if EliteImpactColumns.STRIKE_MAGNETISM_INDEX in strike_df.columns and not strike_df.empty:
        fig_smi = px.line(strike_df, x=ConvexValueColumns.STRIKE, y=EliteImpactColumns.STRIKE_MAGNETISM_INDEX,
                          title='Strike Magnetism Index',
                          template="plotly_dark")
        fig_smi.add_vline(x=und_data.price, line_width=2, line_dash="dash", line_color="yellow", annotation_text="Current Price")
        fig_smi.update_layout(showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
        charts.append(dbc.Col(
            dbc.Card([
                dbc.CardHeader(html.H5("🏰 Strike Magnetism Index", className="mb-0")),
                dbc.CardBody([
                    dcc.Graph(figure=fig_smi, id='strike-magnetism-index-chart')
                ])
            ], className="mb-4 elite-card"),
            width=6
        ))
    else:
        charts.append(dbc.Col(dbc.Card(dbc.CardBody("Strike Magnetism Index data not available."), className="mb-4 elite-card"), width=6))

    # 4. Volatility Pressure Index
    if EliteImpactColumns.VOLATILITY_PRESSURE_INDEX in strike_df.columns and not strike_df.empty:
        fig_vpi = px.line(strike_df, x=ConvexValueColumns.STRIKE, y=EliteImpactColumns.VOLATILITY_PRESSURE_INDEX,
                          title='Volatility Pressure Index',
                          template="plotly_dark")
        fig_vpi.add_vline(x=und_data.price, line_width=2, line_dash="dash", line_color="yellow", annotation_text="Current Price")
        fig_vpi.update_layout(showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
        charts.append(dbc.Col(
            dbc.Card([
                dbc.CardHeader(html.H5("🌪️ Volatility Pressure Index", className="mb-0")),
                dbc.CardBody([
                    dcc.Graph(figure=fig_vpi, id='volatility-pressure-index-chart')
                ])
            ], className="mb-4 elite-card"),
            width=6
        ))
    else:
        charts.append(dbc.Col(dbc.Card(dbc.CardBody("Volatility Pressure Index data not available."), className="mb-4 elite-card"), width=6))

    # 5. Flow Momentum Index
    if EliteImpactColumns.FLOW_MOMENTUM_INDEX in strike_df.columns and not strike_df.empty:
        fig_fmi = px.line(strike_df, x=ConvexValueColumns.STRIKE, y=EliteImpactColumns.FLOW_MOMENTUM_INDEX,
                          title='Flow Momentum Index',
                          template="plotly_dark")
        fig_fmi.add_vline(x=und_data.price, line_width=2, line_dash="dash", line_color="yellow", annotation_text="Current Price")
        fig_fmi.update_layout(showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
        charts.append(dbc.Col(
            dbc.Card([
                dbc.CardHeader(html.H5("🚀 Flow Momentum Analysis", className="mb-0")),
                dbc.CardBody([
                    dcc.Graph(figure=fig_fmi, id='flow-momentum-chart')
                ])
            ], className="mb-4 elite-card"),
            width=6
        ))
    else:
        charts.append(dbc.Col(dbc.Card(dbc.CardBody("Flow Momentum Index data not available."), className="mb-4 elite-card"), width=6))

    # 6. Institutional Flow Score (if available at strike level, otherwise use underlying)
    if EliteImpactColumns.INSTITUTIONAL_FLOW_SCORE in strike_df.columns and not strike_df.empty:
        fig_ifs = px.bar(strike_df, x=ConvexValueColumns.STRIKE, y=EliteImpactColumns.INSTITUTIONAL_FLOW_SCORE,
                         title='Institutional Flow Score by Strike',
                         template="plotly_dark")
        fig_ifs.update_layout(showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
        charts.append(dbc.Col(
            dbc.Card([
                dbc.CardHeader(html.H5("💰 Institutional Flow Score", className="mb-0")),
                dbc.CardBody([
                    dcc.Graph(figure=fig_ifs, id='institutional-flow-score-chart')
                ])
            ], className="mb-4 elite-card"),
            width=6
        ))
    else:
        charts.append(dbc.Col(dbc.Card(dbc.CardBody("Institutional Flow Score data not available."), className="mb-4 elite-card"), width=6))

    return html.Div([
        dbc.Container(elite_overview_card, fluid=True),
        dbc.Container(dbc.Row(charts), fluid=True)
    ])