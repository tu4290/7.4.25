"""
AI Dashboard Callbacks v2.5 - Pydantic-First Architecture
Critical callbacks for AI hub functionality and advanced options metrics integration.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime

from dash import Input, Output
import plotly.graph_objects as go

from data_models import FinalAnalysisBundleV2_5, AdvancedOptionsMetricsV2_5
import compliance_decorators_v2_5

logger = logging.getLogger(__name__)

def register_ai_dashboard_callbacks(app, orchestrator):
    """
    Register AI dashboard callbacks for advanced options metrics and AI hub functionality.
    
    Args:
        app: Dash application instance
        orchestrator: ITS Orchestrator instance for data processing
    """
    
    @app.callback(
        [
            Output('ai-hub-advanced-metrics-display', 'children'),
            Output('ai-hub-lwpai-gauge', 'figure'),
            Output('ai-hub-vabai-gauge', 'figure'),
            Output('ai-hub-aofm-gauge', 'figure'),
            Output('ai-hub-lidb-gauge', 'figure'),
        ],
        [
            Input('main-data-store-id', 'data'),
            Input('symbol-input-id', 'value'),
        ],
        prevent_initial_call=False
    )
    def update_ai_hub_advanced_metrics(main_data: Dict, symbol: str):
        """🚀 PYDANTIC-FIRST: Update AI hub advanced options metrics display with real data."""
        logger.info(f"🔥 AI HUB CALLBACK TRIGGERED! symbol={symbol}, main_data_exists={bool(main_data)}")

        try:
            if not main_data or not symbol:
                logger.warning(f"❌ AI HUB: Missing data - main_data={bool(main_data)}, symbol={symbol}")
                return _create_empty_metrics_display()

            logger.info(f"🔍 AI HUB: Processing data for {symbol}")
            logger.debug(f"🔍 AI HUB: main_data keys: {list(main_data.keys()) if main_data else 'None'}")

            # 🚀 PYDANTIC-FIRST: Use fallback to deserialize from main_data since orchestrator reference may not be available
            # Extract advanced metrics from the serialized data
            advanced_metrics = _extract_advanced_metrics(main_data, symbol)

            if not advanced_metrics:
                logger.error(f"❌ AI HUB: No advanced metrics found for {symbol}")
                return _create_empty_metrics_display()

            logger.info(f"✅ AI HUB: Found advanced metrics for {symbol}")
            logger.debug(f"🔍 AI HUB: Advanced metrics type: {type(advanced_metrics)}")

            # Create metrics display
            metrics_display = _create_metrics_display(advanced_metrics, symbol)
            logger.info(f"✅ AI HUB: Created metrics display for {symbol}")

            # Create gauge figures with safe value handling
            lwpai_gauge = _create_gauge_figure(
                value=getattr(advanced_metrics, 'lwpai', 0.0) or 0.0,
                title="LWPAI",
                range_min=-1.0,
                range_max=1.0,
                color_scheme="RdYlGn"
            )

            vabai_gauge = _create_gauge_figure(
                value=getattr(advanced_metrics, 'vabai', 0.0) or 0.0,
                title="VABAI",
                range_min=-1.0,
                range_max=1.0,
                color_scheme="RdYlBu"
            )

            aofm_gauge = _create_gauge_figure(
                value=getattr(advanced_metrics, 'aofm', 0.0) or 0.0,
                title="AOFM",
                range_min=-1.0,
                range_max=1.0,
                color_scheme="Viridis"
            )

            lidb_gauge = _create_gauge_figure(
                value=getattr(advanced_metrics, 'lidb', 0.0) or 0.0,
                title="LIDB",
                range_min=-1.0,
                range_max=1.0,
                color_scheme="Plasma"
            )

            logger.info(f"✅ AI HUB: Created all gauges for {symbol}")
            logger.info(f"🎉 AI HUB: Returning complete metrics display for {symbol}")

            return metrics_display, lwpai_gauge, vabai_gauge, aofm_gauge, lidb_gauge

        except Exception as e:
            logger.error(f"💥 AI HUB CALLBACK ERROR for {symbol}: {e}")
            logger.error(f"💥 AI HUB ERROR TRACEBACK: {e.__class__.__name__}: {str(e)}")
            return _create_error_display(str(e))

    @app.callback(
        Output('ai-hub-learning-status', 'children'),
        [
            Input('main-data-store-id', 'data'),
            Input('interval-live-update-id', 'n_intervals'),
        ],
        prevent_initial_call=False
    )
    def update_ai_learning_status(main_data: Dict, n_intervals: int):
        """Update AI learning status display."""
        try:
            # Get learning status from orchestrator
            if hasattr(orchestrator, 'adaptive_learning_integration'):
                learning_status = orchestrator.adaptive_learning_integration.get_learning_status()
                return _create_learning_status_display(learning_status)
            else:
                return "AI Learning: Not Available"
                
        except Exception as e:
            logger.error(f"Error updating AI learning status: {e}")
            return f"AI Learning: Error - {str(e)}"

def _extract_advanced_metrics_from_bundle(bundle, symbol: str) -> Optional[AdvancedOptionsMetricsV2_5]:
    """🚀 PYDANTIC-FIRST: Extract advanced options metrics directly from live Pydantic bundle."""
    logger.info(f"🔍 BUNDLE EXTRACT: Starting extraction for {symbol}")

    try:
        # Extract advanced metrics from the live Pydantic model
        logger.info(f"🔍 BUNDLE: Accessing processed_data_bundle for {symbol}")
        underlying_data = bundle.processed_data_bundle.underlying_data_enriched
        logger.info(f"🔍 BUNDLE: Got underlying_data for {symbol}")

        # STRICT PYDANTIC V2 ONLY - No dictionary acceptance
        if isinstance(underlying_data, dict):
            logger.error(f"❌ underlying_data is a dictionary instead of Pydantic model for {symbol}")
            raise TypeError("underlying_data must be a Pydantic v2 model, not a dictionary")

        # Ensure underlying_data is a proper Pydantic v2 model
        if not hasattr(underlying_data, 'model_dump'):
            logger.error(f"❌ underlying_data is not a Pydantic v2 model for {symbol}")
            raise TypeError("underlying_data must be a Pydantic v2 model")

        # Access advanced metrics using Pydantic model attributes only
        if hasattr(underlying_data, 'advanced_options_metrics') and underlying_data.advanced_options_metrics:
            logger.info(f"✅ BUNDLE: Found advanced_options_metrics in Pydantic model for {symbol}")

            # Ensure advanced_options_metrics is also a Pydantic model
            if isinstance(underlying_data.advanced_options_metrics, dict):
                logger.error(f"❌ advanced_options_metrics is a dictionary instead of Pydantic model for {symbol}")
                raise TypeError("advanced_options_metrics must be a Pydantic v2 model, not a dictionary")

            if not hasattr(underlying_data.advanced_options_metrics, 'model_dump'):
                logger.error(f"❌ advanced_options_metrics is not a Pydantic v2 model for {symbol}")
                raise TypeError("advanced_options_metrics must be a Pydantic v2 model")

            logger.info(f"✅ BUNDLE: Advanced metrics is proper Pydantic v2 model for {symbol}")
            return underlying_data.advanced_options_metrics

        # Fallback: Check ticker_context as well
        if isinstance(underlying_data, dict):
            if 'ticker_context_dict_v2_5' in underlying_data and underlying_data['ticker_context_dict_v2_5']:
                logger.info(f"🔍 BUNDLE: Checking ticker_context_dict_v2_5 as fallback for {symbol}")
                ticker_context = underlying_data['ticker_context_dict_v2_5']

                # STRICT PYDANTIC V2 ONLY - No dictionary acceptance
                if isinstance(ticker_context, dict):
                    logger.error(f"❌ ticker_context is a dictionary instead of Pydantic model for {symbol}")
                    raise TypeError("ticker_context must be a Pydantic v2 model, not a dictionary")
        else:
            if hasattr(underlying_data, 'ticker_context_dict_v2_5') and underlying_data.ticker_context_dict_v2_5:
                logger.info(f"🔍 BUNDLE: Checking ticker_context_dict_v2_5 as fallback for {symbol}")
                ticker_context = underlying_data.ticker_context_dict_v2_5

                if hasattr(ticker_context, 'advanced_options_metrics') and ticker_context.advanced_options_metrics:
                    logger.info(f"✅ BUNDLE: Found advanced_options_metrics in ticker_context for {symbol}")
                    return ticker_context.advanced_options_metrics

        logger.warning(f"❌ BUNDLE: No advanced options metrics found in bundle for {symbol}")
        return None

    except Exception as e:
        logger.error(f"💥 BUNDLE: Error extracting advanced metrics from bundle for {symbol}: {e}")
        logger.error(f"💥 BUNDLE: Exception type: {type(e).__name__}")
        return None

def _extract_advanced_metrics(main_data: Dict, symbol: str) -> Optional[AdvancedOptionsMetricsV2_5]:
    """LEGACY: Extract advanced options metrics from serialized data store (fallback only)."""
    logger.info(f"🔍 EXTRACTING ADVANCED METRICS for {symbol}")

    try:
        if not main_data:
            logger.error(f"❌ EXTRACT: No main_data provided for {symbol}")
            return None

        logger.info(f"🔍 EXTRACT: main_data structure for {symbol}: {list(main_data.keys())}")

        # 🚀 PYDANTIC-FIRST: Deserialize the main data back to FinalAnalysisBundleV2_5
        from data_models import FinalAnalysisBundleV2_5

        try:
            logger.info(f"🔄 EXTRACT: Attempting to deserialize bundle for {symbol}")
            bundle = FinalAnalysisBundleV2_5.model_validate(main_data)
            logger.info(f"✅ EXTRACT: Successfully deserialized bundle for {symbol}")

            result = _extract_advanced_metrics_from_bundle(bundle, symbol)
            logger.info(f"🎯 EXTRACT: Advanced metrics extraction result for {symbol}: {result is not None}")
            return result

        except Exception as e:
            logger.error(f"❌ EXTRACT: Failed to deserialize bundle for {symbol}: {e}")
            logger.error(f"❌ EXTRACT: Bundle validation error type: {type(e).__name__}")
            return None

    except Exception as e:
        logger.error(f"💥 EXTRACT: Error extracting advanced metrics for {symbol}: {e}")
        return None

def _create_gauge_figure(value: float, title: str, range_min: float, range_max: float, color_scheme: str) -> go.Figure:
    """Create a gauge figure for advanced options metrics."""
    try:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title, 'font': {'size': 16, 'color': 'white'}},
            delta={'reference': 0, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
            gauge={
                'axis': {'range': [range_min, range_max], 'tickcolor': 'white'},
                'bar': {'color': "lightblue"},
                'steps': [
                    {'range': [range_min, range_min + (range_max - range_min) * 0.33], 'color': "red"},
                    {'range': [range_min + (range_max - range_min) * 0.33, range_min + (range_max - range_min) * 0.66], 'color': "yellow"},
                    {'range': [range_min + (range_max - range_min) * 0.66, range_max], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': value
                }
            }
        ))
        
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "white", 'family': "Arial"},
            height=200,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating gauge figure for {title}: {e}")
        return go.Figure()

def _create_metrics_display(metrics: AdvancedOptionsMetricsV2_5, symbol: str) -> List:
    """Create HTML display for advanced options metrics."""
    try:
        from dash import html
        
        return [
            html.H4(f"Advanced Options Metrics - {symbol}", style={'color': 'white', 'textAlign': 'center'}),
            html.Div([
                html.P(f"LWPAI: {metrics.lwpai:.6f}", style={'color': 'cyan'}),
                html.P(f"VABAI: {metrics.vabai:.6f}", style={'color': 'lime'}),
                html.P(f"AOFM: {metrics.aofm:.6f}", style={'color': 'orange'}),
                html.P(f"LIDB: {metrics.lidb:.6f}", style={'color': 'magenta'}),
                html.Hr(style={'borderColor': 'gray'}),
                html.P(f"Confidence: {metrics.confidence_score:.3f}", style={'color': 'yellow'}),
                html.P(f"Contracts Analyzed: {metrics.contracts_analyzed}", style={'color': 'lightblue'}),
                html.P(f"Last Updated: {datetime.now().strftime('%H:%M:%S')}", style={'color': 'gray', 'fontSize': '12px'})
            ], style={'padding': '10px'})
        ]
        
    except Exception as e:
        logger.error(f"Error creating metrics display: {e}")
        return [html.P(f"Error displaying metrics: {str(e)}", style={'color': 'red'})]

def _create_empty_metrics_display() -> tuple:
    """Create empty display when no metrics are available."""
    from dash import html
    
    empty_display = [
        html.H4("Advanced Options Metrics", style={'color': 'white', 'textAlign': 'center'}),
        html.P("No data available", style={'color': 'gray', 'textAlign': 'center'})
    ]
    
    empty_gauge = go.Figure()
    empty_gauge.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=200
    )
    
    return empty_display, empty_gauge, empty_gauge, empty_gauge, empty_gauge

def _create_error_display(error_msg: str) -> tuple:
    """Create error display when metrics update fails."""
    from dash import html
    
    error_display = [
        html.H4("Advanced Options Metrics", style={'color': 'white', 'textAlign': 'center'}),
        html.P(f"Error: {error_msg}", style={'color': 'red', 'textAlign': 'center'})
    ]
    
    error_gauge = go.Figure()
    error_gauge.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=200
    )
    
    return error_display, error_gauge, error_gauge, error_gauge, error_gauge

def _create_learning_status_display(learning_status: Dict) -> str:
    """Create learning status display."""
    try:
        if not learning_status:
            return "AI Learning: Initializing..."
        
        status = learning_status.get('status', 'Unknown')
        last_update = learning_status.get('last_update', 'Never')
        
        return f"AI Learning: {status} | Last Update: {last_update}"
        
    except Exception as e:
        logger.error(f"Error creating learning status display: {e}")
        return f"AI Learning: Error - {str(e)}"

def register_collapsible_callbacks(app):
    """
    Register collapsible callbacks for AI dashboard info sections.
    This function is called by the callback manager.
    """
    try:
        from dash import Input, Output

        @app.callback(
            Output('ai-hub-info-collapse', 'is_open'),
            [Input('ai-hub-info-button', 'n_clicks')],
            prevent_initial_call=True
        )
        def toggle_ai_hub_info(n_clicks):
            """Toggle AI hub info section."""
            if n_clicks:
                return not (n_clicks % 2 == 0)
            return False

        @app.callback(
            Output('advanced-metrics-info-collapse', 'is_open'),
            [Input('advanced-metrics-info-button', 'n_clicks')],
            prevent_initial_call=True
        )
        def toggle_advanced_metrics_info(n_clicks):
            """Toggle advanced metrics info section."""
            if n_clicks:
                return not (n_clicks % 2 == 0)
            return False

        logger.info("✅ AI dashboard collapsible callbacks registered successfully")

    except Exception as e:
        logger.error(f"Error registering collapsible callbacks: {e}")

# Export the registration functions
__all__ = ['register_ai_dashboard_callbacks', 'register_collapsible_callbacks']
