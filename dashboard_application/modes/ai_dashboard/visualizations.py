"""
AI Dashboard Visualizations Module for EOTS v2.5
================================================

This module contains all chart and graph creation functions for the AI dashboard,
including the Legendary Market Compass and various metric gauges. It serves as the
single source for all complex Plotly figures.

Author: EOTS v2.5 Development Team (Refactored)
Version: 2.5.1
"""

import logging
from typing import Dict, Any
import plotly.graph_objects as go
from .components import AI_COLORS

logger = logging.getLogger(__name__)


def create_legendary_market_compass(metrics_data: Dict[str, float], symbol: str = "SPY") -> go.Figure:
    """
    Creates the legendary 12-dimensional Market Compass visualization.
    This function is designed to be the centerpiece of the AI Hub, accepting a
    dictionary of real-time EOTS metrics to generate a multi-layered radar chart.

    Args:
        metrics_data (Dict[str, float]): A dictionary containing the 12 core and custom EOTS metrics.
        symbol (str): The trading symbol for the chart title.

    Returns:
        go.Figure: A Plotly graph object representing the Market Compass.
    """
    try:
        if not metrics_data:
            logger.warning("No metrics data provided for Market Compass. Returning empty figure.")
            return go.Figure()

        # Define the 12 dimensions of the compass for consistent ordering
        compass_dimensions = [
            'VAPI-FA', 'DWFD', 'TW-LAF', 'VRI 2.0', 'A-DAG', 'GIB',
            'LWPAI', 'VABAI', 'AOFM', 'LIDB', 'SVR', 'TPDLF'
        ]

        # --- Multi-Timeframe Layers ---
        # In a production system, you would pass real multi-timeframe data.
        # Here, we simulate it to maintain the visual layered effect.
        timeframes = ['5m', '15m', '1h', '4h']
        timeframe_colors = [
            'rgba(66, 165, 245, 0.8)', 'rgba(107, 207, 127, 0.6)',
            'rgba(255, 167, 38, 0.4)', 'rgba(255, 71, 87, 0.3)'
        ]

        fig = go.Figure()

        for i, (timeframe, color) in enumerate(zip(timeframes, timeframe_colors)):
            # Simulate variation across timeframes. Replace with real data if available.
            variation = 1.0 - (i * 0.15)
            # Use .get(dim, 0.0) to safely handle any missing metrics
            values = [metrics_data.get(dim, 0.0) * variation for dim in compass_dimensions]

            # The last point must be the same as the first to close the shape
            theta = compass_dimensions + [compass_dimensions[0]]
            r = values + [values[0]]

            fig.add_trace(go.Scatterpolar(
                r=r,
                theta=theta,
                fill='toself',
                fillcolor=color,
                line=dict(color=color.replace('rgba', 'rgb').replace(', 0.', ', 1.'), width=2),
                name=f'{timeframe} Timeframe',
                hovertemplate=f'<b>%{{theta}}</b><br>Value: %{{r:.2f}}<br>Timeframe: {timeframe}<extra></extra>'
            ))

        # --- Extreme Reading Indicators ---
        # Highlight metrics that are showing extreme values (> 2.0 standard deviations)
        extreme_metrics = {name: value for name, value in metrics_data.items() if abs(value) > 2.0}
        for name, value in extreme_metrics.items():
            fig.add_trace(go.Scatterpolar(
                r=[abs(value) * 1.1], # Position the star just outside the reading
                theta=[name],
                mode='markers',
                marker=dict(
                    size=15,
                    color=AI_COLORS['danger'],
                    symbol='star',
                    line=dict(width=2, color='white')
                ),
                name=f'Extreme: {name}',
                showlegend=False,
                hovertemplate=f'<b>EXTREME READING</b><br>{name}: {value:.2f}<extra></extra>'
            ))

        # --- Layout and Styling ---
        fig.update_layout(
            polar=dict(
                bgcolor='rgba(0, 0, 0, 0.1)',
                radialaxis=dict(
                    visible=True,
                    range=[-3, 3],  # Set a fixed range for consistency
                    tickfont=dict(size=10, color='white'),
                    gridcolor='rgba(255, 255, 255, 0.2)',
                    linecolor='rgba(255, 255, 255, 0.3)'
                ),
                angularaxis=dict(
                    tickfont=dict(size=12, color='white', family='Arial Black'),
                    gridcolor='rgba(255, 255, 255, 0.3)',
                    linecolor='rgba(255, 255, 255, 0.5)'
                )
            ),
            title=dict(
                text=f'🧭 LEGENDARY MARKET COMPASS - {symbol.upper()}',
                x=0.5, y=0.95, font=dict(size=20, color='white', family='Arial Black')
            ),
            paper_bgcolor='rgba(15, 23, 42, 0.95)',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            font=dict(color='white'),
            height=500,
            margin=dict(l=50, r=50, t=80, b=50),
            legend=dict(
                orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5,
                font=dict(color='white', size=10)
            )
        )
        return fig

    except Exception as e:
        logger.error(f"Error creating Legendary Market Compass: {e}", exc_info=True)
        # Return a styled empty figure on error
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor='rgba(15, 23, 42, 0.95)', plot_bgcolor='rgba(0, 0, 0, 0)',
            height=500,
            annotations=[dict(text="Compass Unavailable", showarrow=False,
                              font=dict(color=AI_COLORS['danger'], size=16))]
        )
        return fig


def create_metric_gauge(value: float, title: str, range_min: float = -3.0, range_max: float = 3.0) -> go.Figure:
    """
    Creates a standardized, styled gauge for displaying a single metric.

    Args:
        value (float): The metric value to display.
        title (str): The title of the gauge.
        range_min (float): The minimum value for the gauge axis.
        range_max (float): The maximum value for the gauge axis.

    Returns:
        go.Figure: A Plotly graph object representing the gauge.
    """
    try:
        # Determine color based on how extreme the value is
        if abs(value) >= 2.0: color = AI_COLORS['danger']
        elif abs(value) >= 1.5: color = AI_COLORS['warning']
        elif abs(value) >= 1.0: color = AI_COLORS['primary']
        else: color = AI_COLORS['success']

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title, 'font': {'size': 12, 'color': 'white'}},
            number={'font': {'size': 16, 'color': color}},
            gauge={
                'axis': {'range': [range_min, range_max], 'tickcolor': 'white', 'tickfont': {'size': 8}},
                'bar': {'color': color, 'thickness': 0.7},
                'bgcolor': 'rgba(0, 0, 0, 0.1)',
                'borderwidth': 1,
                'bordercolor': 'rgba(255, 255, 255, 0.3)',
                'steps': [
                    {'range': [range_min, -1], 'color': 'rgba(255, 71, 87, 0.2)'},
                    {'range': [-1, 1], 'color': 'rgba(107, 207, 127, 0.2)'},
                    {'range': [1, range_max], 'color': 'rgba(255, 71, 87, 0.2)'}
                ]
            }
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0, 0, 0, 0)', plot_bgcolor='rgba(0, 0, 0, 0)',
            font={'color': 'white'}, height=120, margin=dict(l=10, r=10, t=30, b=10)
        )
        return fig
    except Exception as e:
        logger.error(f"Error creating metric gauge for {title}: {e}", exc_info=True)
        return go.Figure()


def create_regime_transition_gauge(transition_prob: float, regime_confidence: float) -> go.Figure:
    """
    Creates a gauge visualizing the probability of a market regime transition.

    Args:
        transition_prob (float): The probability of a regime change (0 to 1).
        regime_confidence (float): The confidence in the current regime (0 to 1).

    Returns:
        go.Figure: A Plotly graph object representing the transition gauge.
    """
    try:
        if transition_prob >= 0.7: gauge_color, gauge_level = AI_COLORS['danger'], "High Risk"
        elif transition_prob >= 0.4: gauge_color, gauge_level = AI_COLORS['warning'], "Moderate Risk"
        else: gauge_color, gauge_level = AI_COLORS['success'], "Low Risk"

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=transition_prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Transition Risk: {gauge_level}", 'font': {'size': 12, 'color': 'white'}},
            number={'font': {'size': 18, 'color': gauge_color}},
            gauge={
                'axis': {'range': [None, 100], 'tickcolor': 'white', 'tickfont': {'size': 9}},
                'bar': {'color': gauge_color, 'thickness': 0.7},
                'steps': [
                    {'range': [0, 40], 'color': 'rgba(107, 207, 127, 0.2)'},
                    {'range': [40, 70], 'color': 'rgba(255, 167, 38, 0.2)'},
                    {'range': [70, 100], 'color': 'rgba(255, 71, 87, 0.2)'}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 3},
                    'thickness': 0.75,
                    'value': 70 # High risk threshold
                }
            }
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0, 0, 0, 0)', plot_bgcolor='rgba(0, 0, 0, 0)',
            font={'color': 'white'}, height=150, margin=dict(l=20, r=20, t=40, b=20)
        )
        return fig
    except Exception as e:
        logger.error(f"Error creating regime transition gauge: {e}", exc_info=True)
        return go.Figure()