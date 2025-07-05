"""
AI Dashboard Components Module for EOTS v2.5
============================================

This module contains all reusable UI components for the AI dashboard including:
- Cards and containers
- Badges and indicators
- Meters and gauges
- Buttons and controls
- Styling utilities

Author: EOTS v2.5 Development Team
Version: 2.5.0
"""

import logging
from typing import Dict

from dash import html, Input, Output, State

from data_models import FinalAnalysisBundleV2_5

# Import regime display utilities
from dashboard_application.utils.regime_display_utils import get_tactical_regime_name

logger = logging.getLogger(__name__)

# ===== AI DASHBOARD STYLING CONSTANTS =====
# Exact styling from original to maintain visual consistency

AI_COLORS = {
    'primary': '#00d4ff',      # Electric Blue - Main brand color
    'secondary': '#ffd93d',    # Golden Yellow - Secondary highlights
    'accent': '#ff6b6b',       # Coral Red - Alerts and warnings
    'success': '#6bcf7f',      # Green - Positive values
    'danger': '#ff4757',       # Red - Negative values
    'warning': '#ffa726',      # Orange - Caution
    'info': '#42a5f5',         # Light Blue - Information
    'dark': '#ffffff',         # White text for dark theme
    'light': 'rgba(255, 255, 255, 0.1)',  # Light overlay for dark theme
    'muted': 'rgba(255, 255, 255, 0.6)',  # Muted white text
    'card_bg': 'rgba(255, 255, 255, 0.05)', # Dark card background
    'card_border': 'rgba(255, 255, 255, 0.1)' # Subtle border
}

AI_TYPOGRAPHY = {
    'title_size': '1.5rem',
    'subtitle_size': '1.2rem',
    'body_size': '0.9rem',
    'small_size': '0.8rem',
    'tiny_size': '0.7rem',
    'title_weight': '600',
    'subtitle_weight': '500',
    'body_weight': '400'
}

AI_SPACING = {
    'xs': '4px',
    'sm': '8px',
    'md': '12px',
    'lg': '16px',
    'xl': '24px',
    'xxl': '32px'
}

AI_EFFECTS = {
    'card_shadow': '0 8px 32px rgba(0, 0, 0, 0.3)',
    'card_shadow_hover': '0 12px 48px rgba(0, 0, 0, 0.4)',
    'box_shadow': '0 8px 32px rgba(0, 212, 255, 0.1)',
    'shadow': '0 4px 16px rgba(0, 0, 0, 0.2)',
    'shadow_lg': '0 8px 32px rgba(0, 0, 0, 0.3)',
    'border_radius': '16px',
    'border_radius_sm': '8px',
    'backdrop_blur': 'blur(20px)',
    'transition': 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
    'gradient_bg': 'linear-gradient(135deg, rgba(0, 0, 0, 0.8) 0%, rgba(20, 20, 20, 0.9) 50%, rgba(0, 0, 0, 0.8) 100%)',
    'glass_bg': 'rgba(0, 0, 0, 0.4)',
    'glass_border': '1px solid rgba(255, 255, 255, 0.1)'
}

# ===== CARD STYLING FUNCTIONS =====

def get_card_style(variant='default'):
    """Get unified card styling matching AI Performance Tracker aesthetic."""

    # Base style matching AI Performance Tracker
    if variant == 'analysis' or variant == 'primary':
        return {
            'background': 'linear-gradient(145deg, #1e1e2e, #2a2a3e)',
            'border': '1px solid #00d4ff',
            'borderRadius': '15px',
            'boxShadow': '0 8px 32px rgba(0, 212, 255, 0.1)',
            'padding': '20px',
            'marginBottom': '20px',
            'transition': 'all 0.3s ease',
            'color': '#ffffff'
        }
    elif variant == 'recommendations' or variant == 'secondary':
        return {
            'background': 'linear-gradient(145deg, #2e1e1e, #3e2a2a)',
            'border': '1px solid #ffd93d',
            'borderRadius': '15px',
            'boxShadow': '0 8px 32px rgba(255, 217, 61, 0.1)',
            'padding': '20px',
            'marginBottom': '20px',
            'transition': 'all 0.3s ease',
            'color': '#ffffff'
        }
    elif variant == 'regime' or variant == 'success':
        return {
            'background': 'linear-gradient(145deg, #1e2e1e, #2a3e2a)',
            'border': '1px solid #6bcf7f',
            'borderRadius': '15px',
            'boxShadow': '0 8px 32px rgba(107, 207, 127, 0.1)',
            'padding': '20px',
            'marginBottom': '20px',
            'transition': 'all 0.3s ease',
            'color': '#ffffff'
        }
    elif variant == 'insights' or variant == 'warning':
        return {
            'background': 'linear-gradient(145deg, #2e2e1e, #3e3e2a)',
            'border': '1px solid #ffd93d',
            'borderRadius': '15px',
            'boxShadow': '0 8px 32px rgba(255, 217, 61, 0.1)',
            'padding': '20px',
            'marginBottom': '20px',
            'transition': 'all 0.3s ease',
            'color': '#ffffff'
        }
    elif variant == 'performance':
        return {
            'background': 'linear-gradient(145deg, #1e2e2e, #2a3e3e)',
            'border': '1px solid #6bcf7f',
            'borderRadius': '15px',
            'boxShadow': '0 8px 32px rgba(107, 207, 127, 0.1)',
            'padding': '20px',
            'marginBottom': '20px',
            'transition': 'all 0.3s ease',
            'color': '#ffffff'
        }
    elif variant == 'accent':
        return {
            'background': 'linear-gradient(145deg, #2e1e2e, #3e2a3e)',
            'border': '1px solid #ff6b6b',
            'borderRadius': '15px',
            'boxShadow': '0 8px 32px rgba(255, 107, 107, 0.1)',
            'padding': '20px',
            'marginBottom': '20px',
            'transition': 'all 0.3s ease',
            'color': '#ffffff'
        }
    else:  # default
        return {
            'background': 'linear-gradient(145deg, #1e1e2e, #2a2a3e)',
            'border': '1px solid #00d4ff',
            'borderRadius': '15px',
            'boxShadow': '0 8px 32px rgba(0, 212, 255, 0.1)',
            'padding': '20px',
            'marginBottom': '20px',
            'transition': 'all 0.3s ease',
            'color': '#ffffff'
        }

# ===== CORE UI COMPONENTS =====

def create_placeholder_card(title: str, message: str) -> html.Div:
    """Create a placeholder card for components that aren't available."""
    return html.Div([
        html.Div([
            html.H4(title, className="card-title mb-3", style={
                "color": AI_COLORS['dark'],
                "fontSize": AI_TYPOGRAPHY['title_size'],
                "fontWeight": AI_TYPOGRAPHY['title_weight']
            }),
            html.P(message, className="text-muted", style={
                "color": AI_COLORS['muted'],
                "fontSize": AI_TYPOGRAPHY['body_size'],
                "marginBottom": "0",
                "lineHeight": "1.5"
            })
        ], style=get_card_style('default'))
    ], className="ai-placeholder-card")


def create_enhanced_confidence_meter(confidence: float, bundle_data: FinalAnalysisBundleV2_5) -> html.Div:
    """Create enhanced confidence meter with real-time data integration."""
    try:
        # Determine confidence level and styling
        if confidence >= 0.8:
            confidence_level = "High"
            color = AI_COLORS['success']
            bg_color = "rgba(107, 207, 127, 0.1)"
        elif confidence >= 0.6:
            confidence_level = "Moderate"
            color = AI_COLORS['warning']
            bg_color = "rgba(255, 167, 38, 0.1)"
        else:
            confidence_level = "Low"
            color = AI_COLORS['danger']
            bg_color = "rgba(255, 71, 87, 0.1)"

        # Extract additional context from bundle
        symbol = bundle_data.target_symbol
        timestamp = bundle_data.bundle_timestamp

        return html.Div([
            html.Div([
                html.H6("🎯 AI Confidence", className="mb-2", style={
                    "color": AI_COLORS['dark'],
                    "fontSize": AI_TYPOGRAPHY['subtitle_size'],
                    "fontWeight": AI_TYPOGRAPHY['subtitle_weight']
                }),
                html.Div([
                    html.Div([
                        html.Div(style={
                            "width": f"{confidence * 100}%",
                            "height": "8px",
                            "background": color,
                            "borderRadius": "4px",
                            "transition": AI_EFFECTS['transition']
                        })
                    ], style={
                        "width": "100%",
                        "height": "8px",
                        "background": "rgba(255, 255, 255, 0.1)",
                        "borderRadius": "4px",
                        "marginBottom": AI_SPACING['sm']
                    }),
                    html.Div([
                        html.Span(f"{confidence:.0%}", style={
                            "fontSize": AI_TYPOGRAPHY['title_size'],
                            "fontWeight": AI_TYPOGRAPHY['title_weight'],
                            "color": color
                        }),
                        html.Span(f" {confidence_level}", style={
                            "fontSize": AI_TYPOGRAPHY['body_size'],
                            "color": AI_COLORS['muted'],
                            "marginLeft": AI_SPACING['sm']
                        })
                    ], className="d-flex align-items-center"),
                    html.Small(f"Updated: {timestamp.strftime('%H:%M:%S')}", style={
                        "color": AI_COLORS['muted'],
                        "fontSize": AI_TYPOGRAPHY['small_size']
                    })
                ])
            ], style={
                "padding": f"{AI_SPACING['md']} {AI_SPACING['lg']}",
                "background": bg_color,
                "borderRadius": AI_EFFECTS['border_radius'],
                "border": f"1px solid {color}",
                "transition": AI_EFFECTS['transition']
            })
        ], className="confidence-meter")

    except Exception as e:
        logger.error(f"Error creating confidence meter: {str(e)}")
        return create_placeholder_card("🎯 AI Confidence", f"Error: {str(e)}")


def create_recommendation_confidence_bar(confidence: float) -> html.Div:
    """Create a recommendation confidence bar visualization."""
    try:
        # Determine color based on confidence level
        if confidence >= 0.8:
            color = AI_COLORS['success']
            label = "High Confidence"
        elif confidence >= 0.6:
            color = AI_COLORS['warning']
            label = "Moderate Confidence"
        else:
            color = AI_COLORS['danger']
            label = "Low Confidence"

        return html.Div([
            html.Div([
                html.Span(f"{confidence:.0%}", style={
                    "fontSize": AI_TYPOGRAPHY['body_size'],
                    "fontWeight": AI_TYPOGRAPHY['subtitle_weight'],
                    "color": color
                }),
                html.Span(f" {label}", style={
                    "fontSize": AI_TYPOGRAPHY['small_size'],
                    "color": AI_COLORS['muted'],
                    "marginLeft": AI_SPACING['xs']
                })
            ], className="mb-1"),
            html.Div([
                html.Div(style={
                    "width": f"{confidence * 100}%",
                    "height": "4px",
                    "background": color,
                    "borderRadius": "2px",
                    "transition": AI_EFFECTS['transition']
                })
            ], style={
                "width": "100%",
                "height": "4px",
                "background": "rgba(255, 255, 255, 0.1)",
                "borderRadius": "2px"
            })
        ], className="recommendation-confidence-bar")

    except Exception as e:
        logger.error(f"Error creating confidence bar: {str(e)}")
        return html.Div("Error creating confidence bar")


def create_enhanced_recommendation_item(index: int, strategy: str, conviction: float, rationale: str) -> html.Div:
    """Create enhanced recommendation item with improved styling."""
    try:
        # Determine conviction styling
        if conviction > 0.7:
            badge_class = "success"
            border_color = AI_COLORS['success']
        elif conviction > 0.5:
            badge_class = "warning"
            border_color = AI_COLORS['warning']
        else:
            badge_class = "secondary"
            border_color = AI_COLORS['secondary']

        return html.Div([
            html.Div([
                html.H6(f"🎯 #{index+1}: {strategy}", className="mb-2", style={
                    "fontSize": AI_TYPOGRAPHY['body_size'],
                    "fontWeight": AI_TYPOGRAPHY['subtitle_weight'],
                    "color": AI_COLORS['dark']
                }),
                html.Div([
                    html.Span("Conviction: ", className="fw-bold", style={
                        "fontSize": AI_TYPOGRAPHY['small_size'],
                        "color": AI_COLORS['dark']
                    }),
                    html.Span(f"{conviction:.1%}",
                            className=f"badge bg-{badge_class}",
                            style={"fontSize": AI_TYPOGRAPHY['small_size']})
                ], className="mb-2"),
                html.P(rationale[:120] + "..." if len(rationale) > 120 else rationale,
                       className="small text-muted", style={
                           "fontSize": AI_TYPOGRAPHY['small_size'],
                           "lineHeight": "1.3",
                           "color": AI_COLORS['muted'],
                           "marginBottom": "0"
                       })
            ])
        ], className="recommendation-item p-2 mb-2", style={
            "background": f"rgba({border_color[1:3]}, {border_color[3:5]}, {border_color[5:7]}, 0.1)",
            "borderRadius": AI_EFFECTS['border_radius'],
            "border": f"1px solid rgba({border_color[1:3]}, {border_color[3:5]}, {border_color[5:7]}, 0.3)",
            "transition": AI_EFFECTS['transition'],
            "cursor": "pointer"
        })

    except Exception as e:
        logger.error(f"Error creating recommendation item: {str(e)}")
        return html.Div("Error creating recommendation item")


def create_enhanced_insight_item(insight: str, insight_type: str = "info") -> html.Div:
    """Create enhanced insight item with type-based styling."""
    try:
        # Determine styling based on insight type
        type_config = {
            "success": {"color": AI_COLORS['success'], "icon": "✅"},
            "warning": {"color": AI_COLORS['warning'], "icon": "⚠️"},
            "danger": {"color": AI_COLORS['danger'], "icon": "🚨"},
            "info": {"color": AI_COLORS['info'], "icon": "ℹ️"},
            "primary": {"color": AI_COLORS['primary'], "icon": "🎯"}
        }
        
        config = type_config.get(insight_type, type_config["info"])
        
        return html.Div([
            html.Div([
                html.Span(config["icon"], style={
                    "marginRight": AI_SPACING['sm'],
                    "fontSize": AI_TYPOGRAPHY['body_size']
                }),
                html.Span(insight, style={
                    "fontSize": AI_TYPOGRAPHY['small_size'],
                    "color": AI_COLORS['dark'],
                    "lineHeight": "1.4"
                })
            ])
        ], className="insight-item p-2 mb-2", style={
            "background": f"rgba({config['color'][1:3]}, {config['color'][3:5]}, {config['color'][5:7]}, 0.1)",
            "borderRadius": AI_EFFECTS['border_radius'],
            "border": f"1px solid rgba({config['color'][1:3]}, {config['color'][3:5]}, {config['color'][5:7]}, 0.2)",
            "transition": AI_EFFECTS['transition']
        })

    except Exception as e:
        logger.error(f"Error creating insight item: {str(e)}")
        return html.Div("Error creating insight item")


def create_quick_action_buttons(bundle_data: FinalAnalysisBundleV2_5, symbol: str) -> html.Div:
    """Create quick action buttons for AI dashboard."""
    try:
        return html.Div([
            html.Div([
                html.Button([
                    html.I(className="fas fa-refresh me-2"),
                    "Refresh Analysis"
                ], className="btn btn-outline-primary btn-sm me-2", style={
                    "borderColor": AI_COLORS['primary'],
                    "color": AI_COLORS['primary'],
                    "fontSize": AI_TYPOGRAPHY['small_size']
                }),
                html.Button([
                    html.I(className="fas fa-download me-2"),
                    "Export Data"
                ], className="btn btn-outline-secondary btn-sm me-2", style={
                    "borderColor": AI_COLORS['secondary'],
                    "color": AI_COLORS['secondary'],
                    "fontSize": AI_TYPOGRAPHY['small_size']
                }),
                html.Button([
                    html.I(className="fas fa-cog me-2"),
                    "Settings"
                ], className="btn btn-outline-info btn-sm", style={
                    "borderColor": AI_COLORS['info'],
                    "color": AI_COLORS['info'],
                    "fontSize": AI_TYPOGRAPHY['small_size']
                })
            ], className="d-flex flex-wrap gap-2")
        ], className="quick-actions mb-3")

    except Exception as e:
        logger.error(f"Error creating quick action buttons: {str(e)}")
        return html.Div("Error creating action buttons")


def create_regime_transition_indicator(current_regime: str, transition_prob: float) -> html.Div:
    """Create regime transition probability indicator."""
    try:
        # Determine transition risk level
        if transition_prob >= 0.7:
            risk_level = "High"
            color = AI_COLORS['danger']
            icon = "🚨"
        elif transition_prob >= 0.4:
            risk_level = "Moderate"
            color = AI_COLORS['warning']
            icon = "⚠️"
        else:
            risk_level = "Low"
            color = AI_COLORS['success']
            icon = "✅"

        return html.Div([
            html.Div([
                html.Span(icon, style={"marginRight": AI_SPACING['sm']}),
                html.Span("Regime Transition Risk: ", style={
                    "fontSize": AI_TYPOGRAPHY['small_size'],
                    "fontWeight": AI_TYPOGRAPHY['subtitle_weight'],
                    "color": AI_COLORS['dark']
                }),
                html.Span(f"{transition_prob:.0%} ({risk_level})", style={
                    "fontSize": AI_TYPOGRAPHY['small_size'],
                    "color": color,
                    "fontWeight": AI_TYPOGRAPHY['subtitle_weight']
                })
            ], className="mb-2"),
            html.Div([
                html.Small(f"Current: {get_tactical_regime_name(current_regime)}", style={
                    "color": AI_COLORS['muted'],
                    "fontSize": AI_TYPOGRAPHY['small_size']
                })
            ])
        ], className="regime-transition-indicator p-2", style={
            "background": f"rgba({color[1:3]}, {color[3:5]}, {color[5:7]}, 0.1)",
            "borderRadius": AI_EFFECTS['border_radius'],
            "border": f"1px solid rgba({color[1:3]}, {color[3:5]}, {color[5:7]}, 0.2)"
        })

    except Exception as e:
        logger.error(f"Error creating regime transition indicator: {str(e)}")
        return html.Div("Error creating transition indicator")


# ===== STYLING UTILITY FUNCTIONS =====

def get_unified_badge_style(badge_type: str) -> Dict[str, str]:
    """Get unified badge styling for consistency across components."""
    base_style = {
        "fontSize": AI_TYPOGRAPHY['small_size'],
        "fontWeight": AI_TYPOGRAPHY['subtitle_weight'],
        "padding": f"{AI_SPACING['xs']} {AI_SPACING['sm']}",
        "borderRadius": "4px",
        "border": "none"
    }

    type_styles = {
        "primary": {
            "background": AI_COLORS['primary'],
            "color": AI_COLORS['dark']
        },
        "secondary": {
            "background": AI_COLORS['secondary'],
            "color": AI_COLORS['light']
        },
        "success": {
            "background": AI_COLORS['success'],
            "color": AI_COLORS['light']
        },
        "danger": {
            "background": AI_COLORS['danger'],
            "color": AI_COLORS['light']
        },
        "warning": {
            "background": AI_COLORS['warning'],
            "color": AI_COLORS['dark']
        },
        "info": {
            "background": AI_COLORS['info'],
            "color": AI_COLORS['light']
        }
    }

    return {**base_style, **type_styles.get(badge_type, type_styles["secondary"])}


def get_unified_card_style(card_type: str = "default") -> Dict[str, str]:
    """Get unified card styling for consistency across components."""
    base_style = {
        "background": AI_COLORS['card_bg'],
        "border": f"1px solid {AI_COLORS['border']}",
        "borderRadius": AI_EFFECTS['border_radius'],
        "boxShadow": AI_EFFECTS['shadow'],
        "transition": AI_EFFECTS['transition']
    }

    type_styles = {
        "elevated": {
            "boxShadow": AI_EFFECTS['shadow_lg'],
            "transform": "translateY(-2px)"
        },
        "highlighted": {
            "border": f"2px solid {AI_COLORS['primary']}",
            "boxShadow": "0 0 20px rgba(255, 217, 61, 0.3)"
        },
        "danger": {
            "border": f"2px solid {AI_COLORS['danger']}",
            "boxShadow": "0 0 20px rgba(255, 71, 87, 0.2)"
        }
    }

    return {**base_style, **type_styles.get(card_type, {})}


def get_unified_text_style(text_type: str) -> Dict[str, str]:
    """Get unified text styling for consistency across components."""
    styles = {
        "title": {
            "fontSize": AI_TYPOGRAPHY['title_size'],
            "fontWeight": AI_TYPOGRAPHY['title_weight'],
            "color": AI_COLORS['dark'],
            "marginBottom": AI_SPACING['sm']
        },
        "subtitle": {
            "fontSize": AI_TYPOGRAPHY['subtitle_size'],
            "fontWeight": AI_TYPOGRAPHY['subtitle_weight'],
            "color": AI_COLORS['dark'],
            "marginBottom": AI_SPACING['xs']
        },
        "body": {
            "fontSize": AI_TYPOGRAPHY['body_size'],
            "color": AI_COLORS['dark'],
            "lineHeight": "1.5"
        },
        "muted": {
            "fontSize": AI_TYPOGRAPHY['small_size'],
            "color": AI_COLORS['muted'],
            "lineHeight": "1.4"
        },
        "small": {
            "fontSize": AI_TYPOGRAPHY['small_size'],
            "color": AI_COLORS['dark'],
            "lineHeight": "1.3"
        }
    }

    return styles.get(text_type, styles["body"])


def get_sentiment_color(sentiment_score: float) -> str:
    """Get color based on sentiment score."""
    if sentiment_score > 0.1:
        return AI_COLORS['success']
    elif sentiment_score < -0.1:
        return AI_COLORS['danger']
    else:
        return AI_COLORS['warning']


# ===== COLLAPSIBLE INFO COMPONENTS =====

def create_collapsible_info_section(info_id: str, info_content: str, is_open: bool = False) -> html.Div:
    """Create a collapsible information section that can be toggled by clicking."""
    return html.Div([
        html.Div([
            html.P([
                info_content
            ], style={
                "fontSize": AI_TYPOGRAPHY['small_size'],
                "lineHeight": "1.6",
                "color": AI_COLORS['dark'],
                "margin": "0",
                "padding": f"{AI_SPACING['md']} {AI_SPACING['lg']}",
                "background": "rgba(255, 255, 255, 0.05)",
                "borderRadius": AI_EFFECTS['border_radius_sm'],
                "border": "1px solid rgba(255, 255, 255, 0.1)",
                "fontFamily": "'Inter', -apple-system, BlinkMacSystemFont, sans-serif"
            })
        ], style={
            "marginTop": AI_SPACING['sm'],
            "transition": "all 0.3s ease-in-out"
        })
    ], id=f"collapse-{info_id}", style={"display": "block" if is_open else "none"})


def create_clickable_title_with_info(title: str, info_id: str, info_content: str,
                                   title_style: Dict[str, str] = None,
                                   badge_text: str = None,
                                   badge_style: str = 'success') -> html.Div:
    """Create a clickable title that toggles an information section using HTML details/summary."""

    default_title_style = {
        "color": AI_COLORS['dark'],
        "fontSize": AI_TYPOGRAPHY['title_size'],
        "fontWeight": AI_TYPOGRAPHY['title_weight'],
        "margin": "0",
        "cursor": "pointer",
        "userSelect": "none",
        "transition": AI_EFFECTS['transition']
    }

    if title_style:
        default_title_style.update(title_style)

    # Create the summary content (clickable title)
    summary_content = [title]

    if badge_text:
        summary_content.append(" ")
        summary_content.append(
            html.Span(badge_text, className="badge", style=get_unified_badge_style(badge_style))
        )

    return html.Details([
        # Summary (clickable title)
        html.Summary([
            html.H5(summary_content, className="mb-0", style=default_title_style)
        ], style={
            "cursor": "pointer",
            "listStyle": "none",
            "outline": "none"
        }),

        # Collapsible content
        html.Div([
            html.P([
                info_content
            ], style={
                "fontSize": AI_TYPOGRAPHY['small_size'],
                "lineHeight": "1.6",
                "color": AI_COLORS['dark'],
                "margin": "0",
                "padding": f"{AI_SPACING['md']} {AI_SPACING['lg']}",
                "background": "rgba(255, 255, 255, 0.05)",
                "borderRadius": AI_EFFECTS['border_radius_sm'],
                "border": "1px solid rgba(255, 255, 255, 0.1)",
                "fontFamily": "'Inter', -apple-system, BlinkMacSystemFont, sans-serif"
            })
        ], style={
            "marginTop": AI_SPACING['sm'],
            "animation": "fadeIn 0.3s ease-in-out"
        })
    ], style={
        "border": "none",
        "padding": "0",
        "margin": "0"
    })


def create_simple_clickable_title(title: str, info_id: str, badge_text: str = None, badge_style: str = 'success') -> html.Div:
    """Create a simple clickable title without the info content (for use with external callbacks)."""
    title_style = {
        "color": AI_COLORS['dark'],
        "fontSize": AI_TYPOGRAPHY['title_size'],
        "fontWeight": AI_TYPOGRAPHY['title_weight'],
        "cursor": "pointer",
        "userSelect": "none",
        "transition": AI_EFFECTS['transition']
    }

    # Add visual indicator that title is clickable
    title_with_indicator = f"{title} ℹ️"

    title_content = [
        html.H5(title_with_indicator, className="mb-0", style=title_style)
    ]

    if badge_text:
        title_content.append(
            html.Span(badge_text, className="badge", style=get_unified_badge_style(badge_style))
        )

    return html.Div(
        title_content,
        className="d-flex justify-content-between align-items-center",
        style={"cursor": "pointer"},
        id=f"title-button-{info_id}",
        n_clicks=0
    )


# ===== COLLAPSIBLE INFO CALLBACKS (Consolidated from callbacks.py) =====

# AI Module Information Blurbs
AI_MODULE_INFO = {
    "unified_intelligence": """🧠 Unified AI Intelligence Hub: This is your COMMAND CENTER for all AI-powered market analysis. The 4-quadrant layout provides: TOP-LEFT: AI Confidence Barometer showing system conviction levels with real-time data quality scoring. TOP-RIGHT: Signal Confluence Barometer measuring agreement between multiple EOTS metrics (VAPI-FA, DWFD, TW-LAF, GIB). BOTTOM-LEFT: Unified Intelligence Analysis combining Alpha Vantage news sentiment, MCP server insights, and ATIF recommendations. BOTTOM-RIGHT: Market Dynamics Radar showing 6-dimensional market forces (Volatility, Flow, Momentum, Structure, Sentiment, Risk). 💡 TRADING INSIGHT: When AI Confidence > 80% AND Signal Confluence > 70% = HIGH CONVICTION setup. Watch for Market Dynamics radar showing EXTREME readings (outer edges) = potential breakout/breakdown. The Unified Intelligence text provides CONTEXTUAL NARRATIVE explaining WHY the system is confident. This updates every 15 minutes with fresh data integration!""",

    "regime_analysis": """🌊 AI Regime Analysis: This 4-quadrant system identifies and analyzes the CURRENT MARKET REGIME using advanced EOTS metrics. TOP-LEFT: Regime Confidence Barometer showing conviction in current regime classification with transition risk assessment. TOP-RIGHT: Regime Characteristics Analysis displaying 4 key market properties (Volatility, Flow Direction, Risk Level, Momentum) with DYNAMIC COLOR CODING. BOTTOM-LEFT: Enhanced AI Analysis showing current regime name, key Z-score metrics (VAPI-FA, DWFD, TW-LAF), and AI-generated insights. BOTTOM-RIGHT: Transition Gauge measuring probability of regime change with stability metrics. 💡 TRADING INSIGHT: Regime Confidence > 70% = STABLE regime, trade WITH the characteristics. Transition Risk > 60% = UNSTABLE regime, expect volatility and potential reversals. When characteristics show EXTREME values (Very High/Low) = regime at INFLECTION POINT. Use regime insights to adjust position sizing and strategy selection!""",

    "raw_metrics": """🔢 Raw EOTS Metrics Dashboard: This displays the CORE EOTS v2.5 metrics in their purest form, validated against Pydantic schemas. Shows real-time Z-scores for: VAPI-FA (Volume-Adjusted Put/Call Imbalance with Flow Alignment), DWFD (Delta-Weighted Flow Direction), TW-LAF (Time-Weighted Liquidity-Adjusted Flow), GIB (Gamma Imbalance Barometer), and underlying price/volume data. Each metric is STANDARDIZED to Z-scores for easy comparison. 💡 TRADING INSIGHT: Z-scores > +2.0 = EXTREMELY BULLISH signal. Z-scores < -2.0 = EXTREMELY BEARISH signal. Z-scores between -1.0 and +1.0 = NEUTRAL/CONSOLIDATION. When MULTIPLE metrics show same direction (all positive or all negative) = HIGH CONVICTION directional bias. DIVERGENCE between metrics = UNCERTAINTY, potential for volatility. These are the RAW BUILDING BLOCKS that feed into all other AI analysis!""",

    "recommendations": """🎯 AI Recommendations Engine: This panel displays ADAPTIVE TRADE IDEA FRAMEWORK (ATIF) generated strategies with AI-enhanced conviction scoring. Each recommendation includes: Strategy Type, Conviction Level (0-100%), AI-generated Rationale, and Risk Assessment. The system combines EOTS metrics, regime analysis, and market structure to generate ACTIONABLE trade ideas. 💡 TRADING INSIGHT: Conviction > 80% = HIGH PROBABILITY setup, consider larger position size. Conviction 60-80% = MODERATE setup, standard position size. Conviction < 60% = LOW PROBABILITY, small position or avoid. When multiple recommendations AGREE on direction = STRONG CONFLUENCE. Pay attention to the AI rationale - it explains the LOGIC behind each recommendation. Recommendations update based on changing market conditions and new data!""",

    "learning_center": """📚 AI Learning Center: This tracks the system's ADAPTIVE LEARNING capabilities and pattern recognition evolution. Displays: Learning Velocity (how fast AI is adapting), Pattern Diversity (variety of market conditions learned), Success Rate Evolution, and Recent Insights discovered by the AI. The system uses machine learning to improve recommendations over time. 💡 TRADING INSIGHT: High Learning Velocity = AI is rapidly adapting to NEW market conditions. High Pattern Diversity = AI has experience with VARIOUS market scenarios. Watch for 'Recent Insights' - these are NEW patterns the AI has discovered that could provide EDGE. When Success Rate is trending UP = AI is getting BETTER at predictions. Use this to gauge confidence in AI recommendations and adjust your reliance on system signals!""",

    "performance_tracker": """📈 AI Performance Tracker: This monitors the REAL-TIME performance of AI-generated signals and recommendations using Alpha Intelligence™ data. Tracks: Success Rate (% of profitable signals), Average Confidence (system conviction levels), Total Signals Generated, and Learning Score (improvement rate). Includes performance charts showing success rate evolution over time. 💡 TRADING INSIGHT: Success Rate > 70% = AI is performing WELL, trust the signals. Success Rate < 50% = AI struggling, reduce position sizes or switch to manual analysis. Average Confidence trending UP = AI becoming more CERTAIN in its analysis. Learning Score > 0.8 = AI is RAPIDLY IMPROVING. Use this data to calibrate your TRUST in AI recommendations and adjust position sizing accordingly. When performance metrics are ALL positive = HIGH CONFIDENCE in AI system!""",

    "apex_predator": """😈 Apex Predator Brain: This is the ULTIMATE INTELLIGENCE HUB combining Alpha Vantage news sentiment, MCP (Model Context Protocol) servers, and Diabolical Intelligence™. Displays: MCP Systems Status (Knowledge Graph, Sequential Thinking, Memory), Consolidated Intelligence Insights, Alpha Intelligence™ sentiment analysis, and Market Attention metrics. This is where ALL intelligence sources converge. 💡 TRADING INSIGHT: When MCP Systems show 'ACTIVE' status = FULL AI POWER engaged. Diabolical Insights provide CONTRARIAN perspectives that others miss. Alpha Intelligence™ sentiment EXTREME readings (>0.8 or <-0.2) = potential REVERSAL signals. High Market Attention = increased VOLATILITY expected. Use this as your FINAL CHECK before executing trades - it provides the MACRO CONTEXT that pure technical analysis misses. This is your EDGE over other traders!"""
}

def create_info_content_div(info_id: str, info_content: str) -> html.Div:
    """Create the info content div that will be toggled."""
    return html.Div([
        html.P([
            info_content
        ], style={
            "fontSize": "0.8rem",
            "lineHeight": "1.6",
            "color": "#ffffff",
            "margin": "0",
            "padding": "12px 16px",
            "background": "rgba(255, 255, 255, 0.05)",
            "borderRadius": "8px",
            "border": "1px solid rgba(255, 255, 255, 0.1)",
            "fontFamily": "'Inter', -apple-system, BlinkMacSystemFont, sans-serif"
        })
    ], id=f"info-content-{info_id}", style={
        "marginTop": "8px",
        "display": "none",
        "transition": "all 0.3s ease-in-out"
    })

# Register callbacks for each module
def register_collapsible_callbacks(app):
    """Register all collapsible callbacks with the Dash app."""

    module_ids = [
        "unified_intelligence",
        "regime_analysis",
        "raw_metrics",
        "recommendations",
        "learning_center",
        "performance_tracker",
        "apex_predator"
    ]

    # Register individual callbacks for each module to avoid closure issues
    def create_callback(module_id):
        @app.callback(
            Output(f"info-content-{module_id}", "style"),
            Input(f"title-button-{module_id}", "n_clicks"),
            State(f"info-content-{module_id}", "style"),
            prevent_initial_call=True
        )
        def toggle_info_section(n_clicks, current_style):
            """Toggle the visibility of the info section."""
            if n_clicks and n_clicks > 0:
                # Toggle display
                if current_style and current_style.get("display") == "none":
                    return {
                        "marginTop": "8px",
                        "display": "block",
                        "transition": "all 0.3s ease-in-out",
                        "opacity": "1",
                        "transform": "translateY(0)"
                    }
                else:
                    return {
                        "marginTop": "8px",
                        "display": "none",
                        "transition": "all 0.3s ease-in-out"
                    }

            # Default hidden state
            return {
                "marginTop": "8px",
                "display": "none",
                "transition": "all 0.3s ease-in-out"
            }

    # Create callbacks for each module
    for module_id in module_ids:
        create_callback(module_id)

    logger.info(f"✅ Registered {len(module_ids)} collapsible callbacks for Pydantic-first AI dashboard")
