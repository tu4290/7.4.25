"""AI Hub Layout Components Module for EOTS v2.5
============================================

This module contains reusable UI components specifically for the AI Hub layout including:
- Styling constants
- Card components
- Button components
- Text styling utilities

Author: EOTS v2.5 Development Team
Version: 2.5.0
"""

import logging
from typing import Dict, Any, Optional

from dash import html

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
    else:  # default
        return {
            'background': 'linear-gradient(145deg, #1e1e1e, #2a2a2a)',
            'border': '1px solid rgba(255, 255, 255, 0.1)',
            'borderRadius': '15px',
            'boxShadow': '0 8px 32px rgba(0, 0, 0, 0.3)',
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


def create_quick_action_buttons(bundle_data, symbol: str) -> html.Div:
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
        },
        "danger": {
            "fontSize": AI_TYPOGRAPHY['body_size'],
            "color": AI_COLORS['danger'],
            "lineHeight": "1.5"
        }
    }

    return styles.get(text_type, styles["body"])


def get_unified_badge_style(badge_style: str = 'success') -> Dict[str, str]:
    """Get unified badge styling."""
    base_style = {
        "fontSize": AI_TYPOGRAPHY['tiny_size'],
        "padding": f"{AI_SPACING['xs']} {AI_SPACING['sm']}",
        "borderRadius": AI_EFFECTS['border_radius_sm'],
        "fontWeight": AI_TYPOGRAPHY['subtitle_weight']
    }
    
    if badge_style == 'success':
        base_style.update({
            "backgroundColor": AI_COLORS['success'],
            "color": "#000000"
        })
    elif badge_style == 'warning':
        base_style.update({
            "backgroundColor": AI_COLORS['warning'],
            "color": "#000000"
        })
    elif badge_style == 'danger':
        base_style.update({
            "backgroundColor": AI_COLORS['danger'],
            "color": "#ffffff"
        })
    else:  # primary
        base_style.update({
            "backgroundColor": AI_COLORS['primary'],
            "color": "#000000"
        })
    
    return base_style


def create_clickable_title_with_info(title: str, info_id: str, info_content: str,
                                   title_style: Optional[Dict[str, str]] = None,
                                   badge_text: Optional[str] = None,
                                   badge_style: str = 'success') -> html.Details:
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
    summary_content = [html.Span(title, style=title_style or {})]
    if badge_text is not None:
        summary_content.append(html.Span(" "))
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
    ], id=info_id, style={
        "marginBottom": AI_SPACING['md']
    })