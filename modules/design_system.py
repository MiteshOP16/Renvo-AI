"""
Comprehensive Design System for Renvo AI
Provides consistent colors, typography, spacing, and component styles
"""

import streamlit as st
from typing import Literal

# ============================================================================
# COLOR PALETTE
# ============================================================================

class Colors:
    """Semantic color definitions"""
    # Primary Colors
    PRIMARY = "#2563EB"  # Blue
    PRIMARY_LIGHT = "#DBEAFE"
    PRIMARY_DARK = "#1E40AF"
    
    # Secondary Colors
    SECONDARY = "#7C3AED"  # Purple
    SECONDARY_LIGHT = "#EDE9FE"
    SECONDARY_DARK = "#6D28D9"
    
    # Success Colors
    SUCCESS = "#10B981"  # Green
    SUCCESS_LIGHT = "#D1FAE5"
    SUCCESS_DARK = "#047857"
    
    # Warning Colors
    WARNING = "#F59E0B"  # Amber
    WARNING_LIGHT = "#FEF3C7"
    WARNING_DARK = "#D97706"
    
    # Error Colors
    ERROR = "#EF4444"  # Red
    ERROR_LIGHT = "#FEE2E2"
    ERROR_DARK = "#DC2626"
    
    # Info Colors
    INFO = "#3B82F6"  # Light Blue
    INFO_LIGHT = "#EFF6FF"
    INFO_DARK = "#1D4ED8"
    
    # Neutral Colors
    NEUTRAL_50 = "#F9FAFB"
    NEUTRAL_100 = "#F3F4F6"
    NEUTRAL_200 = "#E5E7EB"
    NEUTRAL_300 = "#D1D5DB"
    NEUTRAL_400 = "#9CA3AF"
    NEUTRAL_500 = "#6B7280"
    NEUTRAL_600 = "#4B5563"
    NEUTRAL_700 = "#374151"
    NEUTRAL_800 = "#1F2937"
    NEUTRAL_900 = "#111827"
    
    # Background
    BACKGROUND = "#FFFFFF"
    BACKGROUND_SECONDARY = "#F9FAFB"
    BACKGROUND_TERTIARY = "#F3F4F6"
    
    # Text
    TEXT_PRIMARY = "#1F2937"
    TEXT_SECONDARY = "#6B7280"
    TEXT_TERTIARY = "#9CA3AF"
    TEXT_INVERSE = "#FFFFFF"
    
    # Borders
    BORDER_LIGHT = "#E5E7EB"
    BORDER_MEDIUM = "#D1D5DB"
    BORDER_DARK = "#9CA3AF"
    
    # Status
    CRITICAL = "#DC2626"
    MODERATE = "#F59E0B"
    LOW = "#10B981"


# ============================================================================
# TYPOGRAPHY
# ============================================================================

class Typography:
    """Typography definitions for consistent text styling"""
    
    # Font families
    FONT_FAMILY_BASE = "sans-serif"
    FONT_FAMILY_MONO = "monospace"
    
    # Font sizes (in pixels)
    SIZE_XS = 12
    SIZE_SM = 14
    SIZE_BASE = 16
    SIZE_LG = 18
    SIZE_XL = 20
    SIZE_2XL = 24
    SIZE_3XL = 30
    SIZE_4XL = 36
    
    # Font weights
    WEIGHT_NORMAL = 400
    WEIGHT_MEDIUM = 500
    WEIGHT_SEMIBOLD = 600
    WEIGHT_BOLD = 700
    
    # Line heights
    LINE_HEIGHT_TIGHT = 1.25
    LINE_HEIGHT_NORMAL = 1.5
    LINE_HEIGHT_RELAXED = 1.75
    LINE_HEIGHT_LOOSE = 2


# ============================================================================
# SPACING
# ============================================================================

class Spacing:
    """Consistent spacing scale"""
    
    XS = 4
    SM = 8
    MD = 16
    LG = 24
    XL = 32
    XXL = 48
    XXXL = 64


# ============================================================================
# BORDER RADIUS
# ============================================================================

class BorderRadius:
    """Border radius definitions"""
    
    NONE = 0
    SM = 4
    MD = 8
    LG = 12
    XL = 16
    FULL = 9999


# ============================================================================
# SHADOWS
# ============================================================================

class Shadows:
    """Box shadow definitions"""
    
    NONE = "none"
    SM = "0 1px 2px rgba(0, 0, 0, 0.05)"
    MD = "0 4px 6px rgba(0, 0, 0, 0.1)"
    LG = "0 10px 15px rgba(0, 0, 0, 0.1)"
    XL = "0 20px 25px rgba(0, 0, 0, 0.15)"
    FOCUS = "0 0 0 3px rgba(37, 99, 235, 0.1), 0 0 0 4px rgba(37, 99, 235, 1)"


# ============================================================================
# RESPONSIVE BREAKPOINTS
# ============================================================================

class Breakpoints:
    """Responsive design breakpoints"""
    
    SM = 480
    MD = 768
    LG = 1024
    XL = 1280
    XXL = 1536


# ============================================================================
# BUTTON STYLES
# ============================================================================

class ButtonStyle:
    """Button styling configurations"""
    
    @staticmethod
    def primary(size: Literal["sm", "md", "lg"] = "md") -> dict:
        """Primary button style"""
        sizes = {
            "sm": {"padding": "8px 16px", "font_size": 14},
            "md": {"padding": "12px 24px", "font_size": 16},
            "lg": {"padding": "16px 32px", "font_size": 18},
        }
        return {
            "background_color": Colors.PRIMARY,
            "text_color": Colors.TEXT_INVERSE,
            "border_color": Colors.PRIMARY,
            "border_radius": BorderRadius.MD,
            "font_weight": Typography.WEIGHT_SEMIBOLD,
            **sizes.get(size, sizes["md"])
        }
    
    @staticmethod
    def secondary(size: Literal["sm", "md", "lg"] = "md") -> dict:
        """Secondary button style"""
        sizes = {
            "sm": {"padding": "8px 16px", "font_size": 14},
            "md": {"padding": "12px 24px", "font_size": 16},
            "lg": {"padding": "16px 32px", "font_size": 18},
        }
        return {
            "background_color": Colors.SECONDARY_LIGHT,
            "text_color": Colors.SECONDARY_DARK,
            "border_color": Colors.SECONDARY,
            "border_radius": BorderRadius.MD,
            "font_weight": Typography.WEIGHT_SEMIBOLD,
            **sizes.get(size, sizes["md"])
        }
    
    @staticmethod
    def success(size: Literal["sm", "md", "lg"] = "md") -> dict:
        """Success button style"""
        sizes = {
            "sm": {"padding": "8px 16px", "font_size": 14},
            "md": {"padding": "12px 24px", "font_size": 16},
            "lg": {"padding": "16px 32px", "font_size": 18},
        }
        return {
            "background_color": Colors.SUCCESS,
            "text_color": Colors.TEXT_INVERSE,
            "border_color": Colors.SUCCESS,
            "border_radius": BorderRadius.MD,
            "font_weight": Typography.WEIGHT_SEMIBOLD,
            **sizes.get(size, sizes["md"])
        }
    
    @staticmethod
    def danger(size: Literal["sm", "md", "lg"] = "md") -> dict:
        """Danger button style"""
        sizes = {
            "sm": {"padding": "8px 16px", "font_size": 14},
            "md": {"padding": "12px 24px", "font_size": 16},
            "lg": {"padding": "16px 32px", "font_size": 18},
        }
        return {
            "background_color": Colors.ERROR,
            "text_color": Colors.TEXT_INVERSE,
            "border_color": Colors.ERROR,
            "border_radius": BorderRadius.MD,
            "font_weight": Typography.WEIGHT_SEMIBOLD,
            **sizes.get(size, sizes["md"])
        }
    
    @staticmethod
    def outline(size: Literal["sm", "md", "lg"] = "md") -> dict:
        """Outline button style"""
        sizes = {
            "sm": {"padding": "8px 16px", "font_size": 14},
            "md": {"padding": "12px 24px", "font_size": 16},
            "lg": {"padding": "16px 32px", "font_size": 18},
        }
        return {
            "background_color": Colors.BACKGROUND,
            "text_color": Colors.PRIMARY,
            "border_color": Colors.PRIMARY,
            "border_width": 2,
            "border_radius": BorderRadius.MD,
            "font_weight": Typography.WEIGHT_SEMIBOLD,
            **sizes.get(size, sizes["md"])
        }


# ============================================================================
# GLOBAL STYLING
# ============================================================================

def apply_global_styles():
    """Apply consistent global CSS styling across the application"""
    css = f"""
    <style>
    /* ========== ROOT & BASE STYLES ========== */
    :root {{
        /* Colors */
        --color-primary: {Colors.PRIMARY};
        --color-primary-light: {Colors.PRIMARY_LIGHT};
        --color-primary-dark: {Colors.PRIMARY_DARK};
        --color-secondary: {Colors.SECONDARY};
        --color-success: {Colors.SUCCESS};
        --color-warning: {Colors.WARNING};
        --color-error: {Colors.ERROR};
        --color-info: {Colors.INFO};
        
        /* Neutral Colors */
        --color-neutral-100: {Colors.NEUTRAL_100};
        --color-neutral-200: {Colors.NEUTRAL_200};
        --color-neutral-300: {Colors.NEUTRAL_300};
        --color-neutral-500: {Colors.NEUTRAL_500};
        --color-neutral-700: {Colors.NEUTRAL_700};
        --color-neutral-800: {Colors.NEUTRAL_800};
        --color-neutral-900: {Colors.NEUTRAL_900};
        
        /* Text Colors */
        --color-text-primary: {Colors.TEXT_PRIMARY};
        --color-text-secondary: {Colors.TEXT_SECONDARY};
        --color-text-tertiary: {Colors.TEXT_TERTIARY};
        --color-text-inverse: {Colors.TEXT_INVERSE};
        
        /* Spacing */
        --spacing-xs: {Spacing.XS}px;
        --spacing-sm: {Spacing.SM}px;
        --spacing-md: {Spacing.MD}px;
        --spacing-lg: {Spacing.LG}px;
        --spacing-xl: {Spacing.XL}px;
        
        /* Typography */
        --font-family: {Typography.FONT_FAMILY_BASE};
        --font-family-mono: {Typography.FONT_FAMILY_MONO};
        --font-size-base: {Typography.SIZE_BASE}px;
        --font-weight-normal: {Typography.WEIGHT_NORMAL};
        --font-weight-semibold: {Typography.WEIGHT_SEMIBOLD};
        --font-weight-bold: {Typography.WEIGHT_BOLD};
        
        /* Border Radius */
        --border-radius-md: {BorderRadius.MD}px;
        --border-radius-lg: {BorderRadius.LG}px;
        
        /* Shadows */
        --shadow-md: {Shadows.MD};
        --shadow-lg: {Shadows.LG};
    }}
    
    * {{
        box-sizing: border-box;
    }}
    
    html, body {{
        font-family: var(--font-family);
        font-size: var(--font-size-base);
        color: var(--color-text-primary);
        background-color: {Colors.BACKGROUND};
        line-height: {Typography.LINE_HEIGHT_NORMAL};
    }}
    
    /* ========== TYPOGRAPHY ========== */
    h1, h2, h3, h4, h5, h6 {{
        font-weight: var(--font-weight-bold);
        line-height: {Typography.LINE_HEIGHT_TIGHT};
        color: var(--color-text-primary);
        margin-top: var(--spacing-lg);
        margin-bottom: var(--spacing-md);
    }}
    
    h1 {{
        font-size: {Typography.SIZE_3XL}px;
        letter-spacing: -0.02em;
    }}
    
    h2 {{
        font-size: {Typography.SIZE_2XL}px;
        letter-spacing: -0.01em;
    }}
    
    h3 {{
        font-size: {Typography.SIZE_XL}px;
    }}
    
    h4 {{
        font-size: {Typography.SIZE_LG}px;
    }}
    
    h5 {{
        font-size: {Typography.SIZE_BASE}px;
    }}
    
    h6 {{
        font-size: {Typography.SIZE_SM}px;
    }}
    
    p {{
        margin: 0 0 var(--spacing-md) 0;
        line-height: {Typography.LINE_HEIGHT_RELAXED};
    }}
    
    a {{
        color: var(--color-primary);
        text-decoration: none;
        transition: color 0.2s ease;
    }}
    
    a:hover {{
        color: var(--color-primary-dark);
        text-decoration: underline;
    }}
    
    a:focus {{
        outline: 2px solid var(--color-primary);
        outline-offset: 2px;
        border-radius: 2px;
    }}
    
    /* ========== BUTTONS ========== */
    button, [role="button"] {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 12px 24px;
        font-size: var(--font-size-base);
        font-weight: var(--font-weight-semibold);
        border: 2px solid transparent;
        border-radius: var(--border-radius-md);
        cursor: pointer;
        transition: all 0.2s ease;
        text-decoration: none;
        user-select: none;
        white-space: nowrap;
        min-height: 44px;
        min-width: 44px;
    }}
    
    button:focus, [role="button"]:focus {{
        outline: none;
        box-shadow: {Shadows.FOCUS};
    }}
    
    button:disabled, [role="button"]:disabled {{
        opacity: 0.6;
        cursor: not-allowed;
    }}
    
    /* Primary buttons */
    button.primary, [role="button"].primary {{
        background-color: var(--color-primary);
        color: var(--color-text-inverse);
        border-color: var(--color-primary);
    }}
    
    button.primary:hover, [role="button"].primary:hover {{
        background-color: var(--color-primary-dark);
        border-color: var(--color-primary-dark);
        box-shadow: var(--shadow-md);
    }}
    
    /* Secondary buttons */
    button.secondary, [role="button"].secondary {{
        background-color: {Colors.SECONDARY_LIGHT};
        color: {Colors.SECONDARY_DARK};
        border-color: var(--color-secondary);
    }}
    
    button.secondary:hover, [role="button"].secondary:hover {{
        background-color: var(--color-secondary);
        color: var(--color-text-inverse);
        box-shadow: var(--shadow-md);
    }}
    
    /* Success buttons */
    button.success, [role="button"].success {{
        background-color: var(--color-success);
        color: var(--color-text-inverse);
        border-color: var(--color-success);
    }}
    
    button.success:hover, [role="button"].success:hover {{
        background-color: {Colors.SUCCESS_DARK};
        border-color: {Colors.SUCCESS_DARK};
        box-shadow: var(--shadow-md);
    }}
    
    /* Danger buttons */
    button.danger, [role="button"].danger {{
        background-color: var(--color-error);
        color: var(--color-text-inverse);
        border-color: var(--color-error);
    }}
    
    button.danger:hover, [role="button"].danger:hover {{
        background-color: {Colors.ERROR_DARK};
        border-color: {Colors.ERROR_DARK};
        box-shadow: var(--shadow-md);
    }}
    
    /* Outline buttons */
    button.outline, [role="button"].outline {{
        background-color: transparent;
        color: var(--color-primary);
        border-color: var(--color-primary);
    }}
    
    button.outline:hover, [role="button"].outline:hover {{
        background-color: var(--color-primary-light);
        box-shadow: var(--shadow-md);
    }}
    
    /* Button sizes */
    button.sm, [role="button"].sm {{
        padding: 8px 16px;
        font-size: {Typography.SIZE_SM}px;
        min-height: 36px;
    }}
    
    button.lg, [role="button"].lg {{
        padding: 16px 32px;
        font-size: {Typography.SIZE_LG}px;
        min-height: 52px;
    }}
    
    /* ========== FORM ELEMENTS ========== */
    input, textarea, select {{
        display: block;
        width: 100%;
        padding: 10px 12px;
        font-size: var(--font-size-base);
        font-family: inherit;
        border: 2px solid {Colors.BORDER_LIGHT};
        border-radius: var(--border-radius-md);
        transition: all 0.2s ease;
        background-color: {Colors.BACKGROUND};
        color: var(--color-text-primary);
    }}
    
    input:focus, textarea:focus, select:focus {{
        outline: none;
        border-color: var(--color-primary);
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
    }}
    
    input::placeholder {{
        color: var(--color-text-tertiary);
    }}
    
    label {{
        display: block;
        margin-bottom: var(--spacing-sm);
        font-weight: var(--font-weight-semibold);
        color: var(--color-text-primary);
        font-size: {Typography.SIZE_SM}px;
    }}
    
    /* ========== CARDS & CONTAINERS ========== */
    .card {{
        background-color: {Colors.BACKGROUND};
        border: 1px solid {Colors.BORDER_LIGHT};
        border-radius: var(--border-radius-lg);
        padding: var(--spacing-lg);
        box-shadow: var(--shadow-sm);
        transition: all 0.2s ease;
    }}
    
    .card:hover {{
        box-shadow: var(--shadow-md);
        border-color: {Colors.BORDER_MEDIUM};
    }}
    
    .container {{
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 var(--spacing-md);
    }}
    
    /* ========== STATUS MESSAGES ========== */
    .alert, .alert-success, .alert-warning, .alert-error, .alert-info {{
        padding: var(--spacing-md);
        border-radius: var(--border-radius-md);
        border-left: 4px solid;
        margin-bottom: var(--spacing-md);
    }}
    
    .alert-success {{
        background-color: {Colors.SUCCESS_LIGHT};
        border-left-color: var(--color-success);
        color: {Colors.SUCCESS_DARK};
    }}
    
    .alert-warning {{
        background-color: {Colors.WARNING_LIGHT};
        border-left-color: var(--color-warning);
        color: {Colors.WARNING};
    }}
    
    .alert-error {{
        background-color: {Colors.ERROR_LIGHT};
        border-left-color: var(--color-error);
        color: {Colors.ERROR_DARK};
    }}
    
    .alert-info {{
        background-color: {Colors.INFO_LIGHT};
        border-left-color: var(--color-info);
        color: {Colors.INFO_DARK};
    }}
    
    /* ========== DATA TABLES ========== */
    table {{
        width: 100%;
        border-collapse: collapse;
        margin-bottom: var(--spacing-lg);
        font-size: {Typography.SIZE_SM}px;
    }}
    
    th {{
        background-color: {Colors.BACKGROUND_TERTIARY};
        padding: var(--spacing-md);
        text-align: left;
        font-weight: var(--font-weight-bold);
        border-bottom: 2px solid {Colors.BORDER_LIGHT};
        color: var(--color-text-primary);
    }}
    
    td {{
        padding: var(--spacing-md);
        border-bottom: 1px solid {Colors.BORDER_LIGHT};
        color: var(--color-text-primary);
    }}
    
    tr:hover {{
        background-color: {Colors.BACKGROUND_TERTIARY};
    }}
    
    /* ========== RESPONSIVE DESIGN ========== */
    @media (max-width: {Breakpoints.MD}px) {{
        html, body {{
            font-size: 15px;
        }}
        
        h1 {{
            font-size: {Typography.SIZE_2XL}px;
        }}
        
        h2 {{
            font-size: {Typography.SIZE_XL}px;
        }}
        
        button, [role="button"] {{
            padding: 10px 16px;
            font-size: {Typography.SIZE_SM}px;
        }}
        
        .container {{
            padding: 0 var(--spacing-sm);
        }}
    }}
    
    @media (max-width: {Breakpoints.SM}px) {{
        h1 {{
            font-size: {Typography.SIZE_XL}px;
        }}
        
        h2 {{
            font-size: {Typography.SIZE_LG}px;
        }}
        
        button, [role="button"] {{
            width: 100%;
        }}
    }}
    
    /* ========== PRINT STYLES ========== */
    @media print {{
        body {{
            background-color: white;
            color: black;
        }}
        
        button, [role="button"], .no-print {{
            display: none !important;
        }}
        
        a {{
            color: black;
            text-decoration: underline;
        }}
        
        .card {{
            border: 1px solid #ccc;
            break-inside: avoid;
            page-break-inside: avoid;
        }}
        
        h1, h2, h3 {{
            page-break-after: avoid;
        }}
        
        table {{
            border-collapse: collapse;
            width: 100%;
        }}
        
        th, td {{
            border: 1px solid #ccc;
            padding: 8px;
            text-align: left;
        }}
        
        tr {{
            page-break-inside: avoid;
        }}
        
        img {{
            max-width: 100%;
            height: auto;
        }}
        
        @page {{
            margin: 2cm;
        }}
    }}
    
    /* ========== ACCESSIBILITY ========== */
    @media (prefers-reduced-motion: reduce) {{
        * {{
            animation: none !important;
            transition: none !important;
        }}
    }}
    
    @media (prefers-color-scheme: dark) {{
        html, body {{
            background-color: {Colors.NEUTRAL_900};
            color: {Colors.NEUTRAL_100};
        }}
        
        input, textarea, select {{
            background-color: {Colors.NEUTRAL_800};
            color: {Colors.NEUTRAL_100};
            border-color: {Colors.NEUTRAL_700};
        }}
        
        th {{
            background-color: {Colors.NEUTRAL_800};
        }}
        
        td, tr:hover {{
            background-color: {Colors.NEUTRAL_900};
        }}
    }}
    
    /* ========== UTILITY CLASSES ========== */
    .text-center {{
        text-align: center;
    }}
    
    .text-right {{
        text-align: right;
    }}
    
    .text-sm {{
        font-size: {Typography.SIZE_SM}px;
    }}
    
    .text-lg {{
        font-size: {Typography.SIZE_LG}px;
    }}
    
    .text-bold {{
        font-weight: var(--font-weight-bold);
    }}
    
    .text-muted {{
        color: var(--color-text-tertiary);
    }}
    
    .mb-sm {{
        margin-bottom: var(--spacing-sm);
    }}
    
    .mb-md {{
        margin-bottom: var(--spacing-md);
    }}
    
    .mb-lg {{
        margin-bottom: var(--spacing-lg);
    }}
    
    .mt-sm {{
        margin-top: var(--spacing-sm);
    }}
    
    .mt-md {{
        margin-top: var(--spacing-md);
    }}
    
    .mt-lg {{
        margin-top: var(--spacing-lg);
    }}
    
    .p-sm {{
        padding: var(--spacing-sm);
    }}
    
    .p-md {{
        padding: var(--spacing-md);
    }}
    
    .p-lg {{
        padding: var(--spacing-lg);
    }}
    
    .flex {{
        display: flex;
    }}
    
    .flex-center {{
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    
    .flex-between {{
        display: flex;
        align-items: center;
        justify-content: space-between;
    }}
    
    .gap-sm {{
        gap: var(--spacing-sm);
    }}
    
    .gap-md {{
        gap: var(--spacing-md);
    }}
    
    .gap-lg {{
        gap: var(--spacing-lg);
    }}
    
    /* ========== STREAMLIT SPECIFIC ========== */
    .stButton > button {{
        background-color: var(--color-primary);
        color: var(--color-text-inverse);
        border: 2px solid var(--color-primary);
        border-radius: var(--border-radius-md);
        padding: 12px 24px;
        font-weight: var(--font-weight-semibold);
        font-size: {Typography.SIZE_BASE}px;
        min-height: 44px;
        width: 100%;
        transition: all 0.2s ease;
    }}
    
    .stButton > button:hover {{
        background-color: var(--color-primary-dark);
        border-color: var(--color-primary-dark);
        box-shadow: var(--shadow-md);
    }}
    
    .stButton > button:focus {{
        outline: none;
        box-shadow: {Shadows.FOCUS};
    }}
    
    .stSelectbox {{
        font-size: {Typography.SIZE_BASE}px;
    }}
    
    .stSelectbox select {{
        border: 2px solid {Colors.BORDER_LIGHT};
        border-radius: var(--border-radius-md);
        padding: 10px 12px;
        transition: all 0.2s ease;
    }}
    
    .stSelectbox select:focus {{
        outline: none;
        border-color: var(--color-primary);
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
    }}
    
    .stMetric {{
        background-color: {Colors.BACKGROUND_SECONDARY};
        padding: var(--spacing-md);
        border-radius: var(--border-radius-md);
        border: 1px solid {Colors.BORDER_LIGHT};
    }}
    
    .stSuccess {{
        background-color: {Colors.SUCCESS_LIGHT};
        color: {Colors.SUCCESS_DARK};
        border-left: 4px solid var(--color-success);
        padding: var(--spacing-md);
        border-radius: var(--border-radius-md);
    }}
    
    .stWarning {{
        background-color: {Colors.WARNING_LIGHT};
        color: {Colors.WARNING};
        border-left: 4px solid var(--color-warning);
        padding: var(--spacing-md);
        border-radius: var(--border-radius-md);
    }}
    
    .stError {{
        background-color: {Colors.ERROR_LIGHT};
        color: {Colors.ERROR_DARK};
        border-left: 4px solid var(--color-error);
        padding: var(--spacing-md);
        border-radius: var(--border-radius-md);
    }}
    
    .stInfo {{
        background-color: {Colors.INFO_LIGHT};
        color: {Colors.INFO_DARK};
        border-left: 4px solid var(--color-info);
        padding: var(--spacing-md);
        border-radius: var(--border-radius-md);
    }}
    
    .stDivider {{
        margin: var(--spacing-lg) 0;
        border-color: {Colors.BORDER_LIGHT};
    }}
    
    .stDataFrame {{
        font-size: {Typography.SIZE_SM}px;
    }}
    
    .stDataFrame table {{
        width: 100%;
        border-collapse: collapse;
    }}
    
    .stDataFrame th {{
        background-color: {Colors.BACKGROUND_TERTIARY};
        padding: var(--spacing-sm);
        font-weight: var(--font-weight-bold);
        border-bottom: 2px solid {Colors.BORDER_LIGHT};
    }}
    
    .stDataFrame td {{
        padding: var(--spacing-sm);
        border-bottom: 1px solid {Colors.BORDER_LIGHT};
    }}
    
    .stDataFrame tr:hover {{
        background-color: {Colors.BACKGROUND_SECONDARY};
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
