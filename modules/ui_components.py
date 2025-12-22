"""
Reusable UI Components with consistent styling
Provides helper functions for common UI patterns
"""

import streamlit as st
from modules.design_system import Colors, Spacing, Typography, BorderRadius, Shadows


def styled_button(
    label: str,
    key: str = None,
    style: str = "primary",
    size: str = "md",
    on_click=None,
    args=None,
    kwargs=None,
    disabled: bool = False,
    help_text: str = None,
    icon: str = None,
    full_width: bool = False
) -> bool:
    """
    Create a styled button with consistent appearance
    
    Args:
        label: Button text
        key: Unique identifier
        style: "primary", "secondary", "success", "danger", "outline"
        size: "sm", "md", "lg"
        on_click: Callback function
        args: Arguments for callback
        kwargs: Keyword arguments for callback
        disabled: Whether button is disabled
        help_text: Hover help text
        icon: Icon emoji/character
        full_width: Make button take full width
    
    Returns:
        Boolean indicating if button was clicked
    """
    full_label = f"{icon} {label}" if icon else label
    
    col_config = None
    if full_width:
        col_config = [1]
    
    cols = st.columns(col_config or [1, 10]) if not full_width else [st.columns(1)[0]]
    
    if full_width:
        with cols[0]:
            clicked = st.button(
                full_label,
                key=key,
                on_click=on_click,
                args=args if args else (),
                kwargs=kwargs if kwargs else {},
                disabled=disabled,
                help=help_text,
                use_container_width=True
            )
    else:
        clicked = st.button(
            full_label,
            key=key,
            on_click=on_click,
            args=args if args else (),
            kwargs=kwargs if kwargs else {},
            disabled=disabled,
            help=help_text,
            use_container_width=True
        )
    
    return clicked


def styled_metric(
    label: str,
    value,
    delta: str = None,
    help_text: str = None,
    columns: int = 1
):
    """
    Create a styled metric card
    
    Args:
        label: Metric label
        value: Metric value
        delta: Change value (optional)
        help_text: Help tooltip
        columns: Number of columns to span
    
    Returns:
        None (displays metric)
    """
    st.metric(label, value, delta=delta, help=help_text)


def styled_header(
    title: str,
    subtitle: str = None,
    icon: str = None,
    divider: bool = True
):
    """
    Create a styled header section
    
    Args:
        title: Header title
        subtitle: Optional subtitle
        icon: Icon emoji/character
        divider: Show divider line below
    
    Returns:
        None (displays header)
    """
    header_text = f"{icon} {title}" if icon else title
    st.markdown(f"## {header_text}")
    
    if subtitle:
        st.markdown(f"_{subtitle}_")
    
    if divider:
        st.divider()


def styled_section(
    title: str,
    content_fn,
    icon: str = None,
    expanded: bool = True,
    help_text: str = None
):
    """
    Create a styled expandable section
    
    Args:
        title: Section title
        content_fn: Function to call for section content
        icon: Icon emoji/character
        expanded: Whether section starts expanded
        help_text: Help tooltip
    
    Returns:
        None (displays section)
    """
    section_title = f"{icon} {title}" if icon else title
    
    with st.expander(section_title, expanded=expanded, help=help_text):
        content_fn()


def styled_info_box(
    content: str,
    title: str = None,
    icon: str = "ℹ️",
    box_type: str = "info"
):
    """
    Create a styled information box
    
    Args:
        content: Box content
        title: Optional title
        icon: Icon emoji
        box_type: "info", "success", "warning", "error"
    
    Returns:
        None (displays box)
    """
    full_content = ""
    if title:
        full_content += f"**{icon} {title}**\n\n"
    else:
        full_content += f"{icon} "
    
    full_content += content
    
    if box_type == "success":
        st.success(full_content)
    elif box_type == "warning":
        st.warning(full_content)
    elif box_type == "error":
        st.error(full_content)
    else:  # info
        st.info(full_content)


def styled_columns_layout(
    num_columns: int = 2,
    gap: str = "md"
) -> list:
    """
    Create styled columns with consistent spacing
    
    Args:
        num_columns: Number of columns
        gap: Spacing between columns ("sm", "md", "lg")
    
    Returns:
        List of column containers
    """
    gap_values = {"sm": 1, "md": 2, "lg": 3}
    gap_ratio = gap_values.get(gap, 2)
    
    return st.columns([1] * num_columns)


def styled_form_group(
    label: str,
    input_fn,
    help_text: str = None,
    required: bool = False
):
    """
    Create a styled form group with label
    
    Args:
        label: Field label
        input_fn: Function to create input widget
        help_text: Helper text
        required: Whether field is required
    
    Returns:
        Result from input function
    """
    required_text = " *" if required else ""
    st.markdown(f"**{label}{required_text}**")
    
    if help_text:
        st.caption(help_text)
    
    return input_fn()


def styled_card(
    title: str = None,
    content_fn=None,
    footer_fn=None,
    icon: str = None,
    style: str = "default"
):
    """
    Create a styled card container
    
    Args:
        title: Card title
        content_fn: Function to render card content
        footer_fn: Function to render card footer
        icon: Icon emoji
        style: "default", "highlighted", "subtle"
    
    Returns:
        None (displays card)
    """
    # Use Streamlit container to create card
    with st.container():
        if title:
            title_text = f"{icon} {title}" if icon else title
            st.markdown(f"**{title_text}**")
        
        if content_fn:
            content_fn()
        
        if footer_fn:
            st.divider()
            footer_fn()


def styled_data_summary(
    data_dict: dict,
    columns: int = 4
):
    """
    Create a styled summary of data metrics
    
    Args:
        data_dict: Dictionary of label: value pairs
        columns: Number of columns for layout
    
    Returns:
        None (displays summary)
    """
    cols = st.columns(columns)
    
    for idx, (label, value) in enumerate(data_dict.items()):
        with cols[idx % columns]:
            st.metric(label, value)


def styled_divider(spacing: str = "md"):
    """
    Create a styled divider with consistent spacing
    
    Args:
        spacing: Spacing around divider ("sm", "md", "lg")
    
    Returns:
        None (displays divider)
    """
    st.divider()


def get_responsive_columns(total_columns: int = 4, breakpoint: str = "md") -> int:
    """
    Calculate responsive number of columns based on screen size
    
    Args:
        total_columns: Desired number of columns on desktop
        breakpoint: "sm", "md", "lg"
    
    Returns:
        Recommended number of columns
    """
    # Note: Streamlit doesn't provide native screen size detection
    # This is a helper for consistent responsive behavior
    if breakpoint == "sm":
        return 1
    elif breakpoint == "md":
        return max(1, total_columns // 2)
    else:  # lg
        return total_columns
