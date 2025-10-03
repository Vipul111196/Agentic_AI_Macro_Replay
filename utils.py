"""
Utility Functions
=================
Reusable helper functions to avoid code duplication across the codebase.
"""

from typing import Dict, Any, Callable


def format_stat_value(key: str, value: Any) -> str:
    """
    Format a statistic value based on its type and key name.
    
    Args:
        key: The statistic key name
        value: The statistic value
        
    Returns:
        Formatted string representation
    """
    if isinstance(value, float):
        if 'rate' in key or 'accuracy' in key:
            return f"{value:.1%}"
        else:
            return f"{value:.2f}"
    else:
        return str(value)


def print_statistics(stats: Dict[str, Any], title: str = "Statistics", indent: str = "  • "):
    """
    Print statistics dict with smart formatting.
    
    Args:
        stats: Dictionary of statistics to print
        title: Optional title to print before stats
        indent: String to use for indentation
    """
    if title:
        print(f"\n{title}:")
    
    for key, value in stats.items():
        formatted_value = format_stat_value(key, value)
        print(f"{indent}{key}: {formatted_value}")


def truncate_text(text: str, max_length: int = 50, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length and add suffix.
    
    Args:
        text: Text to truncate
        max_length: Maximum length before truncation
        suffix: Suffix to add when truncated
        
    Returns:
        Truncated text with suffix, or original if shorter than max_length
    """
    if text and len(text) > max_length:
        return text[:max_length] + suffix
    return text


def format_key_label(key: str) -> str:
    """
    Format a snake_case key into a human-readable label.
    
    Args:
        key: Key in snake_case format
        
    Returns:
        Human-readable label (Title Case)
        
    Examples:
        >>> format_key_label("total_nodes")
        'Total Nodes'
        >>> format_key_label("replay_rate")
        'Replay Rate'
    """
    return key.replace('_', ' ').title()

