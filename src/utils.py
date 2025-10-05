"""
Utility Functions
=================
Reusable helper functions to avoid code duplication across the codebase.
"""

import re
import math
from typing import Dict, Any, Callable, Tuple, Optional


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


def extract_click_coordinates(action: str) -> Optional[Tuple[str, int, int]]:
    """
    Extract click type and (x, y) coordinates from a click action string.
    
    Args:
        action: Action string like "click:374,196" or "rightClick: 100, 200"
        
    Returns:
        Tuple of (click_type, x, y) if found, None otherwise
        click_type will be normalized to lowercase (e.g., "click", "rightclick", "doubleclick")
        
    Examples:
        >>> extract_click_coordinates("click:374,196")
        ('click', 374, 196)
        >>> extract_click_coordinates("rightClick: 100, 200")
        ('rightclick', 100, 200)
        >>> extract_click_coordinates("type:hello")
        None
    """
    if not action or 'click' not in action.lower():
        return None
    
    # Match patterns like "click:x,y", "rightClick:x,y", "doubleClick:x,y"
    match = re.search(r'(\w*click)\s*:\s*(\d+)\s*,\s*(\d+)', action, re.IGNORECASE)
    if match:
        click_type = match.group(1).lower()
        x = int(match.group(2))
        y = int(match.group(3))
        return (click_type, x, y)
    
    return None


def calculate_pixel_distance(coord1: Tuple[int, int], coord2: Tuple[int, int]) -> float:
    """
    Calculate Euclidean distance between two pixel coordinates.
    
    Args:
        coord1: First (x, y) coordinate
        coord2: Second (x, y) coordinate
        
    Returns:
        Euclidean distance in pixels
        
    Examples:
        >>> calculate_pixel_distance((100, 100), (103, 104))
        5.0
    """
    x1, y1 = coord1
    x2, y2 = coord2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def compare_actions(predicted: str, correct: str, click_tolerance: int = 10) -> bool:
    """
    Compare two actions with fuzzy matching for click coordinates.
    
    For click actions: considers them equal if SAME click type AND pixel distance < click_tolerance
    For other actions: performs exact string comparison
    
    Args:
        predicted: Predicted action string
        correct: Correct/expected action string
        click_tolerance: Maximum pixel distance to consider clicks equal (default: 10)
        
    Returns:
        True if actions match (exactly or within tolerance), False otherwise
        
    Examples:
        >>> compare_actions("click:100,100", "click:105,103")
        True  # Same type, distance ~5.8 pixels < 10
        >>> compare_actions("click:100,100", "click:150,150")
        False  # Same type, but distance ~70 pixels > 10
        >>> compare_actions("click:100,100", "rightClick:100,100")
        False  # Different click types (click vs rightClick)
        >>> compare_actions("type:hello", "type:hello")
        True  # Exact match
        >>> compare_actions("type:hello", "type:world")
        False  # Different actions
    """
    # Exact match case (fastest)
    if predicted == correct:
        return True
    
    # Try to extract click coordinates and types
    pred_result = extract_click_coordinates(predicted)
    correct_result = extract_click_coordinates(correct)
    
    # If both are click actions with valid coordinates
    if pred_result and correct_result:
        pred_type, pred_x, pred_y = pred_result
        correct_type, correct_x, correct_y = correct_result
        
        # Must be same click type (click, rightClick, doubleClick, etc.)
        if pred_type != correct_type:
            return False
        
        # Check pixel distance
        distance = calculate_pixel_distance((pred_x, pred_y), (correct_x, correct_y))
        return distance < click_tolerance
    
    # Fall back to exact string comparison
    return False

