"""
Improved heatmap color logic for ABSA dashboard.
Intuitive colors: Positive (red→green), Neutral (yellow gradients), Negative (green→red).
"""


def get_aspect_heatmap_color(sentiment_type: str, percentage: float) -> str:
    """
    Get intuitive heatmap color based on sentiment type and percentage.

    Args:
        sentiment_type: 'positive', 'neutral', or 'negative'
        percentage: Value from 0 to 100

    Returns:
        CSS color string (hex or rgba)
    """
    # Clamp percentage to 0-100 range
    pct = max(0, min(100, percentage))

    if sentiment_type.lower() == 'positive':
        # Positive: 0% = Red, 100% = Green
        # More positive sentiment = greener (good)
        if pct < 25:
            # 0-25%: Red to Orange
            red = 255
            green = int(pct * 4 * 1.02)  # 0 to 102
            blue = 0
        elif pct < 50:
            # 25-50%: Orange to Yellow
            red = int(255 - (pct - 25) * 4 * 2.04)  # 255 to 51
            green = int(102 + (pct - 25) * 4 * 1.53)  # 102 to 255
            blue = 0
        elif pct < 75:
            # 50-75%: Yellow to Light Green
            red = int(51 - (pct - 50) * 4 * 0.51)  # 51 to 0
            green = 255
            blue = int((pct - 50) * 4 * 1.28)  # 0 to 128
        else:
            # 75-100%: Light Green to Dark Green
            red = 0
            green = int(255 - (pct - 75) * 4 * 1.02)  # 255 to 153
            blue = int(128 + (pct - 75) * 4 * 1.27)  # 128 to 255

        return f"rgb({red}, {green}, {blue})"

    elif sentiment_type.lower() == 'neutral':
        # Neutral: Yellow gradients - higher percentage = deeper yellow
        # Careful with contrast - readable text on yellow background
        if pct < 20:
            # Very light yellow (almost white) for low percentages
            return f"rgba(255, 255, 200, {0.3 + pct * 0.01})"  # Light yellow with increasing opacity
        elif pct < 50:
            # Light to medium yellow
            intensity = int(255 - (pct - 20) * 1.5)  # 255 to 210
            return f"rgb(255, 255, {intensity})"
        elif pct < 75:
            # Medium to darker yellow
            intensity = int(210 - (pct - 50) * 1.8)  # 210 to 165
            return f"rgb(255, 255, {intensity})"
        else:
            # Dark yellow/gold for high percentages
            intensity = int(165 - (pct - 75) * 1.4)  # 165 to 130
            return f"rgb(255, 255, {intensity})"

    elif sentiment_type.lower() == 'negative':
        # Negative: 0% = Green, 100% = Red (reverse of positive)
        # More negative sentiment = redder (bad)
        if pct < 25:
            # 0-25%: Green to Light Green
            red = 0
            green = int(255 - pct * 4 * 1.02)  # 255 to 153
            blue = int(255 - pct * 4 * 1.27)  # 255 to 128
        elif pct < 50:
            # 25-50%: Light Green to Yellow
            red = int((pct - 25) * 4 * 0.51)  # 0 to 51
            green = 255
            blue = int(128 - (pct - 25) * 4 * 1.28)  # 128 to 0
        elif pct < 75:
            # 50-75%: Yellow to Orange
            red = int(51 + (pct - 50) * 4 * 2.04)  # 51 to 255
            green = int(255 - (pct - 50) * 4 * 1.53)  # 255 to 102
            blue = 0
        else:
            # 75-100%: Orange to Red
            red = 255
            green = int(102 - (pct - 75) * 4 * 1.02)  # 102 to 0
            blue = 0

        return f"rgb({red}, {green}, {blue})"

    else:
        # Fallback: light gray
        return "rgb(240, 240, 240)"


def get_text_color_for_background(background_color: str) -> str:
    """
    Get appropriate text color (black/white) for given background color.
    Ensures good contrast for readability.
    """
    # Extract RGB values from background color
    if background_color.startswith('rgb('):
        rgb_str = background_color[4:-1]  # Remove 'rgb(' and ')'
        r, g, b = map(int, rgb_str.split(', '))
    elif background_color.startswith('rgba('):
        rgba_str = background_color[5:-1]  # Remove 'rgba(' and ')'
        r, g, b = map(int, rgba_str.split(', ')[:3])  # Take only RGB, ignore alpha
    else:
        # Fallback
        return "black"

    # Calculate luminance
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255

    # Use white text on dark backgrounds, black text on light backgrounds
    return "white" if luminance < 0.5 else "black"


def apply_heatmap_styling(df, sentiment_columns):
    """
    Apply improved heatmap styling to DataFrame columns.

    Args:
        df: pandas DataFrame
        sentiment_columns: dict with keys 'positive', 'neutral', 'negative'
                          and values as column names

    Returns:
        Styled DataFrame
    """

    def style_cell(val, sentiment_type):
        # Extract percentage value
        if isinstance(val, str) and '%' in val:
            percentage = float(val.replace('%', ''))
        else:
            percentage = float(val)

        bg_color = get_aspect_heatmap_color(sentiment_type, percentage)
        text_color = get_text_color_for_background(bg_color)

        return f'background-color: {bg_color}; color: {text_color}; font-weight: bold; text-align: center;'

    # Apply styling
    styled_df = df.style

    for sentiment_type, column_name in sentiment_columns.items():
        if column_name in df.columns:
            styled_df = styled_df.applymap(
                lambda x: style_cell(x, sentiment_type),
                subset=[column_name]
            )

    return styled_df