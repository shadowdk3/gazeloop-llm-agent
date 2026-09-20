def check_box_aspect_ratio(x1, y1, x2, y2):
    """
    Tier 1 Geometric Check: Bounding box aspect ratio.
    When standing: Height > Width (ratio < 1.0)
    When fallen/lying: Width > Height (ratio > 1.0)
    """
    width = x2 - x1
    height = y2 - y1
    
    if height == 0:
        return False
        
    aspect_ratio = width / height  # Width divided by Height
    
    # If width is greater than height, the person is horizontal
    if aspect_ratio > 0.7:  # Threshold buffer
        return True
        
    return False