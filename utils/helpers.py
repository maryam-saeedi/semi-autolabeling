import numpy as np

def generate_unique_colors(num_colors):
    colors = [(200,200,200)]    # every first color is gray
    
    def get_random_color():
        # Generate a random color in HSV space
        r = np.random.randint(0,255)
        g = np.random.randint(0,255)
        b = np.random.randint(0,255)
        return tuple((r,g,b))  # Convert to RGB
    
    def is_far_enough(new_color, existing_colors, min_distance=100):
        # Check if the new color is sufficiently far from all existing colors
        for color in existing_colors:
            dist = np.linalg.norm(np.array(new_color) - np.array(color))
            if dist < min_distance:
                return False
        return True
    
    while len(colors) < num_colors:
        new_color = get_random_color()
        if is_far_enough(new_color, colors):
            colors.append(new_color)
    
    return colors
