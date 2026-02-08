import numpy as np
import cv2

def generate_synthetic_data(num_samples=100, img_size=(128, 128)):
    """
    Generates synthetic satellite-like images with roads.
    Returns:
        images: (num_samples, H, W, 3) - RGB images
        masks: (num_samples, H, W) - Binary masks (1=road, 0=background)
    """
    images = []
    masks = []

    for _ in range(num_samples):
        # 1. Background (Green/Brown noise for terrain)
        base_color = np.random.randint(50, 150, (1, 1, 3)) # Earthy tones
        noise = np.random.randint(-20, 20, (img_size[0], img_size[1], 3))
        bg = np.clip(base_color + noise, 0, 255).astype(np.uint8)
        
        # 2. Draw Roads (Grey lines)
        mask = np.zeros(img_size, dtype=np.uint8)
        
        # Random starting and ending points for roads
        num_roads = np.random.randint(1, 4)
        for _ in range(num_roads):
            pt1 = (np.random.randint(0, img_size[1]), np.random.randint(0, img_size[0]))
            pt2 = (np.random.randint(0, img_size[1]), np.random.randint(0, img_size[0]))
            thickness = np.random.randint(3, 8)
            
            # Draw on image (Grey color)
            cv2.line(bg, pt1, pt2, (100, 100, 100), thickness)
            
            # Draw on mask (White)
            cv2.line(mask, pt1, pt2, 1, thickness)
            
        images.append(bg)
        masks.append(mask)

    return np.array(images), np.array(masks)
