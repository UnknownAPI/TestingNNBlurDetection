from PIL import Image, ImageDraw
import numpy as np
import random

from cv2 import GaussianBlur


def generate_random_shapes_image():
    # Créer une image 128x128 en niveaux de gris
    img = Image.new("L", (128, 128))
    draw = ImageDraw.Draw(img)

    # Fond aléatoire (valeur entre 0 et 255 pour niveaux de gris)
    bg_color = random.randint(0, 255)
    draw.rectangle((0, 0, 128, 128), fill=bg_color)

    # Nombre aléatoire de formes (1 à 10)
    num_shapes = random.randint(1, 5)

    for _ in range(num_shapes):
        # Types de formes possibles
        shape_types = [
            "square",
            "triangle",
            "rectangle",
            "ring",
            "circle",
            "oval",
            "pentagon",
            "hexagon",
        ]
        shape_type = random.choice(shape_types)

        # Propriétés aléatoires
        is_filled = random.choice([True, False])  # Plein ou creux
        thickness = random.randint(1, 5) if not is_filled else 0  # Épaisseur si creux
        color = random.randint(0, 255)  # Couleur en niveaux de gris
        width = random.randint(10, 50)  # Largeur
        height = (
            random.randint(10, 50)
            if shape_type not in ["square", "circle", "pentagon", "hexagon"]
            else width
        )  # Hauteur
        center_x = random.randint(width // 2, 128 - width // 2)  # Position X
        center_y = random.randint(height // 2, 128 - height // 2)  # Position Y
        angle = random.uniform(0, 360)  # Angle en degrés

        # Calculer le bounding box ou les points
        if shape_type in ["square", "rectangle"]:
            # Définir les coins du rectangle
            half_w, half_h = width / 2, height / 2
            points = [
                (center_x - half_w, center_y - half_h),
                (center_x + half_w, center_y - half_h),
                (center_x + half_w, center_y + half_h),
                (center_x - half_w, center_y + half_h),
            ]
            # Appliquer la rotation
            rad = np.radians(angle)
            cos_a, sin_a = np.cos(rad), np.sin(rad)
            rotated_points = [
                (
                    center_x + (x - center_x) * cos_a - (y - center_y) * sin_a,
                    center_y + (x - center_x) * sin_a + (y - center_y) * cos_a,
                )
                for x, y in points
            ]
            if is_filled:
                draw.polygon(rotated_points, fill=color)
            else:
                draw.polygon(rotated_points, outline=color, width=thickness)

        elif shape_type == "triangle":
            # Définir un triangle équilatéral
            r = width / np.sqrt(3)  # Rayon pour triangle équilatéral
            points = [
                (
                    center_x + r * np.cos(np.radians(angle + i * 120)),
                    center_y + r * np.sin(np.radians(angle + i * 120)),
                )
                for i in range(3)
            ]
            if is_filled:
                draw.polygon(points, fill=color)
            else:
                draw.polygon(points, outline=color, width=thickness)

        elif shape_type == "pentagon":
            # Définir un pentagone régulier
            r = width / (2 * np.sin(np.pi / 5))  # Rayon pour pentagone
            points = [
                (
                    center_x + r * np.cos(np.radians(angle + i * 72)),
                    center_y + r * np.sin(np.radians(angle + i * 72)),
                )
                for i in range(5)
            ]
            if is_filled:
                draw.polygon(points, fill=color)
            else:
                draw.polygon(points, outline=color, width=thickness)

        elif shape_type == "hexagon":
            # Définir un hexagone régulier
            r = width / 2  # Rayon pour hexagone
            points = [
                (
                    center_x + r * np.cos(np.radians(angle + i * 60)),
                    center_y + r * np.sin(np.radians(angle + i * 60)),
                )
                for i in range(6)
            ]
            if is_filled:
                draw.polygon(points, fill=color)
            else:
                draw.polygon(points, outline=color, width=thickness)

        elif shape_type == "circle":
            # Dessiner un cercle
            left = center_x - width // 2
            top = center_y - width // 2
            right = center_x + width // 2
            bottom = center_y + width // 2
            if is_filled:
                draw.ellipse((left, top, right, bottom), fill=color)
            else:
                draw.ellipse((left, top, right, bottom), outline=color, width=thickness)

        elif shape_type == "oval":
            # Dessiner un ovale (ellipse avec largeur et hauteur différentes)
            left = center_x - width // 2
            top = center_y - height // 2
            right = center_x + width // 2
            bottom = center_y + height // 2
            if is_filled:
                draw.ellipse((left, top, right, bottom), fill=color)
            else:
                draw.ellipse((left, top, right, bottom), outline=color, width=thickness)

        elif shape_type == "ring":
            # Dessiner un anneau (toujours creux)
            left = center_x - width // 2
            top = center_y - width // 2
            right = center_x + width // 2
            bottom = center_y + width // 2
            draw.ellipse((left, top, right, bottom), outline=color, width=thickness)

    return img


def blur_image(image: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur to an image using OpenCV.

    Args:
        image (np.ndarray): The image to be blurred.
        sigma (float): The Gaussian blur standard deviation.

    Returns:
        np.ndarray: The blurred image.
    """

    blurred = image.copy()
    kernel_size = int(6 * sigma + 1) | 1
    blurred = GaussianBlur(blurred, (kernel_size, kernel_size), sigma)
    return blurred