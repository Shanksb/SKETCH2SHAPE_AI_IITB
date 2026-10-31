
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from pathlib import Path


# Convert input into a PIL image
def to_pil(image):
    if isinstance(image, Image.Image):
        return image.copy()

    if isinstance(image, (str, Path)):
        return Image.open(image)

    if isinstance(image, np.ndarray):
        return Image.fromarray(image)

    raise TypeError("Input must be a PIL image, file path, or NumPy array.")


# Otsu thresholding: automatically finds a threshold value
def otsu_threshold(gray_array):
    histogram = np.bincount(gray_array.flatten(), minlength=256)

    total_pixels = gray_array.size
    total_sum = np.dot(np.arange(256), histogram)

    background_weight = 0
    background_sum = 0
    maximum_variance = 0
    best_threshold = 127

    for threshold in range(256):
        background_weight += histogram[threshold]

        if background_weight == 0:
            continue

        foreground_weight = total_pixels - background_weight

        if foreground_weight == 0:
            break

        background_sum += threshold * histogram[threshold]

        mean_background = background_sum / background_weight
        mean_foreground = (
            total_sum - background_sum
        ) / foreground_weight

        variance = (
            background_weight
            * foreground_weight
            * (mean_background - mean_foreground) ** 2
        )

        if variance > maximum_variance:
            maximum_variance = variance
            best_threshold = threshold

    return best_threshold


# --------------------------------------------------
# Function 1: Prepare hand-drawn sketch
# --------------------------------------------------
def prepare_sketch(image, size=512, thicken=3):
    """
    Converts a hand-drawn sketch into a clean image
    with white strokes on a black background.
    """

    img = to_pil(image)

    # Handle transparent images by placing them on white
    if img.mode == "RGBA":
        white_bg = Image.new("RGBA", img.size, "white")
        img = Image.alpha_composite(white_bg, img).convert("RGB")
    else:
        img = img.convert("RGB")

    # Resize image while maintaining aspect ratio
    img.thumbnail((size, size))

    # Convert to grayscale
    gray = ImageOps.grayscale(img)

    # Remove small noise
    gray = gray.filter(ImageFilter.MedianFilter(size=3))

    # Improve contrast
    gray = ImageOps.autocontrast(gray)

    # Convert image into NumPy array
    gray_array = np.array(gray)

    # Check average border brightness to identify background
    border = np.concatenate([
        gray_array[0, :],
        gray_array[-1, :],
        gray_array[:, 0],
        gray_array[:, -1]
    ])

    background_is_light = np.median(border) >= 127

    # Automatically calculate threshold
    threshold = otsu_threshold(gray_array)

    # Detect sketch strokes
    if background_is_light:
        ink = gray_array <= threshold
    else:
        ink = gray_array > threshold

    # Check whether sketch is blank or too small
    if np.count_nonzero(ink) < 10:
        raise ValueError(
            "Sketch is blank. Please draw something and try again."
        )

    # Create white strokes on black background
    sketch_array = np.zeros_like(gray_array, dtype=np.uint8)
    sketch_array[ink] = 255

    sketch = Image.fromarray(sketch_array)

    # Crop around the sketch
    bbox = sketch.getbbox()

    if bbox is None:
        raise ValueError("No sketch detected.")

    sketch = sketch.crop(bbox)

    # Thicken the strokes
    if thicken > 1:
        filter_size = thicken if thicken % 2 == 1 else thicken + 1
        sketch = sketch.filter(
            ImageFilter.MaxFilter(size=filter_size)
        )

    # Resize sketch into a square canvas
    sketch.thumbnail((int(size * 0.85), int(size * 0.85)))

    canvas = Image.new("L", (size, size), 0)

    x = (size - sketch.width) // 2
    y = (size - sketch.height) // 2

    canvas.paste(sketch, (x, y))

    return canvas.convert("RGB")


# --------------------------------------------------
# Function 2: Prepare generated image for 3D model
# --------------------------------------------------
def prepare_for_3d(image, size=512, bg_color=(127, 127, 127)):
    """
    Removes background when rembg is available,
    crops the foreground, centers it, and returns
    a square RGB image.
    """

    img = to_pil(image).convert("RGBA")

    # Try to remove background using rembg
    try:
        from rembg import remove

        img = remove(img)

        if not isinstance(img, Image.Image):
            img = Image.open(img)

        img = img.convert("RGBA")

    except ImportError:
        print("Warning: rembg is not installed.")
        print("Using the original image without background removal.")

    # Get foreground mask from alpha channel
    alpha = img.getchannel("A")

    # Find foreground area
    bbox = alpha.getbbox()

    if bbox is None:
        raise ValueError("No foreground object detected.")

    # Crop around the object
    img = img.crop(bbox)

    # Resize while maintaining aspect ratio
    img.thumbnail((int(size * 0.85), int(size * 0.85)))

    # Create square background
    background = Image.new("RGB", (size, size), bg_color)

    # Calculate center position
    x = (size - img.width) // 2
    y = (size - img.height) // 2

    # Paste object using alpha mask
    background.paste(img, (x, y), img.getchannel("A"))

    return background
