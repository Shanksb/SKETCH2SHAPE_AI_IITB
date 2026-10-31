
import sys
from pathlib import Path

from PIL import Image, ImageDraw

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import prepare_sketch, prepare_for_3d


# Create a sample hand-drawn sketch
def create_sample_sketch():
    img = Image.new("RGB", (400, 400), "white")
    draw = ImageDraw.Draw(img)

    # Draw a simple house
    draw.rectangle((120, 180, 280, 320), outline="black", width=5)
    draw.line((100, 180, 200, 90), fill="black", width=5)
    draw.line((200, 90, 300, 180), fill="black", width=5)
    draw.rectangle((175, 240, 225, 320), outline="black", width=5)

    return img


def main():
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    sample = create_sample_sketch()

    # Test 1: Prepare sketch
    print("Testing prepare_sketch...")

    sketch = prepare_sketch(sample)

    sketch_path = output_dir / "preprocessed_sketch.png"
    sketch.save(sketch_path)

    print("Sketch preprocessing successful!")
    print("Output size:", sketch.size)
    print("Saved at:", sketch_path)

    # Test 2: Prepare image for 3D reconstruction
    print("\nTesting prepare_for_3d...")

    prepared_image = prepare_for_3d(sample)

    image_path = output_dir / "prepared_for_3d.png"
    prepared_image.save(image_path)

    print("3D image preparation successful!")
    print("Output size:", prepared_image.size)
    print("Saved at:", image_path)

    # Test 3: Check blank sketch detection
    print("\nTesting blank sketch detection...")

    blank = Image.new("RGB", (400, 400), "white")

    try:
        prepare_sketch(blank)
        print("ERROR: Blank sketch was not detected.")

    except ValueError as error:
        print("Blank sketch correctly detected:")
        print(error)

    print("\nAll preprocessing tests completed.")


if __name__ == "__main__":
    main()
