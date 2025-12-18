import glob
from PIL import Image
import os

def create_gif(image_folder='images', output_file='cluster_history.gif', duration=500):
    """
    Reads all PNGs from image_folder, sorts them by time, and saves a GIF.
    duration: milliseconds per frame
    """
    # 1. Find all images
    file_pattern = os.path.join(image_folder, "*.png")
    images = glob.glob(file_pattern)
    
    # 2. Sort by filename (which contains the timestamp) ensures chronological order
    images.sort()

    if not images:
        print(f"No images found in {image_folder}")
        return

    print(f"Found {len(images)} frames. Creating GIF...")

    # 3. Load images using Pillow
    frames = [Image.open(image) for image in images]

    # 4. Save as GIF
    # duration is in milliseconds (500ms = 0.5 seconds per frame)
    frame_one = frames[0]
    frame_one.save(
        output_file,
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=duration,
        loop=0 # 0 means loop forever
    )
    
    print(f"GIF saved successfully: {output_file}")

if __name__ == "__main__":
    create_gif()
