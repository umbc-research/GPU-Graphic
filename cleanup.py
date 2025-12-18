import os
import glob

def clean_files():
    # define folders to clean
    directories = ['.', 'images']
    # define file types to remove
    extensions = ['*.png', '*.gif']

    print("Starting cleanup...")
    count = 0

    for folder in directories:
        # Skip if folder doesn't exist
        if not os.path.exists(folder):
            continue

        for ext in extensions:
            # Find files matching the pattern (e.g., images/*.png)
            search_pattern = os.path.join(folder, ext)
            files = glob.glob(search_pattern)

            for file_path in files:
                try:
                    os.remove(file_path)
                    print(f"Removed: {file_path}")
                    count += 1
                except OSError as e:
                    print(f"Error removing {file_path}: {e}")

    print(f"-" * 30)
    print(f"Cleanup complete. Removed {count} files.")

if __name__ == "__main__":
    clean_files()
