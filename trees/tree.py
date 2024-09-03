import os

from treesSettings import trees_base_path, FPS


class Tree:
    def __init__(self, frames_dir='frames', command=None):
        self.frames_dir = frames_dir
        if command is None:
            command = [
                'ffmpeg', '-y', '-framerate', str(FPS), '-i', os.path.join(self.frames_dir, 'frame_%03d.png'),
                '-r', str(FPS), '-pix_fmt', 'yuv420p',
                f'{trees_base_path}{self.__class__.__name__.lower()}_tree_animation.mp4'
            ]
        self.command = command
        self.step = 0

    def clear_frames_directory(self):
        # Remove all files in the frames directory
        for filename in os.listdir(self.frames_dir):
            file_path = os.path.join(self.frames_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Deleted file {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

        # Optionally, remove the directory itself
        os.rmdir(self.frames_dir)
        print(f"Cleared and removed frames directory: {self.frames_dir}")

    def compile_frames_to_video(self):
        import subprocess
        print(f"Compiling frames into video with {FPS} FPS...")
        # Command to convert images to video using ffmpeg

        subprocess.run(self.command, check=True)
        print(f"Video saved as {trees_base_path}{self.__class__.__name__.lower()}_tree_animation.mp4")
