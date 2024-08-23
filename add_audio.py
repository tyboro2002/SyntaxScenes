import os
import random
from moviepy.editor import VideoFileClip, AudioFileClip


def load_files_from_directory(directory, extensions):
    """Load all files with the specified extensions from a directory."""
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(extensions)]


def add_audio_to_video(video_path, audio_path, output_dir):
    """Add audio to a video and save the result."""
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)

    # Check if the audio is long enough to cover the video duration
    if audio.duration < video.duration:
        return f"Warning: Audio file '{audio_path}' is too short for video '{video_path}' (Video duration: {video.duration}, Audio duration: {audio.duration})"

    # Trim the audio to match the video duration
    audio = audio.subclip(0, video.duration)

    # Set the new audio to the video
    video = video.set_audio(audio)

    # Create the output file path
    video_filename = os.path.basename(video_path)
    output_path = os.path.join(output_dir, f"{video_filename}")

    # Write the final video file
    video.write_videofile(output_path, codec='libx264', audio_codec='aac')

    return None  # No warning means everything is okay


def main(video_dir, audio_dir, output_dir):
    # Load video and audio files
    video_files = load_files_from_directory(video_dir, ('.mp4', '.avi', '.mov'))
    audio_files = load_files_from_directory(audio_dir, ('.mp3', '.wav'))

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    warnings = []

    # Loop over video files and add a random audio file to each
    for video_file in video_files:
        # Select a random audio file
        audio_file = random.choice(audio_files)

        # Add audio to the video and capture any warnings
        warning = add_audio_to_video(video_file, audio_file, output_dir)
        if warning:
            warnings.append(warning)

    # Print all warnings
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(warning)
    else:
        print("All videos processed successfully!")

    print("Processing complete!")


if __name__ == "__main__":
    video_directory = "audio_add/videos"  # Replace with your video directory
    audio_directory = "audio_add/songs"  # Replace with your audio directory
    output_directory = "audio_add/output"  # Replace with your output directory

    main(video_directory, audio_directory, output_directory)
