#!/usr/bin/env python3
# Script to download a sample video for testing
import os
import sys
import argparse
import subprocess

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Download a sample video for testing')
    parser.add_argument('--output-dir', type=str, default='data', help='Directory to save the video')
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_arguments()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Output path
    output_path = os.path.join(args.output_dir, "sample_driving.mp4")

    # Use youtube-dl to download a sample driving video from YouTube
    # This is a short driving video from a public domain source
    youtube_url = "https://www.youtube.com/watch?v=1EiC9bvVGnk"

    print(f"Downloading sample video from {youtube_url}")
    print(f"Saving to {output_path}")

    # Check if youtube-dl is installed
    try:
        subprocess.run(["which", "youtube-dl"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        print("Error: youtube-dl is not installed. Please install it with:")
        print("pip install youtube-dl")
        sys.exit(1)

    # Download the video
    try:
        subprocess.run(["youtube-dl", "-f", "mp4", "-o", output_path, youtube_url], check=True)
        print("Download complete!")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading video: {e}")
        sys.exit(1)

    print(f"\nYou can now test the SfM initialization with:")
    print(f"python -m tests.test_sfm --input {output_path}")

if __name__ == "__main__":
    main()
