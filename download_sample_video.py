#!/usr/bin/env python3
# Script to download a sample video for testing
import os
import sys
import argparse
import urllib.request
import shutil

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
    
    # URL of a sample driving video (this is a reliable public domain video)
    video_url = "https://storage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4"
    
    # Output path
    output_path = os.path.join(args.output_dir, "sample_video.mp4")
    
    print(f"Downloading sample video from {video_url}")
    print(f"Saving to {output_path}")
    
    # Download the video
    try:
        with urllib.request.urlopen(video_url) as response, open(output_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        print("Download complete!")
    except Exception as e:
        print(f"Error downloading video: {e}")
        sys.exit(1)
    
    print(f"\nYou can now test the SfM initialization with:")
    print(f"./test_sfm.py --input {output_path}")

if __name__ == "__main__":
    main()
