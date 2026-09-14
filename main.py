import argparse
from src.agent import VisionAgent
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# Full path to the current script file
project_path = Path(__file__).resolve().parent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GazeLoop - Intelligent Vision Monitoring Agent"
    )
        
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help=(
            "Input source: camera index (e.g. '0'), video file path (e.g."
            " 'data/demo.mp4'), or image path. Leave empty to use default images."
        ),
    )
    args = parser.parse_args()
    
    if args.input is not None:
        # Camera index (e.g., --input 0)
        if args.input.isdigit():
            camera_id = int(args.input)
            agent = VisionAgent()
            agent.analyze_stream(source=camera_id)

        # Video file (e.g., --input data/demo.mp4)
        elif args.input.lower().endswith(
                (".mp4", ".avi", ".mov", ".mkv", ".webm")
            ):
            agent = VisionAgent()
            agent.analyze_stream(source=args.input)

        # Single image path (e.g., --input data/custom.png)
        else:
            agent = VisionAgent(image_paths=[args.input])
            agent.analyze_images()
    else:
        image1 = os.path.join(project_path, "data", "demo1.png")
        image2 = os.path.join(project_path, "data", "demo2.png")
        images = [image1, image2]
        agent = VisionAgent(image_paths=images)
        agent.analyze_images()
    