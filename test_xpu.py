import os
import subprocess
import sys

def run_command(command):
    """Runs a command and prints its output."""
    print(f"Running command: {command}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, text=True)
    for line in process.stdout:
        sys.stdout.write(line)
    process.wait()
    return process.returncode

def main():
    """Main function to run the XPU test."""
    # 1. Install dependencies
    print("Installing dependencies...")
    if run_command("pip install -r requirements.txt") != 0:
        print("Error installing base requirements.")
        sys.exit(1)
    if run_command("pip install -r requirements-xpu.txt") != 0:
        print("Error installing XPU requirements.")
        sys.exit(1)

    # 2. Check for XPU device
    try:
        import torch
        import intel_extension_for_pytorch as ipex
        if not ipex.xpu.is_available():
            print("No XPU device found. Skipping test.")
            sys.exit(0)
        print(f"Found {ipex.xpu.device_count()} XPU devices.")
    except ImportError:
        print("Intel Extension for PyTorch not found. Skipping test.")
        sys.exit(0)
    except Exception as e:
        print(f"An error occurred while checking for XPU devices: {e}")
        sys.exit(1)


    # 3. Create dummy data file (already created, but good to have it here for completeness)
    if not os.path.exists("dummy_video_paths.csv"):
        with open("dummy_video_paths.csv", "w") as f:
            f.write("/dummy/video.mp4")

    # 4. Run the training
    print("Launching XPU test training...")
    train_command = (
        "python -m app.main "
        "--fname configs/train/vitl16/pretrain-xpu-test.yaml "
        "--device_type xpu "
        "--devices xpu:0"
    )

    return_code = run_command(train_command)
    if return_code == 0:
        print("XPU test training completed successfully.")
    else:
        print(f"XPU test training failed with return code {return_code}.")
        sys.exit(1)

if __name__ == "__main__":
    main()
