import os
import subprocess
import sys

def run_command(command, env=None):
    """Runs a command and prints its output."""
    print(f"Running command: {command}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, text=True, env=env)
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
        num_devices = ipex.xpu.device_count()
        print(f"Found {num_devices} XPU devices.")
    except ImportError:
        print("Intel Extension for PyTorch not found. Skipping test.")
        sys.exit(0)
    except Exception as e:
        print(f"An error occurred while checking for XPU devices: {e}")
        sys.exit(1)

    # 3. Set environment variables for oneCCL
    env = os.environ.copy()
    env.update({
        "CCL_PROCESS_LAUNCHER": "pmix",
        "CCL_ATL_TRANSPORT": "mpi",
        "CCL_ALLREDUCE_SCALEOUT": "direct:0-1048576;rabenseifner:1048577-max",
        "CCL_BCAST": "double_tree",
        "CCL_KVS_MODE": "mpi",
        "CCL_CONFIGURATION_PATH": "",
        "CCL_CONFIGURATION": "cpu_gpu_dpcpp",
        "CCL_KVS_CONNECTION_TIMEOUT": "600",
        "CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD": "1024",
        "CCL_KVS_USE_MPI_RANKS": "1",
    })

    # 4. Run the oneCCL test
    print("Launching oneCCL test...")
    # Use all available devices for the test
    mpirun_command = f"mpirun -np {num_devices} python test_oneccl.py"

    return_code = run_command(mpirun_command, env=env)
    if return_code == 0:
        print("oneCCL test completed successfully.")
    else:
        print(f"oneCCL test failed with return code {return_code}.")
        sys.exit(1)

if __name__ == "__main__":
    main()
