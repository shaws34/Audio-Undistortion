import subprocess
import sys

# Function to install required packages
def install_packages():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("All required packages have been installed.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install required packages: {e}")
        sys.exit(1)

if __name__ == "__main__":
    install_packages()
    print("Project initialization complete.")
