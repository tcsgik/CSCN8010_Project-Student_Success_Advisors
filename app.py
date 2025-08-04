import subprocess
import os

def main():
    # Ensure correct path to the Streamlit script
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    streamlit_script = os.path.join("src", "ui.py")
    
    # Launch Streamlit app
    subprocess.run(["streamlit", "run", streamlit_script], env=env)

if __name__ == "__main__":
    main()
