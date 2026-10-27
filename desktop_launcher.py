"""
========================================================================================
🎮 APEX RETENTION AI • DESKTOP APPLICATION LAUNCHER
Launches the full Emerald-Gold animated platform inside a dedicated standalone
desktop application window (borderless, zero URL bar, native taskbar integration).
========================================================================================
"""

import os
import sys
import time
import socket
import subprocess
import webbrowser

PORT = 8501
HOST = "localhost"

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((HOST, port)) == 0

def find_edge_path():
    paths = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Edge\Application\msedge.exe"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def main():
    print("="*65)
    print("  🎮 STARTING APEX RETENTION AI • DESKTOP APPLICATION")
    print("="*65)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

    server_proc = None
    if not is_port_in_use(PORT):
        print("• Initializing ApexRetention AI Neural Backend...")
        cmd = [
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", str(PORT),
            "--server.headless", "true",
            "--browser.serverAddress", HOST,
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false"
        ]
        server_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Wait until port is active
        retries = 25
        while retries > 0 and not is_port_in_use(PORT):
            time.sleep(0.4)
            retries -= 1
        print("✓ Server online on port", PORT)
    else:
        print("✓ Existing engine instance detected on port", PORT)

    app_url = f"http://{HOST}:{PORT}"
    browser_exe = find_edge_path()

    if browser_exe:
        print(f"• Launching dedicated Desktop Application Window ({os.path.basename(browser_exe)} App Mode)...")
        app_flags = [
            browser_exe,
            f"--app={app_url}",
            "--window-size=1380,880",
            "--window-position=50,30"
        ]
        gui_proc = subprocess.Popen(app_flags)
        print("✓ Desktop application window opened!")
        gui_proc.wait()
        print("• Desktop application window closed by user.")
    else:
        print("• Opening in default browser window...")
        webbrowser.open(app_url)

    if server_proc:
        print("• Shutting down background engine...")
        server_proc.terminate()
        server_proc.wait(timeout=3)
        print("✓ Shutdown complete.")

if __name__ == "__main__":
    main()

