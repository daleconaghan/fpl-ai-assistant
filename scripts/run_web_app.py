#!/usr/bin/env python3
"""
Quick launch script for FPL AI Assistant web interface
"""

import subprocess
import sys
import webbrowser
import time
from pathlib import Path

def main():
    """Launch the FPL AI Assistant web interface."""
    
    print("🚀 Launching FPL AI Assistant...")
    
    # Navigate to project directory
    project_root = Path(__file__).parent.parent
    
    try:
        # Start Streamlit app
        print("📡 Starting web server...")
        
        # Try different ports
        ports = [8502, 8503, 8504, 8505]
        
        for port in ports:
            try:
                print(f"   Trying port {port}...")
                
                # Run streamlit
                process = subprocess.Popen([
                    sys.executable, "-m", "streamlit", "run", 
                    str(project_root / "app.py"),
                    "--server.port", str(port),
                    "--server.headless", "false"
                ], cwd=project_root)
                
                # Wait a moment for startup
                time.sleep(3)
                
                # Check if process is still running
                if process.poll() is None:
                    print(f"✅ Web server started on port {port}")
                    print(f"🌐 Opening http://localhost:{port}")
                    
                    # Open browser
                    webbrowser.open(f"http://localhost:{port}")
                    
                    print("\n" + "="*60)
                    print("🎯 FPL AI Assistant is now running!")
                    print(f"📱 Web interface: http://localhost:{port}")
                    print("🛑 Press Ctrl+C to stop the server")
                    print("="*60)
                    
                    # Wait for user to stop
                    try:
                        process.wait()
                    except KeyboardInterrupt:
                        print("\n🛑 Stopping server...")
                        process.terminate()
                        process.wait()
                        print("✅ Server stopped")
                    
                    return
                else:
                    print(f"❌ Failed to start on port {port}")
                    
            except Exception as e:
                print(f"❌ Error starting on port {port}: {e}")
                continue
        
        print("❌ Failed to start web server on any port")
        
    except Exception as e:
        print(f"❌ Error launching app: {e}")
        return 1

if __name__ == "__main__":
    main()