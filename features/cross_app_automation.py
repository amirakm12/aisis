"""
Cross-App Automation for AISIS
Automate workflows across creative applications
(Photoshop, GIMP, Blender).
"""

import subprocess
from typing import Optional, Dict, Union


def is_com_object(obj) -> bool:
    # Check for COM object by class name and 'Quit' attribute, not a Popen
    return (
        hasattr(obj, 'Quit') and
        obj.__class__.__name__ == 'CDispatch' and
        not isinstance(obj, subprocess.Popen)
    )


class CrossAppAutomation:
    """
    Automates launching, controlling, and exchanging data between creative apps.
    Supports Photoshop (via COM), GIMP (via Python-fu), and Blender (via bpy).
    """
    def __init__(self):
        self.processes: Dict[str, Union[subprocess.Popen, object]] = {}

    def launch_photoshop(self, path: Optional[str] = None) -> bool:
        """
        Launch Adobe Photoshop (Windows only, via COM or executable path).
        """
        try:
            import win32com.client  # type: ignore
            ps_app = win32com.client.Dispatch("Photoshop.Application")
            ps_app.Visible = True
            self.processes["photoshop"] = ps_app  # COM object
            return True
        except ImportError:
            print("pywin32 not installed. Cannot launch Photoshop via COM.")
        except Exception as e:
            print(f"Photoshop launch error: {e}")
        if path:
            try:
                proc = subprocess.Popen([path])
                self.processes["photoshop"] = proc
                return True
            except Exception as e:
                print(f"Photoshop launch error (exe): {e}")
        return False

    def launch_gimp(self, path: Optional[str] = None) -> bool:
        """
        Launch GIMP (via executable path or default command).
        """
        try:
            cmd = [path] if path else ["gimp"]
            proc = subprocess.Popen(cmd)
            self.processes["gimp"] = proc
            return True
        except Exception as e:
            print(f"GIMP launch error: {e}")
            return False

    def launch_blender(self, path: Optional[str] = None) -> bool:
        """
        Launch Blender (via executable path or default command).
        """
        try:
            cmd = [path] if path else ["blender"]
            proc = subprocess.Popen(cmd)
            self.processes["blender"] = proc
            return True
        except Exception as e:
            print(f"Blender launch error: {e}")
            return False

    def send_to_gimp(self, image_path: str) -> bool:
        """
        Open an image in GIMP using command line.
        """
        try:
            cmd = ["gimp", image_path]
            subprocess.Popen(cmd)
            return True
        except Exception as e:
            print(f"Send to GIMP error: {e}")
            return False

    def send_to_blender(self, script_path: str) -> bool:
        """
        Run a Python script in Blender (for automation).
        """
        try:
            cmd = ["blender", "--background", "--python", script_path]
            subprocess.Popen(cmd)
            return True
        except Exception as e:
            print(f"Send to Blender error: {e}")
            return False

    def close_all(self):
        """
        Close all launched applications (where possible).
        """
        for name, proc in self.processes.items():
            try:
                if is_com_object(proc):
                    proc.Quit()
                elif isinstance(proc, subprocess.Popen):
                    proc.terminate()
            except Exception as e:
                print(f"Error closing {name}: {e}")
        self.processes.clear()


if __name__ == "__main__":
    # Demo: Launch GIMP and Blender, then close them
    automation = CrossAppAutomation()
    print("Launching GIMP...")
    automation.launch_gimp()
    print("Launching Blender...")
    automation.launch_blender()
    input("Press Enter to close all apps...")
    automation.close_all() 