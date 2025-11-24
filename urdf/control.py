import time
import mujoco
import mujoco.viewer
import os

def main():
    # Load the URDF directly (now contains actuators and textures)
    xml_path = "robot.urdf"
    xml_path = os.path.abspath(xml_path)
    print(f"Loading model from {xml_path}...")

    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
    except ValueError as e:
        print(f"Error loading model: {e}")
        return

    data = mujoco.MjData(model)

    print("Launching MuJoCo viewer...")
    print("Use the 'Control' panel in the viewer to adjust joint motors independently.")
    
    # Launch the viewer in passive mode but WITHOUT the sine wave override.
    # This allows the user to use the viewer's built-in control sliders.
    mujoco.viewer.launch(model, data)

if __name__ == "__main__":
    main()
