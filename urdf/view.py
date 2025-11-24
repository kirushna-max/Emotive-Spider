import time
import mujoco
import mujoco.viewer
import os

def main():
    urdf_path = "robot.urdf"
    urdf_path = os.path.abspath(urdf_path)
    print(f"Loading URDF from {urdf_path}...")

    # Load the URDF model
    try:
        model = mujoco.MjModel.from_xml_path(urdf_path)
    except ValueError as e:
        print(f"Error loading URDF: {e}")
        return

    # Save as MJB
    mjb_path = "model.mjb"
    print(f"Saving MJB to {mjb_path}...")
    mujoco.mj_saveModel(model, mjb_path, None)

    data = mujoco.MjData(model)

    print("Launching MuJoCo viewer...")
    mujoco.viewer.launch(model, data)

if __name__ == "__main__":
    main()
