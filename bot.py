import mujoco
import mujoco.viewer
import os
import xml.etree.ElementTree as ET
import time

def main():
    # Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(current_dir, "urdf", "robot.urdf")
    temp_urdf_path = os.path.join(current_dir, "urdf", "robot_no_ground.urdf")

    # 1. Modify URDF to remove ground
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Find and remove 'floor' link
    for link in root.findall('link'):
        if link.get('name') == 'floor':
            root.remove(link)
            print("Removed 'floor' link.")

    # Find and remove 'world_to_floor' joint
    for joint in root.findall('joint'):
        if joint.get('name') == 'world_to_floor':
            root.remove(joint)
            print("Removed 'world_to_floor' joint.")

    # Update 'floor_to_base' joint to connect to 'world' instead of 'floor'
    for joint in root.findall('joint'):
        if joint.get('name') == 'floor_to_base':
            parent = joint.find('parent')
            if parent is not None and parent.get('link') == 'floor':
                parent.set('link', 'world')
                print("Updated 'floor_to_base' joint to parent 'world'.")

    # Save modified URDF
    tree.write(temp_urdf_path)
    print(f"Saved modified URDF to {temp_urdf_path}")

    try:
        # 2. Load Model
        print("Loading MuJoCo model...")
        model = mujoco.MjModel.from_xml_path(temp_urdf_path)
        data = mujoco.MjData(model)

        # 3. Disable Gravity
        model.opt.gravity = (0, 0, 0)
        print("Gravity disabled.")

        # 4. Launch Viewer
        print("Launching viewer. You can move joints freely.")
        # launch_passive allows the user to interact while we control the loop if needed, 
        # but launch is simpler for just visualization and standard interaction.
        # However, for "moving joints freely" without physics fighting back (except inertia), 
        # standard simulation with 0 gravity is good.
        # If the user wants to set angles manually without dynamics, they might prefer a paused sim 
        # or a kinematic mode, but MuJoCo is dynamic by default.
        # With 0 gravity and no stiffness/damping (unless specified in URDF), it should be free moving.
        
        mujoco.viewer.launch(model, data)

    finally:
        # Cleanup
        if os.path.exists(temp_urdf_path):
            os.remove(temp_urdf_path)
            print(f"Removed temp file {temp_urdf_path}")

if __name__ == "__main__":
    main()
