"""
============================================================================
Interactive Contact Point Selector with MuJoCo Visualization
============================================================================
Visualizes contact points as markers and lets you select which ones to keep.

Usage:
    python debug_1.py

Controls:
    - DOUBLE-CLICK near a marker to toggle selection
    - Selected markers turn GREEN, unselected are RED
    - Press 'S' to SAVE selected points as collision spheres
    - Press 'A' to select ALL
    - Press 'C' to CLEAR all selections
    - Press 'R' to RESET robot position
    - ESC to quit
============================================================================
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
import glfw


class InteractiveContactSelector:
    def __init__(self):
        self.model_path = "urdf/robot_converted.xml"
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        
        self.contact_points = []
        self.selected = set()
        
        # Mouse state
        self.last_click_time = 0
        self.click_pos = None
        
    def collect_contacts(self):
        """Run simulation to find contact points."""
        print("\nCollecting contact points...")
        
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.25
        
        seen = set()
        
        for step in range(3000):
            if step % 100 == 0:
                self.data.ctrl[:] = np.random.uniform(-0.4, 0.4, self.model.nu)
            
            mujoco.mj_step(self.model, self.data)
            
            floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
            
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                
                if contact.geom1 == floor_id:
                    robot_geom = contact.geom2
                elif contact.geom2 == floor_id:
                    robot_geom = contact.geom1
                else:
                    continue
                
                body_id = self.model.geom_bodyid[robot_geom]
                body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                
                world_pos = contact.pos.copy()
                body_xpos = self.data.xpos[body_id]
                body_xmat = self.data.xmat[body_id].reshape(3, 3)
                local_pos = body_xmat.T @ (world_pos - body_xpos)
                
                key = (body_id, round(local_pos[0], 2), round(local_pos[1], 2), round(local_pos[2], 2))
                
                if key not in seen:
                    seen.add(key)
                    self.contact_points.append({
                        'body_id': body_id,
                        'body_name': body_name,
                        'local_pos': local_pos.copy(),
                    })
            
            if self.data.qpos[2] < 0.02 or step % 400 == 0:
                mujoco.mj_resetData(self.model, self.data)
                self.data.qpos[2] = 0.15
                q = [1, 0.1*np.random.randn(), 0.1*np.random.randn(), 0]
                self.data.qpos[3:7] = q / np.linalg.norm(q)
        
        print(f"Found {len(self.contact_points)} contact points")
        
        by_body = defaultdict(int)
        for cp in self.contact_points:
            by_body[cp['body_name']] += 1
        for name, count in sorted(by_body.items()):
            print(f"  {name}: {count}")
    
    def get_world_positions(self):
        """Get current world positions of all contact points."""
        positions = []
        for cp in self.contact_points:
            body_id = cp['body_id']
            body_xpos = self.data.xpos[body_id]
            body_xmat = self.data.xmat[body_id].reshape(3, 3)
            world_pos = body_xpos + body_xmat @ cp['local_pos']
            positions.append(world_pos)
        return np.array(positions)
    
    def save_selected(self):
        """Save selected points as 1mm collision spheres."""
        if not self.selected:
            print("No points selected!")
            return
        
        tree = ET.parse(self.model_path)
        root = tree.getroot()
        
        # Disable mesh collision
        for geom in root.iter('geom'):
            if geom.get('type') == 'mesh':
                geom.set('contype', '0')
                geom.set('conaffinity', '0')
        
        body_elements = {b.get('name'): b for b in root.iter('body') if b.get('name')}
        
        count = 0
        for idx in sorted(self.selected):
            cp = self.contact_points[idx]
            if cp['body_name'] not in body_elements:
                continue
            
            sphere = ET.SubElement(body_elements[cp['body_name']], 'geom')
            sphere.set('name', f"col_{cp['body_name']}_{count}")
            sphere.set('type', 'sphere')
            sphere.set('size', '0.001')  # 1mm
            pos = cp['local_pos']
            sphere.set('pos', f"{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}")
            sphere.set('rgba', '0 1 0 0.5')
            count += 1
        
        output_path = "urdf/robot_collision_spheres.xml"
        tree.write(output_path)
        print(f"\n✓ Saved {count} collision spheres (1mm) to: {output_path}")
    
    def run(self):
        """Run the interactive viewer."""
        self.collect_contacts()
        
        if not self.contact_points:
            print("No contact points found!")
            return
        
        print("\n" + "=" * 60)
        print("INTERACTIVE VIEWER")
        print("=" * 60)
        print("Controls:")
        print("  DOUBLE-CLICK near a marker to toggle selection")
        print("  'A' = select ALL    'C' = CLEAR all")
        print("  'S' = SAVE selected as collision spheres")
        print("  'R' = RESET robot    ESC = quit")
        print("=" * 60)
        print(f"\nShowing {len(self.contact_points)} contact points as markers")
        print("RED = unselected, GREEN = selected")
        
        # Reset robot
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.15
        
        # Create scene for custom rendering
        scene = mujoco.MjvScene(self.model, maxgeom=1000)
        
        with mujoco.viewer.launch_passive(
            self.model, 
            self.data,
            key_callback=self.key_callback,
        ) as viewer:
            
            while viewer.is_running():
                # Don't step physics - keep robot still
                mujoco.mj_forward(self.model, self.data)
                
                # Get contact point positions
                positions = self.get_world_positions()
                
                # Add visual markers for contact points
                viewer.user_scn.ngeom = 0
                for i, pos in enumerate(positions):
                    if i >= 500:  # Max geoms
                        break
                    
                    if i in self.selected:
                        rgba = [0, 1, 0, 0.9]  # Green
                        size = 0.006
                    else:
                        rgba = [1, 0, 0, 0.6]  # Red
                        size = 0.004
                    
                    # Add sphere geom
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[i],
                        mujoco.mjtGeom.mjGEOM_SPHERE,
                        [size, 0, 0],
                        pos,
                        np.eye(3).flatten(),
                        rgba
                    )
                    viewer.user_scn.ngeom = i + 1
                
                viewer.sync()
                time.sleep(0.02)
        
        print(f"\nViewer closed. {len(self.selected)} points selected.")
        
        if self.selected:
            self.save_selected()
    
    def key_callback(self, key):
        """Handle keyboard input."""
        if key == ord('A') or key == ord('a'):
            # Select all
            self.selected = set(range(len(self.contact_points)))
            print(f"Selected ALL ({len(self.selected)} points)")
        
        elif key == ord('C') or key == ord('c'):
            # Clear selection
            self.selected.clear()
            print("Cleared all selections")
        
        elif key == ord('S') or key == ord('s'):
            # Save
            self.save_selected()
        
        elif key == ord('R') or key == ord('r'):
            # Reset robot
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[2] = 0.15
            print("Robot reset")
        
        elif key >= ord('1') and key <= ord('9'):
            # Toggle by body index
            idx = key - ord('1')
            by_body = defaultdict(list)
            for i, cp in enumerate(self.contact_points):
                by_body[cp['body_name']].append(i)
            
            body_names = sorted(by_body.keys())
            if idx < len(body_names):
                body = body_names[idx]
                points = by_body[body]
                
                # Toggle
                if all(p in self.selected for p in points):
                    for p in points:
                        self.selected.discard(p)
                    print(f"Deselected {body} ({len(points)} points)")
                else:
                    self.selected.update(points)
                    print(f"Selected {body} ({len(points)} points)")


def main():
    print("=" * 60)
    print("INTERACTIVE CONTACT POINT SELECTOR")
    print("=" * 60)
    
    selector = InteractiveContactSelector()
    selector.run()


if __name__ == "__main__":
    main()
