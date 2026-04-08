import mujoco
import mujoco.viewer
import numpy as np
import time
from scipy.spatial.transform import Rotation as R

XML_FILE = "Drone_xml/scene.xml" 

class QuadrotorEnv:
    def __init__(self, xml_path, real_time_factor=1.0):
        # 1. Load model from external file
        try:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
        except Exception as e:
            print(f"Error loading XML: {e}")
            raise

        self.data = mujoco.MjData(self.model)
        self.real_time_factor = real_time_factor
        self.dt = self.model.opt.timestep

        # 2. Launch Passive Viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        # In this XML, the body is named "cf2"
        self.body_name = "Drone"
        self.body_id = self.model.body(self.body_name).id
        
        # Configure Camera Tracking
        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.viewer.cam.trackbodyid = self.body_id

        # 3. Extract Physical Parameters
        self.mass = self.model.body_mass[self.body_id]
        self.inertia = self.model.body_inertia[self.body_id]

        print("\n" + "="*40)
        print(f"MODEL LOADED: {xml_path}")
        print(f"Total Mass: {self.mass:.4f} kg")
        print(f"Inertia: Ixx={self.inertia[0]:.6e}, Iyy={self.inertia[1]:.6e}, Izz={self.inertia[2]:.6e}")
        print(f"Control Inputs: {self.model.nu} (Thrust, Roll_M, Pitch_M, Yaw_M)")
        print("="*40 + "\n")

    def get_state(self):
        """
        Extracts the state: pos, lin_vel, roll, p, pitch, q, yaw, r
        """
        # Get orientation from the freejoint (qpos indices 3,4,5,6 are quat wxyz)
        # Note: Indexing for a single freejoint body starts at 0
        quat_wxyz = self.data.qpos[3:7]
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        
        rotation = R.from_quat(quat_xyzw)
        roll, pitch, yaw = rotation.as_euler('xyz', degrees=False)

        # Angular velocity in body frame (qvel indices 3,4,5)
        # p = rollrate, q = pitchrate, r = yawrate
        ang_vel = self.data.qvel[3:6]

        # Global Position (qpos 0,1,2)
        pos = self.data.qpos[0:3]

        # Global Linear Velocity (qvel 0,1,2)
        lin_vel = self.data.qvel[0:3]

        return pos, lin_vel, roll, ang_vel[0], pitch, ang_vel[1], yaw, ang_vel[2]

    def step(self, ctrl_values):
        """
        Apply controls and advance simulation.
        ctrl_values = [Thrust, X_Moment, Y_Moment, Z_Moment]
        """
        step_start = time.time()
        
        # Apply control to actuators
        self.data.ctrl[:] = ctrl_values
        
        # Step the physics
        mujoco.mj_step(self.model, self.data)

        # Update Viewer
        if self.viewer.is_running():
            self.viewer.sync()
        else:
            return None

        # Real-time synchronization
        elapsed = time.time() - step_start
        expected = self.dt / self.real_time_factor
        if elapsed < expected:
            time.sleep(expected - elapsed)

        return self.get_state()

    def reset(self, height=0.1):
        """Resets the drone to a specific height."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = height # Set initial Z
        mujoco.mj_forward(self.model, self.data)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

# ============================================================
#                 MPC INTEGRATION TEST
# ============================================================

if __name__ == "__main__":
    # Path to your new XML file
    XML_FILE = "Drone_xml/scene.xml" 
    
    
    env = QuadrotorEnv(XML_FILE)

    # Hover equilibrium for THIS model (from keyframe in XML)
    # Total mass 0.027 * 9.81 = 0.26487 N
    hover_thrust = 0.027 * 9.81 
    
    sim_time = 0.0
    duration = 50.0

    try:
        while sim_time < duration:
            # Get Current State
            state = env.get_state()
            pos, vel, roll, p, pitch, q, yaw, r = state

            # EXAMPLE: Slight Pitch Moment to move forward
            # ctrl = [Thrust, Roll_M, Pitch_M, Yaw_M]
            U_total = np.array([hover_thrust, 0.0, 0.0, 0])

            # Step simulation
            if env.step(U_total) is None:
                break


            # 3. Comprehensive Print Statement
            # We use f-string formatting to keep the columns aligned
            print(f"Time: {sim_time:.2f}s | "
                  f"Pos: X={pos[0]:.2f}, Y={pos[1]:.2f}, Z={pos[2]:.2f} | "
                  f"Roll={np.degrees(roll):.2f}° ({np.degrees(p):.2f}°/s) | "
                  f"Pitch={np.degrees(pitch):.2f}° ({np.degrees(q):.2f}°/s) | "
                  f"Yaw={np.degrees(yaw):.2f}° ({np.degrees(r):.2f}°/s)", end="\r")
            
            sim_time += env.dt

    finally:
        env.close()