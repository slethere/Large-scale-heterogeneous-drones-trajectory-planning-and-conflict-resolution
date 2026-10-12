import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Fixed-wing UAV Model with Reward & Obstacle Avoidance
class FixedWingUAV:
    def __init__(self, callsign='FixedWing-001', craft_type='pure_fixed_wing', weight_N=300,
                 init_state=[0, 0, 0, 15.6, 0, 0, 0, 0], # [x,y,z,vx,vy,vz,heading(rad),roll(rad)]
                 des_state=[1000,1000,1000,15.6,0,0]):
        # Basic Constant - Priority Definition
        self.g = 9.81
        self.max_roll = np.pi/3  # Max Roll Angle ±60°
        
        # Core Base Attribute
        self.callsign = callsign
        self.craft_type = craft_type  # pure_fixed_wing / vtol_tiltrotor / hybrid_fixed_rotor
        self.weight_N = weight_N
        self.init_state = init_state
        self.des_state = des_state
        self.state = init_state.copy()
        self.state_list = []
        
        # Speed Related Attr (1:1 Match Your Doc Data)
        self.min_speed = 9.5        # No stall min speed
        self.cruise_speed = 15.6    # Economic cruise speed (m/s)
        self.max_speed = 40.379 if craft_type != 'pure_fixed_wing' else 51.0
        self.vertical_speed = 0.0 if craft_type == 'pure_fixed_wing' else 4.0
        
        # Safe & Flight Attr
        self.min_turn_radius = self.calc_min_turn_radius()

    def calc_min_turn_radius(self):
        """Calculate min turn radius (speed=50m/s, roll=30°) for pure fixed-wing UAV"""
        if self.craft_type == 'pure_fixed_wing':
            turn_radius = (50**2) / (self.g * np.tan(30 * np.pi / 180))
            return round(turn_radius, 1)
        return 500.0

    def reset(self):
        """Reset UAV to initial state & clear flight history"""
        self.state = self.init_state.copy()
        self.state_list = []

    def hover(self):
        """Hover function: Only VTOL/Tiltrotor/Hybrid support"""
        if self.craft_type != 'pure_fixed_wing':
            self.state[3:8] = [0, 0, 0, 0, 0]

    def get_next_state(self, action):
        """
        Core State Transfer Function (100% Effective)
        action = [delta_heading, delta_roll, throttle]
        Support: Turn / Height Change / Speed Up/Down / State Switch
        """
        d_heading, d_roll, throttle = action
        x, y, z, vx, vy, vz, heading, roll = self.state

        # Speed Control - Throttle Real-time Effective
        curr_speed = np.sqrt(vx**2 + vy**2)
        new_speed = np.clip(curr_speed + throttle * 2, self.min_speed, self.max_speed)
        
        # Heading & Roll Angle Update with Constraint
        new_heading = heading + d_heading
        new_roll = np.clip(roll + d_roll, -self.max_roll, self.max_roll)

        # X/Y Horizontal Position Update
        dx = new_speed * np.cos(new_heading)
        dy = new_speed * np.sin(new_heading)

        # Z Vertical Altitude Update (Different Logic for 3 UAV Type)
        if self.craft_type == 'pure_fixed_wing':
            dz = new_speed * np.sin(new_roll) * 0.4  # Roll -> Height Climb/Descend
        else:
            dz = self.vertical_speed  # VTOL Direct Vertical Lift

        # Update Velocity & Full Flight State
        new_vx = dx
        new_vy = dy
        new_vz = dz
        self.state = [x+dx, y+dy, z+dz, new_vx, new_vy, new_vz, new_heading, new_roll]

    def update_state_list(self):
        """Record real-time flight state to history list"""
        self.state_list.append(self.state.copy())

    def get_1s_state(self, action):
        """Predict next 1s state (same logic as get_next_state but no in-place update)"""
        d_heading, d_roll, throttle = action
        x, y, z, vx, vy, vz, heading, roll = self.state

        curr_speed = np.sqrt(vx**2 + vy**2)
        new_speed = np.clip(curr_speed + throttle * 2, self.min_speed, self.max_speed)
        new_heading = heading + d_heading
        new_roll = np.clip(roll + d_roll, -self.max_roll, self.max_roll)

        dx = new_speed * np.cos(new_heading)
        dy = new_speed * np.sin(new_heading)
        if self.craft_type == 'pure_fixed_wing':
            dz = new_speed * np.sin(new_roll) * 0.4
        else:
            dz = self.vertical_speed

        new_vx = dx
        new_vy = dy
        new_vz = dz
        return [x+dx, y+dy, z+dz, new_vx, new_vy, new_vz, new_heading, new_roll]

    def get_5s_state(self, action):
        """Predict next 5s state by iterating 1s steps"""
        temp_state = self.state.copy()
        for _ in range(5):
            d_heading, d_roll, throttle = action
            x, y, z, vx, vy, vz, heading, roll = temp_state

            curr_speed = np.sqrt(vx**2 + vy**2)
            new_speed = np.clip(curr_speed + throttle * 2, self.min_speed, self.max_speed)
            new_heading = heading + d_heading
            new_roll = np.clip(roll + d_roll, -self.max_roll, self.max_roll)

            dx = new_speed * np.cos(new_heading)
            dy = new_speed * np.sin(new_heading)
            if self.craft_type == 'pure_fixed_wing':
                dz = new_speed * np.sin(new_roll) * 0.4
            else:
                dz = self.vertical_speed

            new_vx = dx
            new_vy = dy
            new_vz = dz
            temp_state = [x+dx, y+dy, z+dz, new_vx, new_vy, new_vz, new_heading, new_roll]
        return temp_state

    def get_closest_ob(self, obdic):
        """Find closest obstacle (same logic as classUAV.py)"""
        d = 999999
        key = 'ob1'
        for i in obdic.keys():
            # obdic format: {ob1:[ob_x, ob_y, ob_hmax, ob_hmin, ob_r]}
            df = np.sqrt((self.state[0]-obdic[i][0])**2 + (self.state[1]-obdic[i][1])**2) - obdic[i][-1]
            if df < d:
                d = df
                key = i
        return obdic[key]

    def get_action_space(self):
        """Generate action space (refer to classUAV.py's action space logic)"""
        # Action: [delta_heading, delta_roll, throttle]
        if -np.pi/6 <= self.state[7] <= np.pi/6 and self.min_speed <= np.sqrt(self.state[3]**2 + self.state[4]**2) <= self.max_speed:
            delta_heading = [-0.1, 0, 0.1]  # rad
            delta_roll = [-0.05, 0, 0.05]   # rad
            throttle = [-0.5, 0, 0.5]       # speed change factor
            actionlis = np.zeros((28, 4))   # 3*3*3 +1 =28, last col for reward
            ss = 0
            for i in range(len(delta_heading)):
                for j in range(len(delta_roll)):
                    for k in range(len(throttle)):
                        actionlis[ss, 0] = delta_heading[i]
                        actionlis[ss, 1] = delta_roll[j]
                        actionlis[ss, 2] = throttle[k]
                        ss += 1
            # Emergency recovery action
            actionlis[27, 0] = -self.state[6]
            actionlis[27, 1] = -self.state[7]
            actionlis[27, 2] = 0
        else:
            # Only emergency recovery action
            actionlis = np.array([[-self.state[6], -self.state[7], 0, 0]])
        return actionlis

# -------------------------- Reward Functions (Optimized) --------------------------
def d_reward(x1, y1, h1, x1_1s, y1_1s, h1_1s, destination1x, destination1y, destination1h):
    """Destination reward: encourage approaching target (Optimized for positive incentive)"""
    D1 = np.sqrt((x1-destination1x)**2 + (y1-destination1y)**2 + (h1-destination1h)**2)
    D2 = np.sqrt((x1_1s-destination1x)**2 + (y1_1s-destination1y)**2 + (h1_1s-destination1h)**2)
    
    # Optimized reward logic: positive for approaching, mild negative for moving away
    if D1 < 20:  # Close to target: big positive reward
        r1 = 100
    elif D2 < D1:  # Approaching target: positive reward
        r1 = 20 * (D1 - D2) / (D1 + 1e-10)
    else:  # Moving away: mild negative reward
        r1 = -5 * (D2 - D1) / (D1 + 1e-10)
    return r1

def o_reward(x1_1s, y1_1s, h1_1s, x1_5s, y1_5s, h1_5s, ob_x, ob_y, ob_hmax, ob_hmin, ob_r, rd):
    """Obstacle avoidance reward: punish collision risk"""
    # Check 1s state collision (severe punishment)
    if np.sqrt((x1_1s - ob_x)**2 + (y1_1s - ob_y)**2) <= (ob_r + rd) and ob_hmin <= h1_1s <= ob_hmax:
        o_r1 = -500
    else:
        o_r1 = 0
    
    # Check 5s state collision (moderate punishment)
    if np.sqrt((x1_5s - ob_x)**2 + (y1_5s - ob_y)**2) <= (ob_r + rd) and ob_hmin <= h1_5s <= ob_hmax:
        o_r2 = -100
    else:
        o_r2 = 0
    
    return o_r1 + o_r2
