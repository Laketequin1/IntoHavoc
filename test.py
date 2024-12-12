import numpy as np

def euler_to_rotation_matrix(roll, pitch, yaw):
    # Compute rotation matrices for each axis
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    # Combined rotation matrix
    return R_z @ R_y @ R_x

def first_collision_with_rotated_cuboid_euler(p1, p2, cuboid_center, cuboid_size, euler_angles):
    # Unpack cuboid properties
    cx, cy, cz = cuboid_center
    w, h, d = cuboid_size
    roll, pitch, yaw = euler_angles

    # Convert Euler angles to rotation matrix
    R = euler_to_rotation_matrix(roll, pitch, yaw)
    R_inv = R.T  # Inverse of rotation matrix is its transpose
    
    # Transform line segment to cuboid's local space
    p1_local = R_inv @ (np.array(p1) - np.array([cx, cy, cz]))
    p2_local = R_inv @ (np.array(p2) - np.array([cx, cy, cz]))
    
    # Axis-aligned bounds in local space
    bounds = [
        (-w / 2, w / 2),  # x-bounds
        (-h / 2, h / 2),  # y-bounds
        (-d / 2, d / 2)   # z-bounds
    ]
    
    # Slab method in local space
    t_entry = 0
    t_exit = 1
    for i in range(3):  # Iterate over x, y, z axes
        p1_axis = p1_local[i]
        p2_axis = p2_local[i]
        min_bound, max_bound = bounds[i]
        
        direction = p2_axis - p1_axis
        if abs(direction) < 1e-8:  # Line is parallel to the axis
            if p1_axis < min_bound or p1_axis > max_bound:
                return None  # No collision
            continue
        
        t_min = (min_bound - p1_axis) / direction
        t_max = (max_bound - p1_axis) / direction
        if t_min > t_max:  # Normalize
            t_min, t_max = t_max, t_min
        
        t_entry = max(t_entry, t_min)
        t_exit = min(t_exit, t_max)
        
        if t_entry > t_exit:
            return None  # No collision
    
    # Compute collision point in local space
    collision_local = p1_local + t_entry * (p2_local - p1_local)
    
    # Transform collision point back to world space
    collision_world = R @ collision_local + np.array([cx, cy, cz])
    return collision_world

# Example Usage
p1 = (1, 1, 1)
p2 = (6, 6, 6)
cuboid_center = (4, 4, 4)
cuboid_size = (3, 2, 5)  # width, height, depth
euler_angles = (np.radians(30), np.radians(45), np.radians(60))  # Roll, Pitch, Yaw in radians


import time
start = time.perf_counter()
collision_point = first_collision_with_rotated_cuboid_euler(p1, p2, cuboid_center, cuboid_size, euler_angles)
end = time.perf_counter()
print(1 / (end-start))
print("Collision Point:", collision_point)
