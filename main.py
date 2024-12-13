### Imports ###
import glfw
import OpenGL.GL as gl
import numpy as np
import ctypes
import OpenGL.GL.shaders as gls
import pyrr

import threading
import atexit
import math
import sys
import time
import random
from datetime import datetime
from PIL import Image
from functools import wraps
import copy

from src import COLOURS

### Constants ###
FLIGHT_ENABLED = True
DEBUG = True
DEBUG_TRANSPARENCY = 0.2
DEBUG_RECT_MODEL = "debug_rect.obj"
PRINT_FRAME_RATE = False

TPS = 60
FPS_CAP = 1 if not DEBUG else 0 # Set to 0 for uncapped FPS, 1 for VSYNC, 2+ for CAP
FOV = 93

SPT = 1 / TPS

GL_ERROR_CHECK_DELAY_SEC = 5

MAX_LIGHTS = 100

MOUSE_SENSITIVITY = 0.12
MAX_LOOK_THETA = 89.95 # Must be < 90 degrees

PLAYER_ACCELERATION = 1.56 / TPS
PLAYER_ACCELERATION_SPRINT_MULTIPLIER = 1.88
JUMP_STRENGTH = 0.3
GRAVITY = 9.81 / 8 / TPS
HORIZONTAL_DRAG = 0.8
VERTICAL_DRAG = 0.998
DEFAULT_PLAYER_HEIGHT = 3
OBJECT_DRAG = 0.7

SHADERS_PATH = "shaders/"
MODELS_PATH = "models/"
GFX_PATH = "gfx/"
SCREENSHOTS_PATH = "screenshots/"

CUBOID_MODEL = "cube.obj"
SPHERE_MODEL = "sphere.obj"

GLOBAL_UP = np.array([0, 1, 0], dtype=np.float32)

SKYBOX_COLOR = (12/255, 13/255, 18/255)


### Thread Handling ###
events = {"exit": threading.Event()}
locks = {}


### Exit Handling ###
def exit_handler() -> None:
    """
    Runs before main threads terminates.
    """
    events["exit"].set()
    glfw.terminate()

atexit.register(exit_handler)


### DEBUGGING ###
class FrameRateMonitor:
    def __init__(self, name=""):
        if not PRINT_FRAME_RATE:
            return
        
        self.frame_times = []
        self.last_update_time = None
        self.name = name

        self.total_elapsed = 0

    def print_fps(self):
        if not PRINT_FRAME_RATE:
            return
        
        if len(self.frame_times) and self.total_elapsed:
            fps = len(self.frame_times) / self.total_elapsed

            print(f"[{self.name}] FPS: {round(fps, 3)} LOW: {round(len(self.frame_times) / (max(*self.frame_times, 0.001) * len(self.frame_times)), 3)} HIGH: {round(len(self.frame_times) / (max(min(self.frame_times), 0.001) * len(self.frame_times)), 3)}")

        self.frame_times = []
        self.total_elapsed = 0

    def run(self):
        if not PRINT_FRAME_RATE:
            return

        current_time = time.time()

        if self.last_update_time == None:
            self.last_update_time = current_time
            return

        elapsed = current_time - self.last_update_time
        self.last_update_time = current_time
        self.total_elapsed += elapsed
        self.frame_times.append(elapsed)

        if self.total_elapsed > 1:
            self.print_fps()


### Functions ###
def safe_file_readlines(path):
    try:
        with open(path, 'r') as f:
            return f.readlines()
    except FileNotFoundError:
        raise Exception(f"Error: Vertex shader file not found at '{path}'.")
    except IOError as e:
        raise Exception(f"Error: Unable to read vertex shader file '{path}': {e}")

def async_lock(func):
    """Decorator to automatically acquire and release a lock."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with self.lock:
            return copy.deepcopy(func(self, *args, **kwargs))
    return wrapper

def euler_to_rotation_matrix(roll, pitch, yaw): 
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(yaw), -np.sin(yaw)],
        [0, np.sin(yaw), np.cos(yaw)]
    ])
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    R_z = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1]
    ])
    
    return R_z @ R_y @ R_x

def line_circle_intersection(p1, p2, circle_center, radius):
    print(p1, p2, circle_center, radius)
    # Unpack points
    x1, y1 = p1
    x2, y2 = p2
    cx, cy = circle_center
    
    # Vector for the line
    dx = x2 - x1
    dy = y2 - y1
    
    # Circle equation: (x - cx)^2 + (y - cy)^2 = r^2
    # Parametric line equations: x(t) = x1 + t * dx, y(t) = y1 + t * dy
    # Substituting these into the circle's equation gives a quadratic in t

    # Coefficients for the quadratic equation At^2 + Bt + C = 0
    A = dx**2 + dy**2
    B = 2 * (dx * (x1 - cx) + dy * (y1 - cy))
    C = (x1 - cx)**2 + (y1 - cy)**2 - radius**2
    
    # Solve the quadratic equation
    discriminant = B**2 - 4*A*C
    
    # No intersection (discriminant < 0)
    if discriminant < 0 or A == 0:
        return None

    # Two intersections, we want the first one (smallest t)
    t1 = (-B - np.sqrt(discriminant)) / (2 * A)
    t2 = (-B + np.sqrt(discriminant)) / (2 * A)
    
    # Return the first intersection point (smallest t > 0)
    t = min(t1, t2) if min(t1, t2) >= 0 else max(t1, t2)

    print(t)

    if t < -1 or t > 1 or t is np.nan:
        return None
    
    # Calculate the intersection point
    intersection_x = x1 + t * dx
    intersection_y = y1 + t * dy
    
    return np.array([intersection_x, intersection_y], dtype=np.float32)

def line_sphere_intersection(p1, p2, sphere_center, radius):
    print(p1, p2, sphere_center, radius)
    # Convert points to numpy arrays
    p1 = np.array(p1, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)
    sphere_center = np.array(sphere_center, dtype=np.float32)
    
    # Vector from p1 to p2 (direction of the line)
    line_dir = p2 - p1
    # Vector from p1 to sphere center
    p1_to_center = p1 - sphere_center
    
    # Coefficients of the quadratic equation A * t^2 + B * t + C = 0
    A = np.dot(line_dir, line_dir)
    B = 2 * np.dot(p1_to_center, line_dir)
    C = np.dot(p1_to_center, p1_to_center) - radius**2
    
    # Discriminant of the quadratic equation
    discriminant = B**2 - 4 * A * C

    if discriminant < 0 or A == 0:
        # No intersection
        return None
    
    # Solve for t (the parameter of the intersection point)
    sqrt_disc = np.sqrt(discriminant)
    t1 = (-B - sqrt_disc) / (2 * A)
    t2 = (-B + sqrt_disc) / (2 * A)

    t_list = [t for t in [t1, t2] if t >= 0]

    if len(t_list) == 0:
        return None
    
    # We want the smallest positive t (the first intersection)
    t = min(t for t in [t1, t2] if t >= 0)

    print(f"T: {t}")
    
    if t < -1 or t > 1 or t is np.nan:
        # The line does not intersect the sphere in the positive direction
        return None
    
    # Compute the intersection point
    intersection_point = p1 + t * line_dir
    return intersection_point

"""
def line_cuboid_intersection(ray_start, ray_finish, cuboid_center, cuboid_size):
    width, height, depth = cuboid_size

    # Convert input points to NumPy arrays for efficient computation
    box_min = np.array([cuboid_center[0] - width / 2, cuboid_center[1] - height / 2, cuboid_center[2] - depth / 2])
    box_max = np.array([cuboid_center[0] + width / 2, cuboid_center[1] + height / 2, cuboid_center[2] + depth / 2])
    ray_start = np.array(ray_start)
    ray_finish = np.array(ray_finish)
    
    # Ray direction (normalized)
    ray_dir = ray_finish - ray_start
    if ray_dir.all() > 0:
        ray_inv_dir = 1.0 / ray_dir
    else:
        ray_inv_dir = 9999999

    # Initialize tmin and tmax for ray-box intersection in each dimension
    tmin = (box_min - ray_start) * ray_inv_dir
    tmax = (box_max - ray_start) * ray_inv_dir
    
    # For each axis (x, y, z), calculate the min and max t values
    tmin = np.minimum(tmin, tmax)
    tmax = np.maximum(tmin, tmax)
    
    # Find the overall tmin and tmax
    tmin_final = np.max(tmin)
    tmax_final = np.min(tmax)
    
    # Check if there is an intersection
    if tmax_final >= tmin_final:
        # Calculate the intersection point
        intersection_point = ray_start + tmin_final * ray_dir
        return intersection_point
    else:
        return None

def line_cuboid_collision(p1, p2, cuboid_center, cuboid_size, euler_angles):
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

    collision = line_cuboid_intersection(p1_local, p2_local, (0, 0, 0), cuboid_size)

    return collision
"""

def line_rounded_cuboid_collision(p1, p2, radius, cuboid_center, cuboid_size, euler_angles, col):
    extended_cuboid_size = cuboid_size + radius * 2

    p1 = np.array(p1)
    p2 = np.array(p2)

    # Unpack cuboid properties
    cx, cy, cz = cuboid_center
    w, h, d = extended_cuboid_size
    roll, pitch, yaw = euler_angles

    # Convert Euler angles to rotation matrix
    rotation_matrix = euler_to_rotation_matrix(roll, pitch, yaw)
    R_inv = np.transpose(rotation_matrix)  # Inverse of rotation matrix is its transpose
    
    # Transform line segment to cuboid's local space
    p1_local = R_inv @ (p1 - np.array([cx, cy, cz]))
    p2_local = R_inv @ (p2 - np.array([cx, cy, cz]))

    pt = copy.deepcopy(np.array([cx, cy, cz]) + np.array([3, 5, 0]))

    pt_local = R_inv @ (pt - np.array([cx, cy, cz]))

    pt_local[1] += 10

    print("\n\n###")
    print(p1_local, p2_local)
    print(p1, p2, radius, cuboid_center, cuboid_size, euler_angles)

    # Axis-aligned bounds in local space
    bounds = [
        (-w / 2, w / 2),  # x-bounds
        (-h / 2, h / 2),  # y-bounds
        (-d / 2, d / 2)   # z-bounds
    ]

    col["1"].set_corners(np.array([-w / 2, -h / 2 + 10, -d / 2]), np.array([w / 2, h / 2 + 10, d / 2]))

    p1ah = copy.deepcopy(p1_local)
    p1ah[1] += 10

    p2ah = copy.deepcopy(p2_local)
    p2ah[1] += 10

    col["2"].set_corners(pt_local - 0.2, pt_local + 0.2)
    #col["2"].set_corners(p1ah - 0.2, p2ah + 0.2)
    
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

    """
    collision_local = line_cuboid_collision(p1, p2, cuboid_center, extended_cuboid_size, euler_angles)
    if collision_local is None:
        return collision_local
    """

    abs_pos = np.abs(collision_local)

    axes = [i for i, x in enumerate(abs_pos) if x > cuboid_size[i] / 2]

    if len(axes) == 3: # Corner
        print("CORNER")
        sphere_center = np.array([np.sign(collision_local[0]) * cuboid_size[0] / 2, np.sign(collision_local[1]) * cuboid_size[1] / 2, np.sign(collision_local[2]) * cuboid_size[2] / 2])
        collision_local = line_sphere_intersection(p1_local, p2_local, sphere_center, radius)
        if collision_local is None:
            return None
    elif len(axes) == 2: # Edge
        print("EDGE")
        x = collision_local[axes[0]]
        y = collision_local[axes[1]]

        possible_axes = [0, 1, 2]
        removed_axes = [axis for axis in possible_axes if axis not in axes]

        p1_2D = np.delete(p1_local, removed_axes)
        p2_2D = np.delete(p2_local, removed_axes)
        circle_center = np.array([np.sign(x) * cuboid_size[axes[0]] / 2, np.sign(y) * cuboid_size[axes[1]] / 2])

        collision_local_2D = line_circle_intersection(p1_2D, p2_2D, circle_center, radius)

        if collision_local_2D is None:
            return None
        
        collision_local[axes[0]] = collision_local_2D[0]
        collision_local[axes[1]] = collision_local_2D[1]
    # else: just don't worry about it
    
    # Transform collision point back to world space
    collision_world = rotation_matrix @ collision_local + np.array([cx, cy, cz])
    return collision_world


### Classes ####
class Mesh:
    def __init__(self, path):        
        # x, y, z, s, t, nx, ny, nz
        self.vertices = self.load_mesh(path)
        
        # Each vertex consists of 3 components (x, y, z)
        self.vertex_count = len(self.vertices) // 3
        self.vertices = np.array(self.vertices, dtype=np.float32)
        
        # Vertex Array Object (vao) stores buffer attributes (defines how vertex data is laid out in memory, etc)
        self.vao = gl.glGenVertexArrays(1)
        # Activate Vertex Array Object
        gl.glBindVertexArray(self.vao)
        
        # Vertex Buffer Object (vbo) stores raw data (vertex positions, normals, colors, etc)
        self.vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, gl.GL_STATIC_DRAW) # Upload the vertex data to the GPU

        # Add attribute pointer for position location in buffer so gpu can find vertex data in memory
        # Location 1 - Postion
        gl.glEnableVertexAttribArray(0)
        # Location, number of floats, format (float), gl.GL_FALSE, stride (total length of vertex, 4 bytes times number of floats), ctypes of starting position in bytes (void pointer expected)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 32, ctypes.c_void_p(0))
        
        # Location 2 - ST
        gl.glEnableVertexAttribArray(1)
        # Location, number of floats, format (float), gl.GL_FALSE, stride (total length of vertex, 4 bytes times number of floats), ctypes of starting position in bytes (void pointer expected)
        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, 32, ctypes.c_void_p(12))
        
        # Location 3 - Normal
        gl.glEnableVertexAttribArray(2)
        # Location, number of floats, format (float), gl.GL_FALSE, stride (total length of vertex, 4 bytes times number of floats), ctypes of starting position in bytes (void pointer expected)
        gl.glVertexAttribPointer(2, 3, gl.GL_FLOAT, gl.GL_FALSE, 32, ctypes.c_void_p(20))
    
    @staticmethod
    def load_mesh(filepath):
    
        vertices = []
        flags = {"v": [], "vt": [], "vn": []}
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
            for line in lines:
                line.replace("\n", "")
                
                first_space = line.find(" ")
                flag = line[0:first_space]
                
                if flag in flags.keys():
                    line = line.replace(flag + " ", "")
                    line = line.split(" ")
                    flags[flag].append([float(x) for x in line])
                elif flag == "f":
                    line = line.replace(flag + " ", "")
                    line = line.split(" ")
                    
                    face_vertices = []
                    face_textures = []
                    face_normals = []
                    
                    for vertex in line:
                        l = vertex.split("/")
                        face_vertices.append(flags["v"][int(l[0]) - 1])
                        face_textures.append(flags["vt"][int(l[1]) - 1])
                        face_normals.append(flags["vn"][int(l[2]) - 1])

                    triangles_in_face = len(line) - 2
                    vertex_order = []

                    for x in range(triangles_in_face):
                        vertex_order.extend((0, x + 1, x + 2))
                    for x in vertex_order:
                        vertices.extend((*face_vertices[x], *face_textures[x], *face_normals[x]))
        
        return vertices
    
    def destroy(self):
        # Remove allocated memory
        gl.glDeleteVertexArrays(1, (self.vao, ))
        gl.glDeleteBuffers(1, (self.vbo, ))


class Material:
    def __init__(self, filepath):
        # Allocate space where texture will be stored
        self.texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        
        # S is horizontal of a texture, T is the vertical of a texture, GL_REPEAT means image will loop if S or T over/under 1. MIN_FILTER is downsizing. MAG_FILTER is enlarging.
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        
        # Load image, then get height, and the images data
        image = Image.open(filepath).convert("RGBA")
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        image_width, image_height = image.size
        image_data = image.tobytes("raw", "RGBA")
        
        # Get data for image, then generate the mipmap
        # Texture location, mipmap level, format image is stored as, width, height, border color, input image format, data format, image data
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, image_width, image_height, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, image_data)
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
        
    def use(self):
        # Select active texture 0, then bind texture
        gl.glActiveTexture(gl.GL_TEXTURE0) # OPTIMIZE LATER MULTIPLE ACTIVE
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        
    def destroy(self):
        # Remove allocated memory
        gl.glDeleteTextures(1, (self.texture, ))


class LockedRectCollider: # Create parent class for future colliders
    def __init__(self, corner1, corner2, debug = False, debug_color = COLOURS.RED1):
        # Init vars
        self.corner1 = corner1
        self.corner2 = corner2
        self.debug = debug and DEBUG
        self.debug_color = np.array([*debug_color, DEBUG_TRANSPARENCY], dtype=np.float32)

        # Calculate rect
        self.top = max(corner1[1], corner2[1])
        self.bottom = min(corner1[1], corner2[1])

        self.right = max(corner1[0], corner2[0])
        self.left = min(corner1[0], corner2[0])

        self.back = max(corner1[2], corner2[2])
        self.front = min(corner1[2], corner2[2])

        self.pos = [(self.right + self.left) / 2, (self.top + self.bottom) / 2, (self.back + self.front) / 2]

        self.height = abs(self.top - self.bottom) / 2
        self.width = abs(self.right - self.left) / 2
        self.depth = abs(self.back - self.front) / 2

        self.scale = [self.width, self.height, self.depth]

        #for x in range(len(self.pos)):
        #    self.pos[x] -= self.scale[x] / 2

        # Debug
        if self.debug:
            self.mesh = Mesh(MODELS_PATH + DEBUG_RECT_MODEL)

    def set_corners(self, corner1, corner2):
        self.top = max(corner1[1], corner2[1])
        self.bottom = min(corner1[1], corner2[1])

        self.right = max(corner1[0], corner2[0])
        self.left = min(corner1[0], corner2[0])

        self.back = max(corner1[2], corner2[2])
        self.front = min(corner1[2], corner2[2])

        self.pos = [(self.right + self.left) / 2, (self.top + self.bottom) / 2, (self.back + self.front) / 2]

        self.height = abs(self.top - self.bottom) / 2
        self.width = abs(self.right - self.left) / 2
        self.depth = abs(self.back - self.front) / 2

        self.scale = [self.width, self.height, self.depth]

    def render(self, model_matrix_handle, color_handle):
        if self.debug:
            gl.glUniform4fv(color_handle, 1, self.debug_color)
            
            model_transform = pyrr.matrix44.create_identity(dtype = np.float32)

            # Scale
            model_transform = pyrr.matrix44.multiply(
                m1 = model_transform,
                m2 = pyrr.matrix44.create_from_scale(self.scale, dtype = np.float32)
            )

            # Translate
            model_transform = pyrr.matrix44.multiply(
                m1 = model_transform,
                m2 = pyrr.matrix44.create_from_translation(self.pos, dtype = np.float32)
            )
            
            # Complete transform
            gl.glUniformMatrix4fv(model_matrix_handle, 1, gl.GL_FALSE, model_transform)
            gl.glBindVertexArray(self.mesh.vao)
            
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.mesh.vertex_count)


class Cuboid:
    def __init__(self, material_path, eulers = np.zeros(3, dtype=np.float32), angular_velocity = np.zeros(3, dtype=np.float32), mass = 1, pos = np.zeros(3, dtype=np.float32), velocity = np.zeros(3, dtype=np.float32), size = np.ones(3, dtype=np.float32)):
        self.eulers = np.array(eulers, dtype=np.float32)
        self.angular_velocity = np.array(angular_velocity, dtype=np.float32)
        self.mass = mass
        self.pos = np.array(pos, dtype=np.float32)
        self.velocity = np.array(velocity, dtype=np.float32)
        self.size = np.array(size, dtype=np.float32)
        self.scale = self.size / 2 # Mesh is 2x2x2

        self.volume = np.prod(self.size)
        self.density = self.mass / self.volume

        self.inertia_0 = self.calculate_inertia()

        self.mesh = Mesh(MODELS_PATH + CUBOID_MODEL)
        self.material = Material(material_path)

        self.lock = threading.Lock()

    def calculate_inertia(self):
        return self.density * np.array([self.size[1]**2 + self.size[2]**2, self.size[0]**2 + self.size[2]**2, self.size[0]**2 + self.size[1]**2]) / 12
    
    @async_lock
    def get_pos(self):
        return self.pos
    
    @async_lock
    def get_size(self):
        return self.size
    
    @async_lock
    def get_velocity(self):
        return self.velocity
    
    @async_lock
    def get_eulers(self):
        return self.eulers

    @async_lock
    def get_angular_velocity(self):
        return self.angular_velocity
    
    @async_lock
    def add_angular_velocity(self, d_angular_velocity):
        self.eulers += np.array(d_angular_velocity)

    @async_lock
    def add_pos(self, d_pos):
        self.pos += np.array(d_pos)

    def get_relative_eulers(self):
        return np.array([self.eulers[0], self.eulers[2], self.eulers[1]], dtype=np.float32)

    def simulate(self, _rigid_bodies_list):
        self.add_pos(self.get_velocity())
        self.add_angular_velocity(self.get_angular_velocity())

    def render(self, model_matrix_handle):
        pos = self.get_pos()

        self.material.use()
        
        model_transform = pyrr.matrix44.create_identity(dtype = np.float32)

        # Scale
        model_transform = pyrr.matrix44.multiply(
            m1 = model_transform,
            m2 = pyrr.matrix44.create_from_scale(self.scale, dtype = np.float32)
        )

        # Rotate around origin
        model_transform = pyrr.matrix44.multiply(
            m1 = model_transform,
            m2 = pyrr.matrix44.create_from_eulers(self.get_relative_eulers(), dtype = np.float32)
        )

        # Translate
        model_transform = pyrr.matrix44.multiply(
            m1 = model_transform,
            m2 = pyrr.matrix44.create_from_translation(pos, dtype = np.float32)
        )
        
        # Complete transform
        gl.glUniformMatrix4fv(model_matrix_handle, 1, gl.GL_FALSE, model_transform)
        gl.glBindVertexArray(self.mesh.vao)
        
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.mesh.vertex_count)

    def destroy(self):
        self.mesh.destroy()
        self.material.destroy()


class Sphere:
    def __init__(self, material_path, eulers = np.zeros(3, dtype=np.float32), angular_velocity = np.zeros(3, dtype=np.float32), mass = 1, pos = np.zeros(3, dtype=np.float32), velocity = np.zeros(3, dtype=np.float32), radius = 0.5, col = None):
        self.rotation = np.array(eulers, dtype=np.float32)
        self.angular_velocity = np.array(angular_velocity, dtype=np.float32)
        self.mass = mass
        self.pos = np.array(pos, dtype=np.float32)
        self.velocity = np.array(velocity, dtype=np.float32)
        self.radius = radius
        self.col = col

        self.acceleration = np.zeros(3)
        self.scale = np.array([radius * 2, radius * 2, radius * 2]) / 2 # Mesh is 2x2x2
        self.volume = 4 * np.pi * self.radius**2
        self.density = self.mass / self.volume

        self.inertia_0 = self.calculate_inertia()

        self.mesh = Mesh(MODELS_PATH + SPHERE_MODEL)
        self.material = Material(material_path)

        self.lock = threading.Lock()

    def calculate_inertia(self):
        return (2/5) * self.mass * self.radius**2
    
    # Get
    @async_lock
    def get_pos(self):
        return self.pos
    
    @async_lock
    def get_velocity(self):
        return self.velocity
    
    @async_lock
    def get_acceleration(self):
        return self.acceleration
    
    @async_lock
    def get_angular_velocity(self):
        return self.angular_velocity

    # Set
    @async_lock
    def set_pos(self, new_pos):
        self.pos = np.array(new_pos, dtype=np.float32)

    @async_lock
    def set_velocity(self, new_velocity):
        self.velocity = np.array(new_velocity, dtype=np.float32)

    @async_lock
    def set_acceleration(self, new_acceleration):
        self.acceleration = np.array(new_acceleration, dtype=np.float32)

    # Add
    @async_lock
    def add_pos(self, d_pos):
        self.pos += np.array(d_pos)

    @async_lock
    def add_velocity(self, d_velocity):
        self.velocity += np.array(d_velocity)

    @async_lock
    def add_acceleration(self, d_acceleration):
        self.acceleration += np.array(d_acceleration)
    
    @async_lock
    def add_rotation(self, d_rotation):
        self.rotation += np.array(d_rotation)

    def gravity(self):
        self.add_acceleration([0, -GRAVITY*0.01, 0])

    @async_lock
    def drag(self):
        self.acceleration *= OBJECT_DRAG

    def simulate(self, rigid_bodies_list):
        start_pos = self.get_pos()

        self.gravity()
        self.drag()
        
        self.add_velocity(self.get_acceleration())
        self.add_pos(self.get_velocity())
        self.add_rotation(self.get_angular_velocity())

        end_pos = self.get_pos()

        for rigid_body in rigid_bodies_list:
            if type(rigid_body) != Cuboid or rigid_body == self: #??
                continue
            
            with self.lock:
                pos_at_collision = line_rounded_cuboid_collision(start_pos, end_pos, self.radius, rigid_body.get_pos(), rigid_body.get_size(), rigid_body.get_eulers(), self.col)
            
            if str(pos_at_collision) != "None":
                print(pos_at_collision)
                with self.lock:
                    self.col["collision"].set_corners(pos_at_collision-0.1, pos_at_collision+0.1)
                self.add_pos([0, self.radius + 4.5, 0])
                self.set_acceleration([0, 0, 0])
                self.set_velocity([0, 0, 0])
            
            break

    def render(self, model_matrix_handle):
        pos = self.get_pos()

        self.material.use()
        
        model_transform = pyrr.matrix44.create_identity(dtype = np.float32)

        # Scale
        model_transform = pyrr.matrix44.multiply(
            m1 = model_transform,
            m2 = pyrr.matrix44.create_from_scale(self.scale, dtype = np.float32)
        )

        # Rotate around origin
        model_transform = pyrr.matrix44.multiply(
            m1 = model_transform,
            m2 = pyrr.matrix44.create_from_eulers(self.rotation, dtype = np.float32)
        )

        # Translate
        model_transform = pyrr.matrix44.multiply(
            m1 = model_transform,
            m2 = pyrr.matrix44.create_from_translation(pos, dtype = np.float32)
        )
        
        # Complete transform
        gl.glUniformMatrix4fv(model_matrix_handle, 1, gl.GL_FALSE, model_transform)
        gl.glBindVertexArray(self.mesh.vao)
        
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.mesh.vertex_count)

    def destroy(self):
        self.mesh.destroy()
        self.material.destroy()


class Object:
    def __init__(self, mesh_path, material_path, pos = np.zeros(3), rotation = np.zeros(3), scale = np.ones(3)):
        self.pos = np.array(pos, dtype=np.float32)
        self.rotation = np.array(rotation, dtype=np.float32)
        self.scale = np.array(scale, dtype=np.float32)

        self.mesh = Mesh(mesh_path)
        self.material = Material(material_path)
        #self.material.use()

    def render(self, model_matrix_handle):
        self.material.use()
        
        model_transform = pyrr.matrix44.create_identity(dtype = np.float32)

        # Scale
        model_transform = pyrr.matrix44.multiply(
            m1 = model_transform,
            m2 = pyrr.matrix44.create_from_scale(self.scale, dtype = np.float32)
        )

        # Rotate around origin
        model_transform = pyrr.matrix44.multiply(
            m1 = model_transform,
            m2 = pyrr.matrix44.create_from_eulers(self.rotation, dtype = np.float32)
        )

        # Translate
        model_transform = pyrr.matrix44.multiply(
            m1 = model_transform,
            m2 = pyrr.matrix44.create_from_translation(self.pos, dtype = np.float32)
        )
        
        # Complete transform
        gl.glUniformMatrix4fv(model_matrix_handle, 1, gl.GL_FALSE, model_transform)
        gl.glBindVertexArray(self.mesh.vao)
        
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.mesh.vertex_count)
    
    def destroy(self):
        self.mesh.destroy()
        self.material.destroy()


class Light:
    def __init__(self, position, color, strength):
        self.position = np.array(position, dtype=np.float32)
        self.color = np.array(color, dtype=np.float32)
        self.strength = strength


class GraphicsEngine:
    def __init__(self, aspect):
        # Initilize OpenGL
        gl.glClearColor(*SKYBOX_COLOR, 1)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        # Initilize shader
        self.shader = self.create_shader(SHADERS_PATH + "default.vert", SHADERS_PATH + "default.frag")
        gl.glUseProgram(self.shader)

        # Initilize texture
        texture_handle = gl.glGetUniformLocation(self.shader, "imageTexture")
        gl.glUniform1i(texture_handle, 0) # NEED TO CHANGE FOR EACH OBJECT??!?!?!?!??! (MAYBE)

        # Initilize projection
        projection_handle =        gl.glGetUniformLocation(self.shader, "projection")

        projection_transform = pyrr.matrix44.create_perspective_projection(fovy = FOV, aspect = aspect, near = 0.1, far = 200, dtype = np.float32)
        gl.glUniformMatrix4fv(projection_handle, 1, gl.GL_FALSE, projection_transform)

        # Vertex shader
        self.model_matrix_handle = gl.glGetUniformLocation(self.shader, "model")
        self.view_matrix_handle =  gl.glGetUniformLocation(self.shader, "view")

        # Fragment shader
        self.total_lights_handle = gl.glGetUniformLocation(self.shader, "totalLights") 
        self.camera_pos_handle =   gl.glGetUniformLocation(self.shader, "cameraPosition")
        self.light_handle = {
            "position": [
                gl.glGetUniformLocation(self.shader, f"lights[{i}].position")
                for i in range(MAX_LIGHTS)
                ],
            "color": [
                gl.glGetUniformLocation(self.shader, f"lights[{i}].color")
                for i in range(MAX_LIGHTS)
                ],
            "strength": [
                gl.glGetUniformLocation(self.shader, f"lights[{i}].strength")
                for i in range(MAX_LIGHTS)
                ]
        }

        if DEBUG:
            self.debug_shader = self.create_shader(SHADERS_PATH + "debug.vert", SHADERS_PATH + "debug.frag")
            gl.glUseProgram(self.debug_shader)

            debug_projection_handle =        gl.glGetUniformLocation(self.debug_shader, "projection")

            debug_projection_transform = pyrr.matrix44.create_perspective_projection(fovy = FOV, aspect = aspect, near = 0.1, far = 200, dtype = np.float32)
            gl.glUniformMatrix4fv(debug_projection_handle, 1, gl.GL_FALSE, debug_projection_transform)

            self.debug_model_matrix_handle = gl.glGetUniformLocation(self.debug_shader, "model")
            self.debug_view_matrix_handle =  gl.glGetUniformLocation(self.debug_shader, "view")
            self.debug_color_handle =        gl.glGetUniformLocation(self.debug_shader, "objectColor")

    def create_shader(self, vertex_path, fragment_path):
        vertex_src = safe_file_readlines(vertex_path)
        fragment_src = safe_file_readlines(fragment_path)
        
        shader = gls.compileProgram(
            gls.compileShader(vertex_src, gl.GL_VERTEX_SHADER),
            gls.compileShader(fragment_src, gl.GL_FRAGMENT_SHADER)
        )
        
        return shader
    
    def render_graphics(self, scene):
        # Refresh screen
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glUseProgram(self.shader)
        
        player_pos = scene.get_player_pos()
        player_forwards = scene.get_player_forwards()
        player_up = scene.get_player_up()

        view_transform = pyrr.matrix44.create_look_at(
            eye = np.array(player_pos, dtype = np.float32),
            target = np.array(player_pos + player_forwards, dtype = np.float32),
            up = np.array(player_up, dtype = np.float32),
            dtype = np.float32
        )
        
        gl.glUniformMatrix4fv(self.view_matrix_handle, 1, gl.GL_FALSE, view_transform)
        
        for object in scene.get_objects_list():
            object.render(self.model_matrix_handle)

        for rigid_body in scene.get_rigid_bodies_list():
            rigid_body.render(self.model_matrix_handle)

        static_lights = scene.get_static_lights()[:100]
        for i, light in enumerate(static_lights):
            gl.glUniform3fv(self.light_handle["position"][i], 1, light.position)
            gl.glUniform3fv(self.light_handle["color"][i], 1, light.color)
            gl.glUniform1f(self.light_handle["strength"][i], light.strength)
        
        avalable_lights = MAX_LIGHTS - len(static_lights)
        dynamic_lights = scene.get_dynamic_lights_list()[0:avalable_lights]

        for i, light in enumerate(dynamic_lights):
            gl.glUniform3fv(self.light_handle["position"][i], 1, light.position)
            gl.glUniform3fv(self.light_handle["color"][i], 1, light.color)
            gl.glUniform1f(self.light_handle["strength"][i], light.strength)
        
        total_lights = len(static_lights) + len(dynamic_lights)

        gl.glUniform1i(self.total_lights_handle, total_lights)
        gl.glUniform3fv(self.camera_pos_handle, 1, player_pos)

        if DEBUG:
            gl.glUseProgram(self.debug_shader)

            gl.glUniformMatrix4fv(self.debug_view_matrix_handle, 1, gl.GL_FALSE, view_transform)

            for collider in scene.get_colliders_list():
                collider.render(self.debug_model_matrix_handle, self.debug_color_handle)

    def destroy(self):
        gl.glDeleteProgram(self.shader)


class Window:
    def __init__(self):
        # Set variables
        self.graphics_engine = None
        self.scene = None

        # Initilize variables
        self.running = True
        self.pos_offset = np.array([0, 0], dtype=np.float32)

        self.gl_error_check_time = time.perf_counter()
        self.fps_monitor = FrameRateMonitor("WINDOW")

        # Initilize GLFW
        if not glfw.init():
            raise Exception("GLFW can't be initialized")
        
        self.monitor = glfw.get_primary_monitor()
        if not self.monitor:
            raise Exception("GLFW can't find primary monitor")

        self.video_mode = glfw.get_video_mode(self.monitor)
        if not self.video_mode:
            raise Exception("GLFW can't get video mode")
        
        self.screen_width = self.video_mode.size.width
        self.screen_height = self.video_mode.size.height

        self.aspect = self.screen_width / self.screen_height
        
        self.window = glfw.create_window(self.screen_width, self.screen_height, "IntoHavoc", self.monitor, None)
        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window can't be created")

        glfw.make_context_current(self.window)

        # Max FPS (Disable VSYNC)
        glfw.swap_interval(FPS_CAP)

        glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_HIDDEN)
        if glfw.raw_mouse_motion_supported():
            glfw.set_input_mode(self.window, glfw.RAW_MOUSE_MOTION, glfw.TRUE)

        glfw.set_cursor_pos(self.window, self.screen_width // 2, self.screen_height // 2)

    def init(self, graphics_engine, scene):
        self.graphics_engine = graphics_engine
        self.scene = scene
        
        glfw.set_cursor_pos_callback(self.window, self.mouse_move_func)
    
    def mouse_move_func(self, _window = None, _delta_x = None, _delta_y = None):
        if self.scene.get_should_center_cursor():
            glfw.set_cursor_pos(self.window, self.screen_width // 2, self.screen_height // 2)
            self.scene.set_should_center_cursor(False)

    def handle_window_events(self) -> None:
        """
        Handle GLFW events and closing the window.
        """
        # Check if window should close
        if glfw.window_should_close(self.window) or glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
            self.close()
            return
        
    def screenshot_check(self):
        if self.scene.get_do_screenshot():
            # Get image
            buffer = gl.glReadPixels(0, 0, self.screen_width, self.screen_height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)

            image_data = np.frombuffer(buffer, dtype=np.uint8).reshape((self.screen_height, self.screen_width, 3))
            image_data = np.flipud(image_data)

            image = Image.fromarray(image_data, "RGB")

            # Get filename
            now = datetime.now()
            filename_timestamp = now.strftime("%Y-%m-%d_%H.%M.%S") + f".{now.microsecond // 1000:03d}"
            
            filepath = SCREENSHOTS_PATH + filename_timestamp + ".png"

            # Save
            image.save(filepath)
            print(f"Screenshot saved to {filepath}")

            self.scene.set_do_screenshot(False)

    def render(self, graphics_engine: GraphicsEngine, scene):
        graphics_engine.render_graphics(scene)
        glfw.swap_buffers(self.window) #<---

    def tick(self) -> None:
        """
        Tick (manage frame rate).
        """
        glfw.poll_events()

        self.fps_monitor.run()

    def close(self) -> None:
        """
        Close the GLFW window and terminate.
        """
        self.running = False
        self.scene.quit()
        self.graphics_engine.destroy()
        glfw.terminate()

    def check_gl_error(self):
        if time.perf_counter() > self.gl_error_check_time + GL_ERROR_CHECK_DELAY_SEC:            
            error = gl.glGetError()
            if error != gl.GL_NO_ERROR:
                print(f"OpenGL error: {error}")

            self.gl_error_check_time = time.perf_counter()

    def main(self) -> None:
        """
        Main window loop.
        """
        while self.running:
            self.render(self.graphics_engine, self.scene)
            self.tick()
            self.mouse_move_func()
            self.check_gl_error()
            self.handle_window_events()
            self.screenshot_check()


class Scene():
    def __init__(self, events, window, screen_size):
        self.events = events
        self.window_handler = window
        self.window = window.window
        self.screen_width, self.screen_height = screen_size

        self.lock = threading.Lock()
        self.player_lock = threading.Lock()

        self.running = True
        self.set_player_feet([0, 0, 0])
        self.player_acceleration = np.array([0, 0, 0], dtype=np.float32)
        self.player_forward_vector = np.array([0, 0], dtype=np.float32)
        self.update_player_forwards()

        self.mouse_pos = np.array([self.screen_width / 2, self.screen_height / 2], dtype=np.int16)
        self.should_center_cursor = True

        self.do_screenshot = False

        self.previous_f12_state = False
        self.previous_space_state = False

        # Initilize Objs
        self.objects = {
            'mountain': Object(MODELS_PATH + "mountains.obj", GFX_PATH + "wood.jpeg", [0, -58, 0], [np.pi / 2, np.pi, 0]),
            'ship':     Object(MODELS_PATH + "ship.obj", GFX_PATH + "rendering_texture.jpg", [0, -40, 0], scale = [0.6, 0.6, 0.6]),
            'cube':     Object(MODELS_PATH + "cube.obj", GFX_PATH + "rendering_texture.jpg", [0, -50, 0], scale = [1, 1, 1]),
            'cube2':    Object(MODELS_PATH + "cube.obj", GFX_PATH + "rendering_texture.jpg", [0, -50, 0], scale = [2, 2, 2]),
            'cube3':    Object(MODELS_PATH + "cube.obj", GFX_PATH + "rendering_texture.jpg", [20, -30, 20], scale = [1, 1, 1]),
            'test':     Object(MODELS_PATH + "Pipes.obj", GFX_PATH + "PipesBake.png", [0, -35, 0]),
            'cans':     Object(MODELS_PATH + "cans2.obj", GFX_PATH + "BakeImage.png", [0, -55, 0]),
            'scene':    Object(MODELS_PATH + "StartScenePrev3.obj", GFX_PATH + "BakeTextTT2 copy.png", [0, 0, 0])
        }

        positions = [
            [13.196548, 7.2, 14.3943405],
            [-2.1661716, 5.600004, 0.33361638],
            [5.591684, 8.000006, 3.5366364],
            [-18.63605, 9.400011, -20.17137],
            [-51.035877, 7.00001, -20.661451],
            [-54.75511, 8.20001, 29.686047],
            [-75.78642, 3.400007, -0.28118283]
        ]

        self.static_lights = [
            Light(
                position = pos,
                color = [
                    np.random.uniform(0.2, 1.0),
                    np.random.uniform(0.2, 1.0),
                    np.random.uniform(0.2, 1.0),
                ],
                strength = 18
            )
            for pos in positions
        ]

        self.dynamic_lights = {
        }

        self.colliders = {
            'ground': LockedRectCollider([-100.01, -2, -100.01], [100.01, 0.01, 100.01], debug=True),
            'platform': LockedRectCollider([-23, 4.4, -7], [-37, 4.8, 6.5], debug=True),
            'x': LockedRectCollider([9.5, -0.5, -0.5], [10.5, 0.5, 0.5], debug=True),
            #'y': LockedRectCollider([-0.5, 9.5, -0.5], [0.5, 10.5, 0.5], debug=True),
            'z': LockedRectCollider([-0.5, -0.5, 9.5], [0.5, 0.5, 10.5], debug=True),
            'collision': LockedRectCollider([-0.5, -9.5, -0.5], [0.5, -9.6, 0.5], debug=True),
            '1': LockedRectCollider([-0.5, -9.5, -0.5], [0.5, -9.6, 0.5], debug=True),
            '2': LockedRectCollider([-0.5, -9.5, -0.5], [0.5, -9.6, 0.5], debug=True),
            '3': LockedRectCollider([-0.5, -9.5, -0.5], [0.5, -9.6, 0.5], debug=True),
            '4': LockedRectCollider([-0.5, -9.5, -0.5], [0.5, -9.6, 0.5], debug=True),
            '5': LockedRectCollider([-0.5, -9.5, -0.5], [0.5, -9.6, 0.5], debug=True),
            '6': LockedRectCollider([-0.5, -9.5, -0.5], [0.5, -9.6, 0.5], debug=True),
            '7': LockedRectCollider([-0.5, -9.5, -0.5], [0.5, -9.6, 0.5], debug=True),
            '8': LockedRectCollider([-0.5, -9.5, -0.5], [0.5, -9.6, 0.5], debug=True),

            #'stair1': LockedRectCollider([-18.4, -0.1, 1.2], [-19, 0.6, -1.2], debug=True),
            #'test': LockedRectCollider([-2, -2, -2], [2, 2, 2], debug=True)
        }

        self.rigid_bodies = {
            #'cube':  Cuboid(GFX_PATH + "wood.jpeg", pos = [9, 1.6, 0], velocity = [0.025, 0.05, 0], size=[6, 1, 6], angular_velocity = [0, np.pi / TPS, 0]),
            'cube2': Cuboid(GFX_PATH + "wood.jpeg", pos = [0, DEFAULT_PLAYER_HEIGHT - 1, 0], size=[10, 2, 10], eulers=[np.pi / TPS, np.pi / TPS, np.pi / TPS]),#, angular_velocity = [np.pi / TPS, np.pi / TPS, np.pi / TPS]), #np.pi / 2, -np.pi / 2
            #'cube3': Cuboid(GFX_PATH + "wood.jpeg", pos = [2.2, 1, 0], size=[1, 1, 1]),
            #'cube4': Cuboid(GFX_PATH + "wood.jpeg", pos = [4.4, 1, 0], size=[1, 1, 1]),
            'sphere': Sphere(GFX_PATH + "wood.jpeg", pos = [0, 9, -3], radius=0.4, col = self.colliders),
            'x': Cuboid(GFX_PATH + "wood.jpeg", pos = [10, 0, 0], size=[0.5, 0.5, 0.5], angular_velocity = [np.pi / TPS, 0, 0]),
            #'y': Cuboid(GFX_PATH + "wood.jpeg", pos = [0, 10, 0], size=[0.5, 0.5, 0.5], angular_velocity = [0, np.pi / TPS, 0]),
            'z': Cuboid(GFX_PATH + "wood.jpeg", pos = [0, 0, 10], size=[0.5, 0.5, 0.5], angular_velocity = [0, 0, np.pi / TPS]),
        }

        stair_count = 8
        for x in range(stair_count):
            self.colliders[f"stair{x}"] = LockedRectCollider([-18.4 - x*0.6, -0.7 + x*0.59, 1.2], [-19 - x*0.6, 0.6 + x*0.59, -1.2], debug=True)

        self.fps_monitor = FrameRateMonitor("SCENE")

    def set_window(self, window):
        self.window = window

    def set_player_pos(self, new_player_pos):
        with self.lock:
            self.player_pos = np.array(new_player_pos, dtype=np.float32)

    def get_player_feet(self):
        pos = self.player_pos
        return pos + np.array([0, -DEFAULT_PLAYER_HEIGHT, 0])

    def set_player_feet(self, new_player_feet_pos):
        self.set_player_pos(np.array(new_player_feet_pos, dtype=np.float32) + np.array([0, DEFAULT_PLAYER_HEIGHT, 0], dtype=np.float32))

    def get_player_acceleration(self):
        with self.lock:
            return self.player_acceleration
        
    def set_player_acceleration(self, new_player_acceleration):
        with self.lock:
            self.player_acceleration = new_player_acceleration
        
    def add_player_acceleration(self, add_player_acceleration):
        np_add_player_acceleration = np.array(add_player_acceleration)

        with self.lock:
            self.player_acceleration += np_add_player_acceleration

    def get_player_pos(self):
        with self.lock and self.player_lock:
            return self.player_pos
        
    def get_player_forwards(self):
        with self.lock:
            return self.player_forwards
        
    def get_player_up(self):
        with self.lock:
            return self.player_up
    
    def set_mouse_pos(self, x, y):
        with self.lock:
            self.mouse_pos = np.array([x, y], dtype=np.int16)

    def get_mouse_pos(self):
        with self.lock:
            return self.mouse_pos
        
    def set_do_screenshot(self, bool_should_do_screenshot):
        with self.lock:
            self.do_screenshot = bool_should_do_screenshot

    def get_do_screenshot(self):
        with self.lock:
            return self.do_screenshot

    def set_should_center_cursor(self, bool_should_center_cursor):
        with self.lock:
            self.should_center_cursor = bool_should_center_cursor

    def get_should_center_cursor(self):
        with self.lock:
            return self.should_center_cursor
        
    def get_static_lights(self):
        with self.lock:
            return self.static_lights
        
    def get_dynamic_lights(self):
        with self.lock:
            return self.dynamic_lights

    def get_dynamic_lights_list(self):
        with self.lock:
            return list(self.dynamic_lights.values())
        
    def get_colliders_list(self):
        with self.lock:
            return self.colliders.values()
        
    def get_rigid_bodies_list(self):
        with self.lock:
            return self.rigid_bodies.values()
    
    def get_objects_list(self):
        with self.lock:
            return self.objects.values()
    
    def handle_keys(self):
        combo = 0
        direction_modifier = 0
        d_pos = np.zeros(3, dtype=np.float32)
        d_acceleration = np.zeros(3, dtype=np.float32)
        
        # Handle movement (WASD)
        if glfw.get_key(self.window, glfw.KEY_W): combo += 1
        if glfw.get_key(self.window, glfw.KEY_D): combo += 2
        if glfw.get_key(self.window, glfw.KEY_S): combo += 4
        if glfw.get_key(self.window, glfw.KEY_A): combo += 8
        
        # Direction modifier dictionary for common WASD combinations
        direction_modifiers = {
            1: 360,   # w
            3: 45,    # w & a
            2: 90,    # a
            7: 90,    # w & a & s
            6: 135,   # a & s
            4: 180,   # s
            14: 180,  # a & s
            12: 225,  # s & d
            8: 270,   # d
            13: 270,  # w & s & d
            9: 315,   # w & d
        }
        
        # Check for valid combo and assign corresponding direction modifier
        if combo in direction_modifiers:
            direction_modifier = direction_modifiers[combo]

        sprinting = glfw.get_key(self.window, glfw.KEY_LEFT_SHIFT)
        if sprinting:
            speed_multiplier = PLAYER_ACCELERATION_SPRINT_MULTIPLIER
        else:
            speed_multiplier = 1
        
        # Calculate movement based on direction modifier
        if direction_modifier:
            d_acceleration[0] = PLAYER_ACCELERATION * speed_multiplier * np.cos(np.deg2rad(self.player_forward_vector[0] + direction_modifier))
            d_acceleration[2] = PLAYER_ACCELERATION * speed_multiplier * np.sin(np.deg2rad(self.player_forward_vector[0] + direction_modifier))
        
        # Handle vertical movement (space = up, ctrl = down)
        if FLIGHT_ENABLED:
            if glfw.get_key(self.window, glfw.KEY_SPACE):
                d_pos[1] = PLAYER_ACCELERATION * speed_multiplier * 8
            elif glfw.get_key(self.window, glfw.KEY_LEFT_CONTROL):
                d_pos[1] = -PLAYER_ACCELERATION * speed_multiplier * 8
        else:
            if glfw.get_key(self.window, glfw.KEY_SPACE):
                if not self.previous_space_state:
                    self.add_player_acceleration([0, JUMP_STRENGTH, 0])
                self.previous_space_state = True
            else:
                self.previous_space_state = False

        with self.lock:
            self.player_pos += d_pos

        self.add_player_acceleration(d_acceleration)

        if glfw.get_key(self.window, glfw.KEY_F12):
            if not self.previous_f12_state:
                self.set_do_screenshot(True)
            self.previous_f12_state = True
        else:
            self.previous_f12_state = False

        if glfw.get_key(self.window, glfw.KEY_ENTER):
            print("\033[32m" + str(self.get_player_feet()) + "\033[0m")

    def handle_mouse(self):
        x, y = glfw.get_cursor_pos(self.window)
        self.set_should_center_cursor(True)

        theta_increment = MOUSE_SENSITIVITY * ((self.screen_width // 2) - x)
        phi_increment = MOUSE_SENSITIVITY * ((self.screen_height // 2) - y)
        
        self.spin(-theta_increment, phi_increment)

    def spin(self, d_theta, d_phi):        
        self.player_forward_vector[0] += d_theta
        self.player_forward_vector[0] %= 360
    
        self.player_forward_vector[1] = min(MAX_LOOK_THETA, max(-MAX_LOOK_THETA, self.player_forward_vector[1] + d_phi))
        
        self.update_player_forwards()
    
    def update_player_forwards(self):
        with self.lock:
            self.player_forwards = np.array(
                [
                    np.cos(np.deg2rad(self.player_forward_vector[0])) * np.cos(np.deg2rad(self.player_forward_vector[1])),
                    np.sin(np.deg2rad(self.player_forward_vector[1])),
                    np.sin(np.deg2rad(self.player_forward_vector[0])) * np.cos(np.deg2rad(self.player_forward_vector[1]))
                ],
                dtype = np.float32
            )

            right = np.cross(self.player_forwards, GLOBAL_UP)
            self.player_up = np.cross(right, self.player_forwards)

    def gravity(self):
        if FLIGHT_ENABLED:
            return
        with self.lock:
            self.player_acceleration[1] -= GRAVITY

    def move_player(self):
        with self.lock:
            self.player_pos += self.player_acceleration

            self.player_acceleration *= np.array([HORIZONTAL_DRAG, VERTICAL_DRAG, HORIZONTAL_DRAG])
    
    def player_collision(self):
        pos = self.get_player_feet()
        colliders = self.get_colliders_list()
        
        for collider in colliders:
            if pos[0] > collider.left and pos[0] < collider.right and pos[1] > collider.bottom and pos[1] < collider.top and pos[2] > collider.front and pos[2] < collider.back:
                self.set_player_feet([pos[0], collider.top, pos[2]])
                self.player_acceleration *= np.array([1, 0, 1])

    def rigid_body_collision(self):
        rigid_bodies = self.get_rigid_bodies_list()
        
        for rigid_body in rigid_bodies:
            rigid_body.simulate(rigid_bodies)

    def main(self):
        while self.running:
            start_time = time.perf_counter()

            # PLAYER
            self.handle_keys()
            self.handle_mouse()

            with self.player_lock:
                self.gravity()
                self.move_player()
                self.player_collision()

            self.rigid_body_collision()

            """
            self.dynamic_lights = {
                'player': Light(
                    position = self.get_player_pos(),
                    color = [
                        0.8, 0.6, 0.8
                    ],
                    strength = 9
                )
            }
            """

            # Tick
            self.fps_monitor.run()

            end_time = time.perf_counter()
            remaining_tick_delay = max(SPT - (end_time - start_time), 0)
            time.sleep(remaining_tick_delay)

    def quit(self):
        with self.lock:
            self.running = False

        for object in self.objects.values():
            object.destroy()


### Entry point ###
def main():
    window = Window()
    graphics_engine = GraphicsEngine(window.aspect)
    scene = Scene(events, window, (window.screen_width, window.screen_height))

    #scene.set_window(window.window)
    window.init(graphics_engine, scene)

    scene_thread = threading.Thread(target=scene.main)
    scene_thread.start()

    window.main()

    scene_thread.join()
    sys.exit(0)

if __name__ == "__main__":
    main()