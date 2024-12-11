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
from datetime import datetime
from PIL import Image

from src import COLOURS

### Constants ###
TPS = 60
FPS_CAP = 0 # Set to 0 for uncapped FPS
FOV = 93

SPT = 1 / TPS

GL_ERROR_CHECK_DELAY_SEC = 5

EYE_SPEED = 0.6
MOUSE_SENSITIVITY = 0.12
MAX_LOOK_THETA = 89.95 # Must be < 90 degrees

GRAVITY = 0.000981

SHADERS_PATH = "shaders/"
MODELS_PATH = "models/"
GFX_PATH = "gfx/"
SCREENSHOTS_PATH = "screenshots/"

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
        self.frame_times = []
        self.last_update_time = None
        self.name = name

        self.total_elapsed = 0

    def print_fps(self):
        if len(self.frame_times) and self.total_elapsed:
            fps = len(self.frame_times) / self.total_elapsed

            print(f"[{self.name}] FPS: {round(fps, 3)} LOW: {round(len(self.frame_times) / (max(*self.frame_times, 0.001) * len(self.frame_times)), 3)} HIGH: {round(len(self.frame_times) / (max(min(self.frame_times), 0.001) * len(self.frame_times)), 3)}")

        self.frame_times = []
        self.total_elapsed = 0

    def run(self):
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


class Object:
    def __init__(self, mesh_path, material_path, pos = np.zeros(3), rotation = np.zeros(3), scale = np.ones(3), base_color = COLOURS.RED1, outline = False):
        self.pos = np.array(pos, dtype=np.float32)
        self.rotation = np.array(rotation, dtype=np.float32)
        self.scale = np.array(scale, dtype=np.float32)
        self.base_color = np.array(base_color, dtype=np.float32)
        self.outline = outline

        self.mesh = Mesh(mesh_path)
        self.material = Material(material_path)
        self.material.use()

    def render(self, model_matrix_handle, color_handle):
        gl.glUniform3fv(color_handle, 1, self.base_color)
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

        # Draw debug lines
        if self.outline:
            gl.glUniform3fv(color_handle, 1, self.base_color)
            gl.glDrawArrays(gl.GL_LINES, 0, self.mesh.vertex_count)
    
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
        texture_handle = gl.glGetUniformLocation(self.shader, "imageTexture") # imageTexture
        gl.glUniform1i(texture_handle, 0) # NEED TO CHANGE FOR EACH OBJECT??!?!?!?!??! (MAYBE)

        # Initilize projection
        projection_handle =        gl.glGetUniformLocation(self.shader, "projection")

        projection_transform = pyrr.matrix44.create_perspective_projection(fovy = FOV, aspect = aspect, near = 0.1, far = 200, dtype = np.float32)
        gl.glUniformMatrix4fv(projection_handle, 1, gl.GL_FALSE, projection_transform)

        self.model_matrix_handle = gl.glGetUniformLocation(self.shader, "model")
        self.view_matrix_handle =  gl.glGetUniformLocation(self.shader, "view")
        self.color_handle =        gl.glGetUniformLocation(self.shader, "objectColor")
        self.camera_pos_handle =   gl.glGetUniformLocation(self.shader, "cameraPosition") # cameraPosition

        """
        self.light_handle = {
            "position": [
                gl.glGetUniformLocation(self.shader, f"Lights[{i}].position")
                for i in range(8)
                ],
            "color": [
                gl.glGetUniformLocation(self.shader, f"Lights[{i}].color")
                for i in range(8)
                ],
            "strength": [
                gl.glGetUniformLocation(self.shader, f"Lights[{i}].strength")
                for i in range(8)
                ]
        }
        """
    
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
        
        for object in scene.objects.values():
            object.render(self.model_matrix_handle, self.color_handle)

        """
        for i, light in enumerate(scene.lights):
            gl.glUniform3fv(self.light_location["position"][i], 1, light.position)
            gl.glUniform3fv(self.light_location["color"][i], 1, light.color)
            gl.glUniform1f(self.light_location["strength"][i], light.strength)
        """

        gl.glUniform3fv(self.camera_pos_handle, 1, player_pos)

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
            buffer = gl.glReadPixels(0, 0, self.screen_width, self.screen_height, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)

            image_data = np.frombuffer(buffer, dtype=np.uint8).reshape((self.screen_height, self.screen_width, 4))
            image_data = np.flipud(image_data)

            image = Image.fromarray(image_data, "RGBA")

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

        self.running = True
        self.player_pos = np.array([0, 0, 0], dtype=np.float32)
        self.player_acceleration = np.array([0, 0, 0], dtype=np.float32)
        self.player_forward_vector = np.array([0, 0], dtype=np.float32)
        self.update_player_forwards()

        self.mouse_pos = np.array([self.screen_width / 2, self.screen_height / 2], dtype=np.int16)
        self.should_center_cursor = True

        self.do_screenshot = False
        self.previous_f12_state = False

        # Initilize Objs
        self.objects = {
            'mountain': Object(MODELS_PATH + "mountains.obj", GFX_PATH + "wood.jpeg", [0, -8, 0], [np.pi / 2, np.pi, 0]),
            'ship':     Object(MODELS_PATH + "ship.obj", GFX_PATH + "rendering_texture.jpg", scale = [0.6, 0.6, 0.6]),
            'cube':     Object(MODELS_PATH + "cube.obj", GFX_PATH + "rendering_texture.jpg", [0, 10, 0]),
            'test':     Object(MODELS_PATH + "Pipes.obj", GFX_PATH + "PipesBake.png", [0, 15, 0]),
            'cans':     Object(MODELS_PATH + "cans2.obj", GFX_PATH + "BakeImage.png", [0, -5, 0]),
            'scene':    Object(MODELS_PATH + "StartScenePrev2.obj", GFX_PATH + "BakeSceneImage2.png", [-50, 20, 0])
        }

        self.lights = {
        }

        self.fps_monitor = FrameRateMonitor("SCENE")

    def set_window(self, window):
        self.window = window

    def get_player_pos(self):
        with self.lock:
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

    """
    def handle_inputs(self):
        # Eye movement
        move_x, move_y = 0, 0
        if glfw.get_key(self.window, glfw.KEY_W) == glfw.PRESS:
            move_y += 1
        if glfw.get_key(self.window, glfw.KEY_S) == glfw.PRESS:
            move_y -= 1
        if glfw.get_key(self.window, glfw.KEY_A) == glfw.PRESS:
            move_x += 1
        if glfw.get_key(self.window, glfw.KEY_D) == glfw.PRESS:
            move_x -= 1

        length = math.sqrt(move_x ** 2 + move_y ** 2)
        if length != 0:
            move_x /= length
            move_y /= length

        with self.lock:
            self.player_pos[0] += (move_x * EYE_SPEED)
            self.player_pos[2] += (move_y * EYE_SPEED)
        
        self.handle_mouse()
    """
    
    def handle_keys(self):
        combo = 0
        direction_modifier = 0
        d_pos = np.zeros(3, dtype=np.float32)
        
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
        
        # Calculate movement based on direction modifier
        if direction_modifier:
            d_pos[0] = EYE_SPEED * np.cos(np.deg2rad(self.player_forward_vector[0] + direction_modifier))
            d_pos[2] = EYE_SPEED * np.sin(np.deg2rad(self.player_forward_vector[0] + direction_modifier))
        
        # Handle vertical movement (space = up, ctrl = down)
        if glfw.get_key(self.window, glfw.KEY_SPACE):
            d_pos[1] = EYE_SPEED
        elif glfw.get_key(self.window, glfw.KEY_LEFT_CONTROL):
            d_pos[1] = -EYE_SPEED

        with self.lock:
            self.player_pos += d_pos

        if glfw.get_key(self.window, glfw.KEY_F12):
            if not self.previous_f12_state:
                self.set_do_screenshot(True)
            self.previous_f12_state = True
        else:
            self.previous_f12_state = False

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
        with self.lock:
            self.player_acceleration[1] -= GRAVITY

    def move_player(self):
        with self.lock:
            self.player_pos += self.player_acceleration

    def main(self):
        while self.running:
            start_time = time.perf_counter()

            #self.gravity()

            self.handle_keys()
            self.handle_mouse()

            self.move_player()

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