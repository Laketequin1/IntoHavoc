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


### Constants ###
TPS = 60
FPS_CAP = 60

SPT = 1 / TPS

EYE_SPEED = 1

SHADERS_PATH = "shaders/"
MODELS_PATH = "models/"

PALETTE = {
            "peach": np.array([252/255, 226/255, 219/255], dtype = np.float32),
            "pink": np.array([255/255, 143/255, 177/255], dtype = np.float32),
            "darkPink": np.array([178/255, 112/255, 162/255], dtype = np.float32),
            "purple": np.array([122/255, 68/255, 149/255], dtype = np.float32),
        }


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

        # Upload the vertex data to the GPU
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, gl.GL_STATIC_DRAW)
        
        # Add attribute pointer for position location in buffer so gpu can find vertex data in memory
        gl.glEnableVertexAttribArray(0)
        # location, number of floats, format (float), gl.GL_FALSE, stride (total length of vertex, 4 bytes times number of floats), ctypes of starting position in bytes (void pointer expected)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 12, ctypes.c_void_p(0))
    
    @staticmethod
    def load_mesh(path):
        print(f"Loading mesh {path}...")
        
        vertices = []
        flags = {"v": []}
        
        lines = safe_file_readlines(path)
            
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
                
                for vertex in line:
                    l = vertex.split("/")
                    face_vertices.append(flags["v"][int(l[0]) - 1])
                triangles_in_face = len(line) - 2
                vertex_order = []
                for x in range(triangles_in_face):
                    vertex_order.extend((0, x + 1, x + 2))
                for x in vertex_order:
                    vertices.extend(face_vertices[x])
        
        print(f"Finished loading mesh {path}!")
        return vertices
    
    def destroy(self):
        # Remove allocated memory
        gl.glDeleteVertexArrays(1, (self.vao, ))
        gl.glDeleteBuffers(1, (self.vbo, ))


class Object:
    def __init__(self, mesh_path, base_color, style, pos = np.zeros(3), rotation = np.zeros(3), scale = np.ones(3)):
        self.base_color = base_color
        self.style = style
        self.pos = pos
        self.rotation = rotation
        self.scale = scale

        self.mesh = Mesh(mesh_path)

    def render(self, model_matrix_handle, color_handle):
        gl.glUniform3fv(color_handle, 1, self.base_color)
        
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
        
        # Draw
        if self.style == 0:
            gl.glDrawArrays(gl.GL_LINES, 0, self.mesh.vertex_count)
        elif self.style == 1:
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.mesh.vertex_count)


class GraphicsEngine:
    def __init__(self, aspect):
        # Initilize OpenGL
        gl.glClearColor(*PALETTE["purple"], 1)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        # Initilize shader
        self.shader = self.create_shader(SHADERS_PATH + "default.vert", SHADERS_PATH + "default.frag")
        gl.glUseProgram(self.shader)

        # Initilize projection
        projection_handle =        gl.glGetUniformLocation(self.shader, "projection")

        projection_transform = pyrr.matrix44.create_perspective_projection(fovy = 93, aspect = aspect, near = 0.1, far = 200, dtype = np.float32)
        gl.glUniformMatrix4fv(projection_handle, 1, gl.GL_FALSE, projection_transform)

        self.model_matrix_handle = gl.glGetUniformLocation(self.shader, "model")
        self.view_matrix_handle =  gl.glGetUniformLocation(self.shader, "view")
        self.color_handle =        gl.glGetUniformLocation(self.shader, "object_color")
    
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
        
        pos = scene.get_pos()

        view_transform = pyrr.matrix44.create_look_at(
            eye = np.array([pos[0], 0, pos[1]], dtype = np.float32),
            target = np.array([pos[0], 0, 9999], dtype = np.float32),
            up = np.array([0, 1, 0], dtype = np.float32),
            dtype = np.float32
        )
        
        gl.glUniformMatrix4fv(self.view_matrix_handle, 1, gl.GL_FALSE, view_transform)
        
        for object in scene.objects.values():
            object.render(self.model_matrix_handle, self.color_handle)


class Window:
    def __init__(self):
        # Set variables
        self.graphics_engine = None
        self.scene = None

        # Initilize variables
        self.running = True
        self.pos_offset = np.array([0, 0], dtype=np.float32)

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

        self.aspect = self.screen_width // self.screen_height
        
        self.window = glfw.create_window(self.screen_width, self.screen_height, "Living Tiles", self.monitor, None)
        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window can't be created")
        
        glfw.make_context_current(self.window)

    def init(self, graphics_engine, scene):
        self.graphics_engine = graphics_engine
        self.scene = scene

    def handle_window_events(self) -> None:
        """
        Handle GLFW events and closing the window.
        """
        # Check if window should close
        if glfw.window_should_close(self.window) or glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
            self.close()
            return

    def render(self, graphics_engine: GraphicsEngine, scene):
        graphics_engine.render_graphics(scene)
        glfw.swap_buffers(self.window)

    def tick(self) -> None:
        """
        Tick (manage frame rate).
        """
        glfw.poll_events()

    def close(self) -> None:
        """
        Close the GLFW window and terminate.
        """
        self.running = False
        self.scene.quit()
        glfw.terminate()

    @staticmethod
    def check_gl_error():
        error = gl.glGetError()
        if error != gl.GL_NO_ERROR:
            print(f"OpenGL error: {error}")

    def main(self) -> None:
        """
        Main window loop.
        """
        while self.running:
            pos = self.scene.get_pos()
            self.render(self.graphics_engine, self.scene)
            self.tick()
            self.check_gl_error()
            self.handle_window_events()


class Scene():
    def __init__(self, events):
        self.events = events
        self.window = None

        self.lock = threading.Lock()

        self.running = True
        self.pos = np.array([0, 0], dtype=np.float32)

        # Initilize Objs
        self.objects = {
            'mountain': Object(MODELS_PATH + "mountains.obj", PALETTE["pink"], 0, np.array([0, -8, 0]), np.array([np.pi / 2, np.pi, 0])),
            'ship':     Object(MODELS_PATH + "ship.obj", PALETTE["pink"], 1, scale = np.array([0.6, 0.6, 0.6]))
        }

    def set_window(self, window):
        self.window = window

    def get_pos(self):
        with self.lock:
            return self.pos

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
            self.pos[0] += (move_x * EYE_SPEED)
            self.pos[1] += (move_y * EYE_SPEED)

    def main(self):
        while self.running:
            start_time = time.perf_counter()

            self.handle_inputs()

            end_time = time.perf_counter()
            remaining_tick_delay = max(SPT - (end_time - start_time), 0)
            time.sleep(remaining_tick_delay)

    def quit(self):
        with self.lock:
            self.running = False


### Entry point ###
def main():
    window = Window()
    graphics_engine = GraphicsEngine(window.aspect)
    scene = Scene(events)

    scene.set_window(window.window)
    window.init(graphics_engine, scene)

    scene_thread = threading.Thread(target=scene.main)
    scene_thread.start()

    window.main()

    scene_thread.join()
    sys.exit(0)

if __name__ == "__main__":
    main()