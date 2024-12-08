import pygame as pg
import OpenGL.GL as gl
import numpy as np
import ctypes
import OpenGL.GL.shaders as gls
import pyrr

class App:
    
    def __init__(self, screen_size):
        
        self.screen_width, self.screen_height = screen_size
        
        self.renderer = GraphicsEngine()
        
        self.scene = Scene()
        
        self.last_time = pg.time.get_ticks()
        self.current_time = 0
        self.num_frames = 0
        self.frame_time = 0
        self.light_count = 0
        
        # Start main
        self.main_loop()
        
    def main_loop(self):
        
        running = True
        while running:
            # Check events
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        running = False
            
            self.handle_keys()
            
            self.scene.update(self.frame_time * 0.05)
            
            self.renderer.render(self.scene)
            
            # Timing
            self.calculate_framerate()
            
        self.quit()
    
    def handle_keys(self):
        
        keys = pg.key.get_pressed()
        rate = self.frame_time / 16
        
        # Left, right, space
        if keys[pg.K_LEFT]:
            self.scene.move_player(rate * np.array([0.2, 0, 0], dtype=np.float32))
        if keys[pg.K_RIGHT]:
            self.scene.move_player(rate * np.array([-0.2, 0, 0], dtype=np.float32))
            
        if keys[pg.K_UP]:
            self.scene.move_player(rate * np.array([0, 0.2, 0], dtype=np.float32))
        if keys[pg.K_DOWN]:
            self.scene.move_player(rate * np.array([0, -0.2, 0], dtype=np.float32))
        
        if keys[pg.K_x]:
            self.scene.move_player(rate * np.array([0, 0, 0.2], dtype=np.float32))
        if keys[pg.K_c]:
            self.scene.move_player(rate * np.array([0, 0, -0.2], dtype=np.float32))
            
        if keys[pg.K_SPACE]:
            self.scene.player.shoot()
    
    def calculate_framerate(self):
        
        self.current_time = pg.time.get_ticks()
        delta = self.current_time - self.last_time
        
        if (delta >= 1000):
            framerate = max(1,int(1000.0 * self.num_frames/delta))
            pg.display.set_caption(f"Running at {framerate} fps.")
            self.last_time = self.current_time
            self.num_frames = -1
            self.frame_time = float(1000.0 / max(1,framerate))
            
        self.num_frames += 1
    
    def quit(self):
        
        # Remove allocated memory
        self.renderer.destroy()


class SimpleComponent:
    
    def __init__(self, position, velocity):
        
        self.position = np.array(position, dtype=np.float32)
        self.velocity = np.array(velocity, dtype=np.float32)


class SentientComponent:
    
    def __init__(self, position, eulers, health):
        
        self.position = np.array(position, dtype=np.float32)
        self.eulers = np.array(eulers, dtype=np.float32)
        self.velocity = np.array([0, 0, 0], dtype=np.float32)
        
        self.state = "stable"
        self.health = health
        self.can_shoot = True
        self.reload_time = 0

    def shoot(self):
        
        if self.can_shoot:
            print("Shoot!")
            self.can_shoot = False
            self.reload_time = 5


class Scene:
    
    def __init__(self):        
        self.enemy_spawn_rate = 0.1
        self.powerup_spawn_rate = 0.05
        self.enemy_shoot_rate = 0.1
        
        self.player = SentientComponent(
            position = [0, 1.2, 2],
            eulers = [0, 0, 0],
            health = 36
        )
        
        self.enemys = []
        self.bullets = []
        self.powerups = []
        
    def update(self, rate):        
        pass
    
    def move_player(self, d_pos):            
        self.player.position += d_pos
        
        self.player.position[0] = min(3, max(-3, self.player.position[0]))


class GraphicsEngine:
    
    def __init__(self):        
        self.palette = {
            "peach": np.array([252/255, 226/255, 219/255], dtype = np.float32),
            "pink": np.array([255/255, 143/255, 177/255], dtype = np.float32),
            "darkPink": np.array([178/255, 112/255, 162/255], dtype = np.float32),
            "purple": np.array([122/255, 68/255, 149/255], dtype = np.float32),
        }
        
        # Initilize pygame
        pg.init()
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK,
                                    pg.GL_CONTEXT_PROFILE_CORE)
        pg.display.set_mode((640,480), pg.OPENGL|pg.DOUBLEBUF)
        
        # Initilize OpenGL
        gl.glClearColor(*self.palette["purple"], 1)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        
        # Create renderpasses and resources
        shader = self.create_shader("shaders/vertex.txt", "shaders/fragment.txt")
        self.render_pass = RenderPass(shader)
        self.mountain_mesh = Mesh("models/mountains.obj")
        self.grid_mesh = Grid(26)
        self.ship_mesh = Mesh("models/ship.obj")
        
    def create_shader(self, vertexFilepath, fragmentFilepath):
        
        with open(vertexFilepath, 'r') as f:
            vertex_src = f.readlines()
        
        with open(fragmentFilepath, 'r') as f:
            fragment_src = f.readlines()
        
        shader = gls.compileProgram(
            gls.compileShader(vertex_src, gl.GL_VERTEX_SHADER),
            gls.compileShader(fragment_src, gl.GL_FRAGMENT_SHADER)
        )
        
        return shader
        
    def render(self, scene):
            
        # Refresh screen
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        
        # Draw
        self.render_pass.render(scene, self)
        
        pg.display.flip()
    
    def destroy(self):
        
        pg.quit()


class RenderPass:
    
    def __init__(self, shader):
        
        # Initilize OpenGL
        self.shader = shader
        gl.glUseProgram(self.shader)
        
        projection_transform = pyrr.matrix44.create_perspective_projection(
            fovy = 45, aspect = 800/600,
            near = 0.1, far = 100, dtype = np.float32
        )
        gl.glUniformMatrix4fv(
            gl.glGetUniformLocation(self.shader, "projection"),
            1, gl.GL_FALSE, projection_transform
        )
        self.model_matrix_location = gl.glGetUniformLocation(self.shader, "model")
        self.view_matrix_location = gl.glGetUniformLocation(self.shader, "view")
        self.color_loc = gl.glGetUniformLocation(self.shader, "object_color")
    
    def render(self, scene, engine):
        
        gl.glUseProgram(self.shader)
        
        view_transform = pyrr.matrix44.create_look_at(
            eye = np.array([0, 3, -5], dtype = np.float32),
            target = np.array([0, 3, 1], dtype = np.float32),
            up = np.array([0, 1, 0], dtype = np.float32),
            dtype = np.float32
        )
        
        gl.glUniformMatrix4fv(self.view_matrix_location, 1, gl.GL_FALSE, view_transform)
        
        # Mountains
        gl.glUniform3fv(self.color_loc, 1, engine.palette["pink"])
        
        model_transform = pyrr.matrix44.create_identity(dtype = np.float32)
        
        model_transform = pyrr.matrix44.multiply(
            m1 = model_transform,
            m2 = pyrr.matrix44.create_from_y_rotation(theta = np.radians(90), dtype = np.float32)
        )
        
        model_transform = pyrr.matrix44.multiply(
            m1 = model_transform,
            m2 = pyrr.matrix44.create_from_x_rotation(theta = np.radians(-90), dtype = np.float32)
        )
        
        model_transform = pyrr.matrix44.multiply(
            m1 = model_transform,
            m2 = pyrr.matrix44.create_from_z_rotation(theta = np.radians(90), dtype = np.float32)
        )

        model_transform = pyrr.matrix44.multiply(
            m1 = model_transform,
            m2 = pyrr.matrix44.create_from_translation(vec = np.array([0.8, 0, 20]), dtype = np.float32)
        )
        
        gl.glUniformMatrix4fv(self.model_matrix_location, 1, gl.GL_FALSE, model_transform)
        
        gl.glBindVertexArray(engine.mountain_mesh.vao)
        gl.glDrawArrays(gl.GL_LINES, 0, engine.mountain_mesh.vertex_count)
        
        # Ground
        gl.glUniform3fv(self.color_loc, 1, engine.palette["pink"])
        
        model_transform = pyrr.matrix44.create_identity(dtype = np.float32)

        model_transform = pyrr.matrix44.multiply(
            m1 = model_transform,
            m2 = pyrr.matrix44.create_from_translation(vec = np.array([-12, 0, -5]), dtype = np.float32)
        )
        
        gl.glUniformMatrix4fv(self.model_matrix_location, 1, gl.GL_FALSE, model_transform)
        
        gl.glBindVertexArray(engine.grid_mesh.vao)
        gl.glDrawArrays(gl.GL_LINES, 0, engine.grid_mesh.vertex_count)
        
        # Player
        gl.glUniform3fv(self.color_loc, 1, engine.palette["peach"])
        
        model_transform = pyrr.matrix44.create_identity(dtype = np.float32)

        model_transform = pyrr.matrix44.multiply(
            m1 = model_transform,
            m2 = pyrr.matrix44.create_from_scale(scale = np.array([0.6, 0.6, 0.6]), dtype = np.float32)
        )
        
        model_transform = pyrr.matrix44.multiply(
            m1 = model_transform,
            m2 = pyrr.matrix44.create_from_translation(vec = scene.player.position, dtype = np.float32)
        )
        
        gl.glUniformMatrix4fv(self.model_matrix_location, 1, gl.GL_FALSE, model_transform)
        
        gl.glBindVertexArray(engine.ship_mesh.vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, engine.ship_mesh.vertex_count)
    
    def destroy(self):
        
        gl.glDeleteProgram(self.shader)


class Mesh:
    
    def __init__(self, filepath):
        
        # x, y, z, s, t, nx, ny, nz
        self.vertices = self.load_mesh(filepath)
        
        # // is integer division
        self.vertex_count = len(self.vertices) // 3
        self.vertices = np.array(self.vertices, dtype=np.float32)
        
        # Create a vertex array where attributes for buffer are going to be stored, bind to make active, needs done before buffer
        self.vao = gl.glGenVertexArrays(1) 
        gl.glBindVertexArray(self.vao)
        
        # Create a vertex buffer where the raw data is stored, bind to make active, then store the raw data at the location
        self.vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, gl.GL_STATIC_DRAW)
        
        # Enable attributes for buffer. Add attribute pointer for buffer so gpu knows what data is which. Vertex shader.
        # Location 1 - Postion
        gl.glEnableVertexAttribArray(0)
        # Location, number of floats, format (float), gl.GL_FALSE, stride (total length of vertex, 4 bytes times number of floats), ctypes of starting position in bytes (void pointer expected)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 12, ctypes.c_void_p(0))
    
    @staticmethod
    def load_mesh(filepath):
        
        print("Loading mesh...")
        
        vertices = []
        
        flags = {"v": []}
        
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
                    
                    for vertex in line:
                        l = vertex.split("/")
                        face_vertices.append(flags["v"][int(l[0]) - 1])
                    triangles_in_face = len(line) - 2
                    vertex_order = []
                    for x in range(triangles_in_face):
                        vertex_order.extend((0, x + 1, x + 2))
                    for x in vertex_order:
                        vertices.extend(face_vertices[x])
        
        print("Finished loading mesh!")
        
        return vertices
    
    def destroy(self):
        
        # Remove allocated memory
        gl.glDeleteVertexArrays(1, (self.vao, ))
        gl.glDeleteBuffers(1, (self.vbo, ))
        

class Grid:
    
    def __init__(self, size):
        
        self.vertices = []
        
        for x in range(size):
            self.vertices.extend((x, 0, 0))
            self.vertices.extend((x, 0, size - 1))
        
        for z in range(size):
            self.vertices.extend((0, 0, z))
            self.vertices.extend((size - 1, 0, z))
            
        # // is integer division
        self.vertex_count = len(self.vertices) // 3
        self.vertices = np.array(self.vertices, dtype=np.float32)
        
        # Create a vertex array where attributes for buffer are going to be stored, bind to make active, needs done before buffer
        self.vao = gl.glGenVertexArrays(1) 
        gl.glBindVertexArray(self.vao)
        
        # Create a vertex buffer where the raw data is stored, bind to make active, then store the raw data at the location
        self.vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, gl.GL_STATIC_DRAW)
        
        # Enable attributes for buffer. Add attribute pointer for buffer so gpu knows what data is which. Vertex shader.
        # Location 1 - Postion
        gl.glEnableVertexAttribArray(0)
        # Location, number of floats, format (float), gl.GL_FALSE, stride (total length of vertex, 4 bytes times number of floats), ctypes of starting position in bytes (void pointer expected)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 12, ctypes.c_void_p(0))

    def destroy(self):
        
        # Remove allocated memory
        gl.glDeleteVertexArrays(1, (self.vao, ))
        gl.glDeleteBuffers(1, (self.vbo, ))
        

if __name__ == "__main__":
    myApp = App((800, 600))