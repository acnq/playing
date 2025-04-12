import taichi as ti
ti.init(arch=ti.vulkan)

n = 128
# The n x n grid is normalized
# The distance between two x- or z-axis adjacent points
# is 1.0 / n
quad_size = 1.0 / n
dt = 4e-2 / n
substeps = int(1 / 60 // dt)

# Gravity is a force applied in the negative direction of the y axis,
# and so is set to [0, -9.8, 0]
gravity = ti.Vector([0, -9.8, 0])
# Elastic coefficient of springs
spring_Y = 3e4
# Damping coefficient caused by
# the relative movement of the two mass points
# The assumption here is:
# A mass point can have at most 12 'influential` points
dashpot_damping = 1e4
# Damping coefficient of springs
drag_damping = 1


ball_radius = 0.3
# Use a 1D field for storing the position of the ball center
# The only element in the field is a 3-dimentional floating-point vector
ball_center = ti.Vector.field(3, dtype=float, shape=(1, ))
# Place the ball center at the original point
ball_center[0] = [0, 0, 0]

# x is an n x n field consisting of 3D floating-point vectors
# representing the mass points' positions
x = ti.Vector.field(3, dtype=float, shape=(n, n))
# v is an n x n field consisting of 3D floating-point vectors
# representing the mass points' velocities
v = ti.Vector.field(3, dtype=float, shape=(n, n))

num_triangels = (n - 1) * (n - 1) * 2
indices = ti.field(int, shape=num_triangels * 3)
vertices = ti.Vector.field(3, dtype=float, shape=n * n)
colors = ti.Vector.field(3, dtype=float, shape=n * n)

# bending_springs = False
bending_springs = True

# The @ti.kernel decorator instructs Taichi to
# automatically parallelize all top-level for loops
# inside initialize_mass_points()
@ti.kernel
def initialize_mass_points():
  # A random offset to apply to each mass point
  random_offset = ti.Vector([ti.random() - 0.5, ti.random() - 0.5]) * 0.1
  
  # Field x stores the mass points' positions
  for i, j in x:
    # The piece of cloth is 0.6 (y-axis) above the original point
    #
    # By taking away 0.5 from each mass point's [x,z] coordinate value
    # you move the cloth right above the original point
    x[i, j] = [
      i * quad_size - 0.5 + random_offset[0], 0.6,
      j * quad_size - 0.5 + random_offset[1]
    ]
    # The initial velocity of each mass point is set to 0
    v[i, j] = [0, 0, 0]
    
@ti.kernel
def initial_mesh_indices():
  for i, j in ti.ndrange(n - 1, n - 1):
    quad_id = (i * (n - 1)) + j
    # First triangle of the square
    indices[quad_id * 6 + 0] = i * n + j
    indices[quad_id * 6 + 1] = (i + 1) * n + j
    indices[quad_id * 6 + 2] = i * n + (j + 1)
    # Second triangle of the square
    indices[quad_id * 6 + 3] = (i + 1) * n + (j + 1)
    indices[quad_id * 6 + 4] = i * n + (j + 1)
    indices[quad_id * 6 + 5] = (i + 1) * n + j
    
  for i, j in ti.ndrange(n, n):
    if (i // 4 + j // 4) % 2 == 0:
      colors[i * n + j] = (0.22, 0.72, 0.52)
    else:
      colors[i * n + j] = (1, 0.334, 0.52)
      
initial_mesh_indices()

# The cloth is modeled as a mass-spring grid. Assume that:
# a mass point, whose relative index is [0, 0],
# can be affected by at most 12 surrounding points
#
# spring_offsets is such a list storing
# the relative indices of these 'influential' points
spring_offsets = []
if bending_springs:
  for i in range(-1, 2):
    for j in range(-1, 2):
      if (i, j) != (0, 0):
        spring_offsets.append(ti.Vector([i, j]))
else:
  for i in range(-2, 3):
    for j in range(-2, 3):
      if (i, j) != (0, 0) and abs(i) + abs(j) <= 2:
        spring_offsets.append(ti.Vector([i, j]))

# substep() works out the *accumulative* effects
# of gravity, internal force, damping, and collision
# on the mass-spring system
@ti.kernel
def substep():
  # The for loop iterates over all elements of the field v
  for i in ti.grouped(x):
    v[i] += gravity * dt
  
  # The for loop iterates over all elements of the field v
  
  # Traverses the field x as a 1D array
  #
  # The `i` here refers to the *absolute* index
  # of an element in the field x
  #
  # Note that `i` is a 2-dimentional vector here   
  for i in ti.grouped(x):
    # Initial force exerted to a specific mass point
    force = ti.Vector([0.0, 0.0, 0.0])
    # Traverse the surrounding mass points
    for spring_offset in ti.static(spring_offsets):
      # j is the *absolute* index of an 'influential' point
      # Note that j is a 2-dimensional vector here
      j = i + spring_offset
      #  If the 'influential` point is in the n x n grid,
      # then work out the internal force that it exerts
      # on the current mass point
      if 0 <= j[0] < n and 0 <= j[1] < n:
        # The relative displacement of the two points
        # The internal force is related to it
        x_ij = x[i] - x[j]
        # The relative movement of the two points
        v_ij = v[i] - v[j]
        # d is a normalized vector (its norm is 1)
        d = x_ij.normalized()
        current_dist = x_ij.norm()
        original_dist = quad_size * float(i - j).norm()
        # Internal force of the spring
        force += -spring_Y * d * (current_dist / original_dist - 1)
        # Continues to apply the damping force
        # from the relative movement
        # of the two points
        force += -v_ij.dot(d) * d * dashpot_damping * quad_size
        
    # Continues to add the velocity caused by the internal forces
    # to the current velocity
    v[i] += force * dt 
    
  # Traverse the elements in field v
  for i in ti.grouped(x):
    v[i] *= ti.exp(-drag_damping * dt)
    offset_to_center = x[i] - ball_center[0]
    if offset_to_center.norm() <= ball_radius:
      # velocity projection
      normal = offset_to_center.normalized()
      v[i] -= min(v[i].dot(normal), 0) * normal
      
    # After working out the accumulative v[i],
    # work out the positions of each mass point
    x[i] += dt * v[i]
    
@ti.kernel
def update_vertices():
  for i, j in ti.ndrange(n, n):
    vertices[i * n + j] = x[i, j]
    

window = ti.ui.Window("Taichi Cloth Simulation on GGUI", (1024, 1024),
                      vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

current_t = 0.0
initialize_mass_points()

while window.running:
  if current_t > 1.5:
    # Reset
    initialize_mass_points()
    current_t = 0
    
  for i in range(substeps):
    substep()
    current_t += dt
  update_vertices()
  
  camera.position(0.0, 0.0, 3)
  camera.lookat(0.0, 0.0, 0)
  scene.set_camera(camera)
  
  scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
  scene.ambient_light((0.5, 0.5, 0.5))
  scene.mesh(vertices,
              indices=indices,
              per_vertex_color=colors,
              two_sided=True)
  # Draw a smaller ball to avoid visual penetration
  scene.particles(ball_center, radius=ball_radius * 0.95, color=(0.5, 0.42, 0.8))
  canvas.scene(scene)
  window.show()