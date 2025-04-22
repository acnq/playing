import taichi as ti
import numpy as np

ti.init(arch=ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda)

# 全局参数
n = 128
quad_size = 1.0 / n
dt = 2e-2 / n
substeps = int(1 / 60 // dt)

gravity = ti.Vector([0, -9.8, 0])
spring_Y = 3e4
dashpot_damping = 1e4
drag_damping = 1

ball_radius = 0.3
ball_center = ti.Vector.field(3, dtype=float, shape=(1,))
ball_center[0] = [0, 0, 0]

x = ti.Vector.field(3, dtype=float, shape=(n, n))
v = ti.Vector.field(3, dtype=float, shape=(n, n))

num_triangles = (n - 1) * (n - 1) * 2
indices = ti.field(int, shape=num_triangles * 3)
vertices = ti.Vector.field(3, dtype=float, shape=n * n)
colors = ti.Vector.field(3, dtype=float, shape=n * n)

bending_springs = False
spring_offsets = []

# 拖拽状态变量
dragging = ti.field(ti.u1, shape=())
picked_node = ti.Vector.field(2, dtype=ti.i32, shape=())
drag_pos = ti.Vector.field(3, dtype=float, shape=())

@ti.kernel
def initialize_mass_points():
    random_offset = ti.Vector([ti.random() - 0.5, ti.random() - 0.5]) * 0.1
    for i, j in x:
        x[i, j] = [
            i * quad_size - 0.5 + random_offset[0],
            0.6,
            j * quad_size - 0.5 + random_offset[1],
        ]
        v[i, j] = [0, 0, 0]

@ti.kernel
def initialize_mesh_indices():
    for i, j in ti.ndrange(n - 1, n - 1):
        quad_id = (i * (n - 1)) + j
        # Triangle 1
        indices[quad_id * 6 + 0] = i * n + j
        indices[quad_id * 6 + 1] = (i + 1) * n + j
        indices[quad_id * 6 + 2] = i * n + (j + 1)
        # Triangle 2
        indices[quad_id * 6 + 3] = (i + 1) * n + j + 1
        indices[quad_id * 6 + 4] = i * n + (j + 1)
        indices[quad_id * 6 + 5] = (i + 1) * n + j

    for i, j in ti.ndrange(n, n):
        if (i // 4 + j // 4) % 2 == 0:
            colors[i * n + j] = (0.22, 0.72, 0.52)
        else:
            colors[i * n + j] = (1, 0.334, 0.52)

def initialize_spring_offsets():
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

@ti.kernel
def substep():
    for i in ti.grouped(x):
        v[i] += gravity * dt

    for i in ti.grouped(x):
        force = ti.Vector([0.0, 0.0, 0.0])
        for spring_offset in ti.static(spring_offsets):
            j = i + spring_offset
            if 0 <= j[0] < n and 0 <= j[1] < n:
                x_ij = x[i] - x[j]
                v_ij = v[i] - v[j]
                d = x_ij.normalized()
                current_dist = x_ij.norm()
                original_dist = quad_size * float(i - j).norm()
                force += -spring_Y * d * (current_dist / original_dist - 1)
                force += -v_ij.dot(d) * d * dashpot_damping * quad_size
        v[i] += force * dt

    for i in ti.grouped(x):
        v[i] *= ti.exp(-drag_damping * dt)

        # 拖拽覆盖原速度
        if dragging[None] == 1 and all(i == picked_node[None]):
            v[i] = (drag_pos[None] - x[i]) / dt * 0.5

        offset_to_center = x[i] - ball_center[0]
        if offset_to_center.norm() <= ball_radius:
            normal = offset_to_center.normalized()
            v[i] -= ti.min(v[i].dot(normal), 0) * normal

        x[i] += dt * v[i]
    for i in range(n):
        x[i, n - 1] = [i * quad_size - 0.5, 0.6, 0.5]
        v[i, n - 1] = [0, 0, 0]
@ti.kernel
def update_vertices():
    for i, j in ti.ndrange(n, n):
        vertices[i * n + j] = x[i, j]

def get_mouse_ray(camera, screen_pos, width, height):
    import numpy as np
    view_mat = np.array(camera.get_view_matrix()).reshape(4, 4)
    aspect = width / height
    proj_mat = np.array(camera.get_projection_matrix(aspect)).reshape(4, 4)
    inv_vp = np.linalg.inv(proj_mat @ view_mat)

    ndc_x = screen_pos[0] * 2.0 - 1.0
    ndc_y = 1.0 - screen_pos[1] * 2.0
    near = np.array([ndc_x, ndc_y, -1.0, 1.0])
    far = np.array([ndc_x, ndc_y, 1.0, 1.0])

    near_world = inv_vp @ near
    far_world = inv_vp @ far
    near_world /= near_world[3]
    far_world /= far_world[3]

    ray_origin = near_world[:3]
    ray_dir = far_world[:3] - near_world[:3]
    ray_dir /= np.linalg.norm(ray_dir)
    return ray_origin, ray_dir

def pick_node(ray_ori, ray_dir):
    min_dist = 1e10
    node = None
    for i in range(n):
        for j in range(n):
            pos = x.to_numpy()[i, j]
            to_node = pos - ray_ori
            t = np.dot(to_node, ray_dir)
            closest = ray_ori + t * ray_dir
            dist = np.linalg.norm(closest - pos)
            if dist < 0.02 and dist < min_dist:
                node = (i, j)
                min_dist = dist
    return node

def main():
    window = ti.ui.Window("Cloth Simulation with Drag", (768, 768), vsync=True)
    canvas = window.get_canvas()
    canvas.set_background_color((1, 1, 1))
    scene = window.get_scene()
    camera = ti.ui.Camera()

    initialize_mesh_indices()
    initialize_mass_points()
    initialize_spring_offsets()

    while window.running:
        for _ in range(substeps):
            substep()
        update_vertices()

        # 处理鼠标交互
        if window.is_pressed(ti.ui.LMB):
            mpos = window.get_cursor_pos()
            ray_ori, ray_dir = get_mouse_ray(camera, mpos, 768, 768)
            if dragging[None] == 0:
                picked = pick_node(ray_ori, ray_dir)
                if picked:
                    picked_node[None] = picked
                    dragging[None] = 1
            else:
                # 更新拖拽目标位置
                drag_pos[None] = ray_ori + ray_dir * 1.5
        else:
            dragging[None] = 0

        # 设置相机和绘图
        camera.position(0.0, 0.0, 3)
        camera.lookat(0.0, 0.0, 0)
        scene.set_camera(camera)
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        scene.mesh(vertices, indices=indices, per_vertex_color=colors, two_sided=True)
        scene.particles(ball_center, radius=ball_radius * 0.95, color=(0.5, 0.42, 0.8))
        canvas.scene(scene)
        window.show()

if __name__ == "__main__":
    main()
