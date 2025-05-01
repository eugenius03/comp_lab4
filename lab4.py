import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon
from math import cos, sin
import os

print("Оберіть вісь відносно якої виконати обертання тетраедра:")
print("1 - вісь X")
print("2 - вісь Y")
print("3 - вісь Z")
axis_choice = int(input("Введіть номер осі (1, 2 або 3): "))
while axis_choice not in [1, 2, 3]:
    print("Некоректний вибір")
    axis_choice = int(input("Введіть номер осі (1, 2 або 3): "))


PERSPECTIVE_DISTANCE = 15.0  
ROTATION_SPEED = 0.02 
OBSERVER_POS = np.array([0, 0, -PERSPECTIVE_DISTANCE]) 

scale1 = 1.0
offset1 = np.array([-1.5, 0, 3]) 

scale2 = 0.8
offset2 = np.array([1.5, 0.5, 5]) 


tetra_base_vertices = np.array([
    [1, 1, 1],
    [1, -1, -1],
    [-1, 1, -1],
    [-1, -1, 1]
])

vertices1 = tetra_base_vertices * scale1 + offset1
vertices2 = tetra_base_vertices * scale2 + offset2

faces_indices = np.array([
    [0, 1, 2],
    [0, 3, 1],
    [0, 2, 3],
    [1, 3, 2]
])

objects = [
    {'vertices': vertices1, 'faces': faces_indices, 'color': 'cyan', 'id': 0},
    {'vertices': vertices2, 'faces': faces_indices, 'color': 'magenta', 'id': 1}
]

def rotation_x(angle):
    cos_angle = cos(angle)
    sin_angle = sin(angle)

    R = np.array([
        [1, 0, 0],
        [0, cos_angle, -sin_angle],
        [0, sin_angle, cos_angle]
    ])
    return R


def rotation_y(angle):
    cos_angle = cos(angle)
    sin_angle = sin(angle)

    R = np.array([
        [ cos_angle, 0, sin_angle],
        [ 0,         1, 0],
        [-sin_angle, 0, cos_angle]
    ])
    return R

def rotation_z(angle):
    cos_angle = cos(angle)
    sin_angle = sin(angle)

    R = np.array([
        [cos_angle, -sin_angle, 0],
        [sin_angle,  cos_angle, 0],
        [0,          0,         1]
    ])
    return R

def project(vertices_3d, distance):
    epsilon = 1e-6
    z_coords = vertices_3d[:, 2]
    scale = distance / (distance + z_coords + epsilon)
    scale[scale < 0] = epsilon 
    scale = np.clip(scale, -1000, 1000) 

    projected = vertices_3d[:, :2] * scale[:, np.newaxis]
    return projected

def get_face_properties(face_vertices_3d):
    max_z = np.max(face_vertices_3d[:, 2])
    min_z = np.min(face_vertices_3d[:, 2])
    avg_z = np.mean(face_vertices_3d[:, 2])
    center_3d = np.mean(face_vertices_3d, axis=0)
    return {'vertices_3d': face_vertices_3d, 'max_z': max_z, 'min_z': min_z, 'avg_z': avg_z, 'center_3d': center_3d}

def get_plane_equation(face_vertices_3d):
    p0, p1, p2 = face_vertices_3d[0], face_vertices_3d[1], face_vertices_3d[2]
    v1 = p1 - p0
    v2 = p2 - p0
    
    normal = np.cross(v1, v2)
    norm_len = np.linalg.norm(normal)
    if norm_len < 1e-9: 
       return None 
    normal /= norm_len

    A, B, C = normal
    D = -np.dot(normal, p0)
    return np.array([A, B, C, D])

def point_plane_sign(point_3d, plane_coeffs):
    if plane_coeffs is None:
        return 0 
    return np.dot(plane_coeffs[:3], point_3d) + plane_coeffs[3]

def bboxes_overlap(proj_p, proj_q):
    min_p_x, max_p_x = np.min(proj_p[:, 0]), np.max(proj_p[:, 0])
    min_p_y, max_p_y = np.min(proj_p[:, 1]), np.max(proj_p[:, 1])
    min_q_x, max_q_x = np.min(proj_q[:, 0]), np.max(proj_q[:, 0])
    min_q_y, max_q_y = np.min(proj_q[:, 1]), np.max(proj_q[:, 1])

    overlap_x = (min_p_x <= max_q_x) and (max_p_x >= min_q_x)
    overlap_y = (min_p_y <= max_q_y) and (max_p_y >= min_q_y)

    return overlap_x and overlap_y

def painter_sort(all_faces_properties, observer_pos):
    
    sorted_faces = sorted(all_faces_properties, key=lambda f: f['max_z'], reverse=True)

    n = len(sorted_faces)
    i = 0
    while i < n - 1:
        P = sorted_faces[i]
        Q = sorted_faces[i+1]

        if P['max_z'] < Q['min_z']:
            i += 1
            continue

        proj_p = project(P['vertices_3d'], PERSPECTIVE_DISTANCE)
        proj_q = project(Q['vertices_3d'], PERSPECTIVE_DISTANCE)
        
        if not bboxes_overlap(proj_p, proj_q):
            i += 1
            continue

        plane_p = get_plane_equation(P['vertices_3d'])
        plane_q = get_plane_equation(Q['vertices_3d'])

        if plane_p is None or plane_q is None:
             i += 1
             continue

        sign_obs_p = np.sign(point_plane_sign(observer_pos, plane_p))
        sign_obs_q = np.sign(point_plane_sign(observer_pos, plane_q))

        signs_p_in_q = np.sign([point_plane_sign(v, plane_q) for v in P['vertices_3d']])
        signs_q_in_p = np.sign([point_plane_sign(v, plane_p) for v in Q['vertices_3d']])

        tolerance = 1e-6
        
        all_p_diff_side_q = np.all(np.abs(signs_p_in_q - sign_obs_q) > tolerance) or np.all(signs_p_in_q * sign_obs_q < -tolerance)
        all_q_same_side_p = np.all(np.abs(signs_q_in_p - sign_obs_p) < tolerance) or np.all(signs_q_in_p * sign_obs_p > -tolerance)
  
        test3_passed = all_p_diff_side_q
        test4_passed = all_q_same_side_p
        if test3_passed or test4_passed:
            
            i += 1
            continue

        all_q_diff_side_p = np.all(np.abs(signs_q_in_p - sign_obs_p) > tolerance) or np.all(signs_q_in_p * sign_obs_p < -tolerance) 
        all_p_same_side_q = np.all(np.abs(signs_p_in_q - sign_obs_q) < tolerance) or np.all(signs_p_in_q * sign_obs_q > -tolerance)
        
        if all_q_diff_side_p or all_p_same_side_q:
             
             sorted_faces[i], sorted_faces[i+1] = sorted_faces[i+1], sorted_faces[i]
             
             i = max(0, i - 1) 
             continue 

        
        i += 1

    return sorted_faces

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
angle = 0.0

def update(frame):
    global angle
    angle += ROTATION_SPEED
    ax.clear()

    rotation_func = {
        1: rotation_x,
        2: rotation_y,
        3: rotation_z
    }
    rot_mat = rotation_func[axis_choice](angle)

    all_faces_props = []
    
    for obj in objects:
        rotated_vertices = obj['vertices'] @ rot_mat.T 
        for face_indices in obj['faces']:
            face_verts_3d = rotated_vertices[face_indices]
            props = get_face_properties(face_verts_3d)
            props['color'] = obj['color'] 
            all_faces_props.append(props)

    sorted_faces = painter_sort(all_faces_props, OBSERVER_POS)

    for face_prop in sorted_faces:
        
        projected_verts = project(face_prop['vertices_3d'], PERSPECTIVE_DISTANCE)

        
        polygon = Polygon(projected_verts, closed=True, facecolor=face_prop['color'], edgecolor='black', linewidth=0.5)
        ax.add_patch(polygon)

    ax.set_xlim(-3, 3) 
    ax.set_ylim(-3, 3) 
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f'Обертання тетраедрів (кадр {frame})')
    ax.grid(True, linestyle='--', alpha=0.6)

ani = animation.FuncAnimation(fig, update, frames=range(300), interval=50, repeat=False)
ani.save('tetra_animation.mp4', writer='ffmpeg', fps=20)
os.system('ffmpeg -i tetra_animation.mp4 -vf "fps=20,scale=600:-1:flags=lanczos" -c:v gif tetra_animation.gif')
plt.show()