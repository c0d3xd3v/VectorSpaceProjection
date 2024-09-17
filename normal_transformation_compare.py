import numpy as np

def compute_normal(v1, v2):
    v1_n = v1/np.linalg.norm(v1)
    v2_n = v2/np.linalg.norm(v2)

    normal = np.cross(v1_n, v2_n)

    normal_length = np.linalg.norm(normal)
    if normal_length != 0:
        normal /= normal_length

    return normal


# Vektoren und Matrix definieren
v1 = np.array([-1.0, 5.0, -1.5])  # Erster Vektor des ersten Satzes
v2 = np.array([2.5, -1.0, 0.5])  # Zweiter Vektor des ersten Satzes

matrix = np.array([
    [3.0, 1.0, 1.0],
    [1.0, 5.0, 3.0],
    [2.0, 1.0, 2.0]
])
D = np.linalg.det(matrix)
print(f'Det(M): {D}')
matrix = matrix/D

# Berechnen der transformierten Vektoren
v1_transformed = matrix @ v1
v2_transformed = matrix @ v2

normal1 = compute_normal(v1, v2)
normal2 = compute_normal(v1_transformed, v2_transformed)

naiv_normal_transformed = matrix @ normal1
naiv_length = np.linalg.norm(naiv_normal_transformed)

correct_normal_transformed = np.transpose(np.linalg.inv(matrix)) @ normal1
correct_normal_transformed = correct_normal_transformed/np.linalg.norm(correct_normal_transformed)
correct_length = np.linalg.norm(correct_normal_transformed)

# Berechne den Winkel zwischen den Normalen
dot_product = np.dot(normal1, normal2)
angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
angle_deg = np.degrees(angle_rad)
print(f"Winkel zwischen den Ebenen: {angle_deg:.2f} Grad")

# Berechnen des Winkels zwischen den normalen
dot_product = np.dot(naiv_normal_transformed, normal2)
angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
angle_deg = np.degrees(angle_rad)
print(f"Winkel zwischen den normalen (naiv): {angle_deg:.2f} Grad")

dot_product = np.dot(correct_normal_transformed, normal2)
angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
angle_deg = np.degrees(angle_rad)
print(f"Winkel zwischen den normalen (korregiert): {angle_deg:.2f} Grad")


import vtk_
naivN = vtk_.create_arrow_actor([0.0, 0.0, 0.0], naiv_normal_transformed/(1.0*naiv_length), [1, 0.75, 0.75])
correctN = vtk_.create_arrow_actor([0.0, 0.0, 0.0], correct_normal_transformed/(1.0*correct_length), [0.75, 1, 0.75])
plane1_actor = vtk_.create_plane_actor(normal1,[0, 0, 0], [0.7, 0.75, 0.75])
plane2_actor = vtk_.create_plane_actor(normal2,[0, 0, 0], [0.75, 0.75, 0.7])

actors = [naivN, correctN, plane2_actor]
vtk_.render_actors(actors)

