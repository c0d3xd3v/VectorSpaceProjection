import numpy as np

def transform_vector(v, normal, matrix):
    """
    Decomposes a vector into its normal and tangential components,
    transforms the normal component with the inverse transpose of the matrix (M^-T),
    and the tangential component directly with the matrix (M), and then combines the two components.

    Parameters:
        v (np.array): The vector to be transformed.
        normal (np.array): The normal of the plane onto which the vector is projected.
        matrix (np.array): The transformation matrix.

    Returns:
        transformed_vector (np.array): The transformed vector.
    """
    normal = normal / np.linalg.norm(normal)

    # decomposition normal and tangential part
    normal_component = np.dot(v, normal) * normal
    tangential_component = v - normal_component

    # transform normal component (mit M^-T)
    matrix_inv_T = np.linalg.inv(matrix).T  # M^-T
    transformed_normal_component = matrix_inv_T @ normal_component

    # transform tangetial component (direct with M)
    transformed_tangential_component = matrix @ tangential_component

    # recombination
    transformed_vector = transformed_normal_component + transformed_tangential_component

    return transformed_vector


# Beispielvektor und Normale
v = np.array([1,-1,0.75])
v = v/np.linalg.norm(v)
normal = np.array([0.0, 0.0, 1.0])

# Beispielmatrix
matrix = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [-1.0, 0.0, 1.0]
])
Det = np.linalg.det(matrix)
matrix = matrix / Det

matrix_inv_T = np.linalg.inv(matrix).T
print(Det)

# Transformieren des Vektors
naiv_v = matrix @ v
naiv_length = np.linalg.norm(naiv_v)
correct_v = transform_vector(v, normal, matrix)
correct_length = np.linalg.norm(correct_v)

print("correct transformed vector: ", correct_v/correct_length, correct_length)
print("naiv transformed vector   : ", naiv_v/naiv_length, naiv_length)

dot_product = np.dot(correct_v/correct_length, naiv_v/naiv_length)
angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
angle_deg = np.degrees(angle_rad)
print(f"Winkel : {angle_deg:.2f} Grad")


import vtk_

O = [1.5, 0.0, 0.0]
origv = vtk_.create_arrow_actor(O, O + v, [0.75, 0.75, 1])
naivN = vtk_.create_arrow_actor([0.0, 0.0, 0.0], naiv_v/naiv_length, [1, 0.75, 0.75])
correctN = vtk_.create_arrow_actor([0.0, 0.0, 0.0], correct_v/correct_length, [0.75, 1, 0.75])

plane_actor1 = vtk_.create_plane_actor(normal,O, [0.85, 0.85, 1.0])
plane_actor2 = vtk_.create_plane_actor(matrix_inv_T @ normal,[0, 0, 0], [0.7, 0.85, 0.85])

actors = [plane_actor1, origv, plane_actor2, naivN, correctN]
vtk_.render_actors(actors)

