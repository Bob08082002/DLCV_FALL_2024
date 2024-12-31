import random
from plyfile import PlyData

def create_random_points3D(file_path, num_points=15000):
    """ create random points3D.txt using COLMAP format"""
    with open(file_path, 'w') as file:
        # write the header
        file.write("# 3D point list with the following format:\n")
        file.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")

        for point_id in range(1, num_points + 1):
            # random (X, Y, Z)
            x, y, z = [random.uniform(-70.0, 120.0) for _ in range(3)]

            # random (R, G, B): close to each other
            r = random.randint(5, 250)
            g = r + random.randint(-5, 5)
            b = r + random.randint(-5, 5)
            # Ensure RGB values are within valid bounds (0-255)
            r, g, b = [max(0, min(255, v)) for v in (r, g, b)]

            error = 0.0

            # Write the point data to the file
            file.write(f"{point_id} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {error:.1f}\n")


random_points3D_txt_path = "./random_init_points/points3D.txt"
create_random_points3D(random_points3D_txt_path)
