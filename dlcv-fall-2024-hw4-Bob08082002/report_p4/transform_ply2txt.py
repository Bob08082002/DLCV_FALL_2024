from plyfile import PlyData

def convert_plyfile_to_txt(ply_path, txt_path):
    plydata = PlyData.read(ply_path)
    vertices = plydata['vertex']

    with open(txt_path, 'w') as txt_file:
        txt_file.write("# 3D point list with the following format:\n")
        txt_file.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        
        for i, vertex in enumerate(vertices):
            point_id = i + 1
            x, y, z = vertex['x'], vertex['y'], vertex['z']
            r, g, b = vertex['red'], vertex['green'], vertex['blue']
            txt_file.write(f"{point_id} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} 0.0\n")


points3D_ply_path = "../dataset/train/sparse/0/points3D.ply"
points3D_txt_path = "./SfM_init_points/points3D.txt"
convert_plyfile_to_txt(points3D_ply_path, points3D_txt_path)