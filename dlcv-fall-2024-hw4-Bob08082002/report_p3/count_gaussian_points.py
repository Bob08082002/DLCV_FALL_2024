from plyfile import PlyData
import sys

""" 
    Count number of 3D gaussian points: 
    python3 count_gaussian_points.py <path to points3D.ply file> 
    
"""

def count_gaussian_points(ply_path):
    # Read the PLY file
    ply_data = PlyData.read(ply_path)

    # Get the 'vertex' element and count the number of points
    num_points = len(ply_data['vertex'])
    return num_points


points3D_ply_path = sys.argv[1] # ex: "../report_p4/SfM_init_points/points3D.ply"
num_gaussian_points = count_gaussian_points(points3D_ply_path)
print(f"The file {points3D_ply_path} contains {num_gaussian_points} Gaussian points.")