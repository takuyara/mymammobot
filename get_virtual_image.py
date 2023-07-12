import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

focal_length = 5

mesh_path = "./meshes/Airway_Phantom_AdjustSmooth.stl"
em_path = "E:\\nn-data\\MAMMOBOT-Original\\Dataset_20151113\\Dataset_20151113\\EM_Video_sequences\\logfile_2015-11-13_16-04-19_Phantom_1\\EM\\10.txt"
em_next_path = "E:\\nn-data\\MAMMOBOT-Original\\Dataset_20151113\\Dataset_20151113\\EM_Video_sequences\\logfile_2015-11-13_16-04-19_Phantom_1\\EM\\21.txt"
general_pose = np.loadtxt(em_path).reshape(-1)
position = general_pose[ : 3]
#position1 = np.loadtxt(em_next_path).reshape(-1)[ : 3]
#r1 = (position1 - position) / np.linalg.norm((position1 - position))
position = np.array([105.31233978, -5.09855175, -144.31781006])
rotation = R.from_quat(general_pose[3 : ]).apply(np.array([-0.31512598, -0.7056523, 0.63462622]))
#print(rotation)
rotation = np.array([0.97168467, 0.23531002, 0.02140322])
#rotation = -r1
#rotation = general_pose[-3 : ]
#rotation = R.from_quat(general_pose[3 : ]).as_matrix()
#focal_point = np.linalg.inv(rotation) @ (np.array([[0, 0, focal_length]]).T - position.reshape(3, 1))
#print(focal_point)
#print(rotation, position)
#rotation = rotation / np.linalg.norm(rotation)
#position1 = np.loadtxt(em_next_path).reshape(-1)[ : 3]
#r1 = (position1 - position) / np.linalg.norm((position1 - position))
#print(rotation, r1)
#rotation = r1
#camera_matrix = np.concatenate([rotation, position.reshape(3, 1)], axis = -1)
#camera_matrix = np.concatenate([camera_matrix, np.array([[0, 0, 0, 1]])], axis = 0)
#print(camera_matrix)

surface = pv.read(mesh_path)
p = pv.Plotter(off_screen = False)
p.add_mesh(surface)
camera = pv.Camera()

"""
near_range = 0.1
far_range = 10
camera.clipping_range = (near_range, far_range)
camera.focal_point, camera.direction, camera.position
"""
#camera.direction = rotation

"""
position = np.array([-99.24748992919922, 69.00006103515625, -143.79388427734375])
position1 = np.array([-98.06033325195312, 67.21239471435547, -143.7113800048828])
rotation = position1 - position
rotation = rotation / np.linalg.norm(rotation)
"""


camera.position = position
print(camera.position)

camera.focal_point = focal_length * rotation + position
#camera.model_transform_matrix = camera_matrix
#camera.focal_point = focal_point

p.camera = camera
print(p.camera.focal_point)
p.show()
print(p.camera.focal_point)
"""
img = -p.get_image_depth()
print(img.shape)
plt.imshow(img, cmap = "coolwarm")
plt.colorbar()
plt.show()
"""