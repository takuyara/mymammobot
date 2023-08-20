import numpy as np
import pyvista as pv
from ds_gen.camera_features import camera_params

def get_depth_map(
	p, position, orientation, up, get_outputs = False,
	zoom = 1.0, focal_length = camera_params["focal_length"],
	view_angle = camera_params["view_angle"], clipping_range = camera_params["clipping_range"]
	):
	camera = pv.Camera()
	camera.position = position
	camera.focal_point = position + focal_length * orientation
	camera.up = up
	camera.view_angle = view_angle
	camera.clipping_range = clipping_range
	camera.zoom(zoom)
	p.camera = camera
	p.show(auto_close = False)
	if get_outputs:
		return p.screenshot(None, return_img = True), -p.get_image_depth()
	else:
		return -p.get_image_depth()

def get_zoomed_plotter(img_size, zoom_scale, orig_view_angle = camera_params["view_angle"]):
	half_angle = np.deg2rad(orig_view_angle / 2)
	size_change_rate = np.tan(half_angle / zoom_scale) / np.tan(half_angle)
	zoomed_size = int(img_size * size_change_rate)
	p1 = pv.Plotter(off_screen = True, window_size = (zoomed_size, zoomed_size))
	return p1

