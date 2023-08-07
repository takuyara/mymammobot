dir_list = [
	"E:\\nn-data\\MAMMOBOT-Original\\Dataset_20151113\\Dataset_20151113\\EM_Video_sequences\\logfile_2015-11-13_14-53-18_Tasos\\EM",
	"E:\\nn-data\\MAMMOBOT-Original\\Dataset_20151113\\Dataset_20151113\\EM_Video_sequences\\logfile_2015-11-13_15-03-49_Sam\\EM",
	"E:\\nn-data\\MAMMOBOT-Original\\Dataset_20151113\\Dataset_20151113\\EM_Video_sequences\\logfile_2015-11-13_15-09-47_Sam\\EM",
	"E:\\nn-data\\MAMMOBOT-Original\\Dataset_20151113\\Dataset_20151113\\EM_Video_sequences\\logfile_2015-11-13_16-04-19_Phantom_1\\EM",
	"E:\\nn-data\\MAMMOBOT-Original\\Dataset_20151113\\Dataset_20151113\\EM_Video_sequences\\logfile_2015-11-13_17-08-08_Phantom_2\\EM",
	"E:\\nn-data\\MAMMOBOT-Original\\Dataset_20151113\\Dataset_20151113\\EM_Video_sequences\\logfile_2015-11-13_17-12-48_Phantom_3\\EM",
	]

import numpy as np
import os
import matplotlib.pyplot as plt

titles = ["x_trans", "y_trans", "z_trans", "q_1", "q_2", "q_3", "q_4"]


all_velocities = []

for this_dir in dir_list:
	this_velocities = []
	for i in range(len(os.listdir(this_dir))):
		with open(os.path.join(this_dir, f"{i}.txt")) as f:
			this_num = np.array([float(x) for x in f.read().split()])
		if i > 0:
			all_velocities.append(this_num - prev_num)
			this_velocities.append(this_num - prev_num)
		prev_num = this_num
	this_velocities = np.stack(this_velocities, axis = 0)
	for i in range(7):
		plt.plot(this_velocities[ : , i])
		plt.title(titles[i])
		plt.xlabel("Frame")
		plt.ylabel("Velocity")
		plt.show()

all_velocities = np.stack(all_velocities, axis = 0)
for i in range(7):
	np.save(f"{titles[i]}.npy", all_velocities[ : , i])
	print(np.mean(all_velocities[ : , i]), np.std(all_velocities[ : , i]))

"""
for i in range(7):
	plt.subplot(2, 4, i + 1)
	plt.hist(all_velocities[ : , i], bins = 500)
	plt.xlabel("Velocity value")
	plt.ylabel("Density")
	plt.title(titles[i])
plt.show()
"""
