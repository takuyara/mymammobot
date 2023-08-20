import numpy as np
import pyvista as pv
from tqdm import tqdm
import matplotlib.pyplot as plt
from functools import cmp_to_key

from pose_fixing.similarity import comb_corr_sim, reg_mse_sim, cache_base_data, contour_sim, draw_contours
from pose_fixing.move_camera import random_move, uniform_sampling, move_by_params, reverse_to_geno
from ds_gen.depth_map_generation import get_depth_map

class EvolutionStrategy:
	def __init__(self, mesh_path, img_size, real_depth_map,
		num_parents, num_offsprings, num_generations,
		axial_scale, radial_scale, orientation_scale, rot_scale,
		learning_rate = 0.4, fitness_func = "contour_corr", fitness_kwargs = {},
		parent_selection = "best", converge_tolerance = None,
		explore_distrib = "normal", contour_tolerance = 5, to_norm_scale = 0.3,
		):
		surface = pv.read(mesh_path)
		self.plotter = pv.Plotter(off_screen = True, window_size = (img_size, img_size))
		self.plotter.add_mesh(surface)

		self.plotter_debug = pv.Plotter()
		self.plotter_debug.add_mesh(surface, opacity = 0.5)

		self.real_depth_map = real_depth_map
		self.learning_rate = learning_rate
		self.orientation_scale = orientation_scale
		self.axial_scale = axial_scale
		self.radial_scale = radial_scale
		self.rot_scale = rot_scale
		self.num_parents = num_parents
		self.num_offsprings = num_offsprings
		self.num_generations = num_generations
		self.tol = converge_tolerance
		self.explore_distrib = explore_distrib
		self.to_norm_scale = to_norm_scale
		self.contour_cache = cache_base_data(real_depth_map)
		# Genotype: position, orientation, up, up_rot, sigma_base, value
		self.population = []
		self.contour_tolerance = contour_tolerance
		self.fitness_compare = cmp_to_key(self.get_fitness_compare())
		self.base_position = None
		self.base_orientation = None

		"""
		if fitness_func == "weighted_corr":
			self.fitness_func = comb_corr_sim
		elif fitness_func == "reg_mse":
			self.fitness_func = reg_mse_sim
		else:
			raise NotImplementedError

		if parent_selection == "biased_roulette":
			self.parent_selection = self.biased_roulette_selection
		elif parent_selection == "best":
			self.parent_selection = self.best_selection
		else:
			raise NotImplementedError
		"""

		assert fitness_func == "contour_corr" and parent_selection == "best"

	def fitness_func(self, img):
		c1, c2 = contour_sim(img, *self.contour_cache), comb_corr_sim(self.real_depth_map, img)
		if c1 is None or c2 is None:
			return None
		else:
			return c1, c2

	def get_fitness_compare(self):
		def compare(pheno_1, pheno_2):
			fitness_1, fitness_2 = pheno_1[1][-1], pheno_2[1][-1]
			if abs(fitness_1[0] - fitness_2[0]) <= self.contour_tolerance:
				if fitness_1[1] != fitness_2[1]:
					return -1 if fitness_1[1] < fitness_2[1] else 1
				return 0
			else:
				return -1 if fitness_1[0] < fitness_2[0] else 1
		return compare

	def iterate_sigma(self, sigma):
		return sigma * np.exp(self.learning_rate * np.random.randn())

	def get_phenotype(self, geno):
		# Genotype: orient_rot, orient_norm, radial_rot, radial_norm, axial_norm, up_rot, sigma
		# Sigma not passed to move cameras.
		position, orientation, up = move_by_params(self.base_position, self.base_orientation, *geno[ : -1])
		virtual_depth_map = get_depth_map(self.plotter, position, orientation, up)
		"""
		plt.subplot(1, 2, 1)
		plt.imshow(virtual_depth_map, cmap = "gray")
		plt.subplot(1, 2, 2)
		plt.imshow(draw_contours(virtual_depth_map, *self.contour_cache))
		plt.show()
		"""
		if np.any(np.isnan(virtual_depth_map)):
			return None
		fitness = self.fitness_func(virtual_depth_map)
		"""
		if fitness[0] < -1e5:
			plt.subplot(1, 2, 1)
			plt.imshow(virtual_depth_map, cmap = "gray")
			plt.subplot(1, 2, 2)
			plt.imshow(draw_contours(virtual_depth_map, *self.contour_cache))
			plt.show()
			print(f"Quantile: {self.contour_cache[1]}. Value: {np.quantile(virtual_depth_map.ravel(), self.contour_cache[1])}")
		try:
			pass
		except ValueError as e:
			return None
		"""

		#assert abs(fitness - comb_corr_sim(self.real_depth_map, virtual_depth_map)) < 1e-3
		if fitness is None:
			return None

		return (position, orientation, up, fitness)

	def init_population(self, base_position, base_orientation, norm_samples, rot_samples, up_rot_samples):
		self.population = []
		self.base_position = base_position
		self.base_orientation = base_orientation

		"""
		new_position = np.array([-35.23436981, -26.07758012, -161.09738046])
		new_orientation = np.array([-0.86110491, -0.40747873, -0.30407139])
		new_up = np.array([0.50811759, -0.71058364, -0.4867108])
		geno = reverse_to_geno(base_position, base_orientation, new_position, new_orientation, new_up)
		p1, p2, p3 = move_by_params(base_position, base_orientation, *geno)
		print(self.get_phenotype((*geno, 1))[-1])

		exit()
		"""
		#print(np.linalg.norm(new_position - p1), np.linalg.norm(new_orientation - p2), np.linalg.norm(new_up - p3))

		# Optimal phenotype: (20, 0.15, -150, 1.18, -0.98, -2)

		sample_genos = uniform_sampling(self.rot_scale, self.axial_scale, self.radial_scale, self.orientation_scale, norm_samples, rot_samples, up_rot_samples)
		#print("Initial population: ", len(sample_genos))
		#print("Base position before: ", self.base_position)
		for geno in sample_genos:
			geno = (*geno, 1)
			pheno = self.get_phenotype(geno)
			if pheno is not None:
				self.population.append((geno, pheno))
		#print("Survived population: ", len(self.population))
		#print("Base position after: ", self.base_position)
	"""
	def biased_roulette_selection(self, num_samples):
		weights = np.exp([pop[-1] for pop in self.population])
		weights /= np.sum(weights)
		indices = np.random.choice(len(self.population), num_samples, replace = False, p = weights)
		return [self.population[i] for i in indices]
	"""

	def best_selection(self, num_samples):
		self.population.sort(key = self.fitness_compare, reverse = True)
		return self.population[ : num_samples]

	def show_sigma_samples(self, generation_samples = 500):
		prev_sigmas = [1]
		all_sigmas = [prev_sigmas]
		for i in range(self.num_generations):
			this_sigmas = []
			for j in prev_sigmas:
				this_sigmas.append(self.iterate_sigma(j))
			indices = np.random.choice(len(this_sigmas), generation_samples)
			prev_sigmas = [this_sigmas[j] for j in indices]
			all_sigmas.append(prev_sigmas)
		point_xs, point_ys = [], []
		for i in range(len(all_sigmas)):
			for j in all_sigmas[i]:
				point_xs.append(i)
				point_ys.append(j)
		plt.scatter(point_xs, point_ys)
		plt.show()

	def reduce_scale_for_norm(self):
		self.axial_scale *= self.to_norm_scale
		self.radial_scale *= self.to_norm_scale
		self.orientation_scale *= self.to_norm_scale
		self.rot_scale *= self.to_norm_scale

	def pertub_geno(self, geno):
		sigma = self.iterate_sigma(geno[-1])
		return (*random_move(*geno[ : -1], sigma * self.rot_scale, sigma * self.axial_scale, sigma * self.radial_scale, sigma * self.orientation_scale), sigma)

	def average_population_fitness(self):
		return sum([pop[1][-1][0] for pop in self.population]) / len(self.population)

	def run(self, return_history = False):
		if self.explore_distrib == "normal":
			self.reduce_scale_for_norm()
		global_optim = max(self.population, key = self.fitness_compare)
		avg_contour_history = [self.average_population_fitness()]

		"""
		for geno, pheno in self.population:
			if geno[-2] == 0:
				position, orientation, up = move_by_params(self.base_position, self.base_orientation, *geno[ : -1])
				print(np.linalg.norm(self.base_position - position))
				self.plotter_debug.add_mesh(pv.Arrow(position, orientation), color = "red")
		self.plotter_debug.show()
		"""


		for i in range(self.num_generations):
			offsprings = []
			parents = self.best_selection(self.num_parents)
			sum_sigmas = 0
			total_trys = 0
			sum_optims = 0

			"""
			this_geno = (0, 0, 0, 0, 0, 0, 1)
			for k in range(50):
				new_geno = self.pertub_geno(this_geno)
				position, orientation, up = move_by_params(self.base_position, self.base_orientation, *new_geno[ : -1])
				self.plotter_debug.add_mesh(pv.Arrow(position, orientation), color = "red")
				#self.plotter_debug.add_mesh(pv.Arrow(position, up), color = "green")
			self.plotter_debug.show()
			exit()
			"""


			while len(offsprings) < self.num_offsprings:
				this_parent = parents[np.random.randint(len(parents))]

				#this_parent = ((0, 0, 0, 0, 0, 0, 1), "non-specific")
				new_geno = self.pertub_geno(this_parent[0])
				#print(new_geno)
				new_pheno = self.get_phenotype(new_geno)
				total_trys += 1

				if new_pheno is not None:
					offsprings.append((new_geno, new_pheno))
					sum_sigmas += new_geno[-1]
					sum_optims += new_pheno[-1][0]

				#print(f"{len(offsprings)}/{self.num_offsprings}: {len(offsprings) / total_trys:.4f}")
				"""
				prev_position, prev_orientation, prev_up, prev_up_rot, sigma, __ = this_parent
				this_sigma = self.iterate_sigma(sigma)
				this_position, this_orientation, this_up, this_up_rot = random_move(
					self.axial_scale * this_sigma, self.radial_scale * this_sigma, prev_position,
					self.orientation_scale * this_sigma, prev_orientation,
					self.up_rot_scale * this_sigma, prev_up_rot, distribution = self.explore_distrib,
				)
				this_offspring = self.get_phenotype(this_position, this_orientation, this_up, this_up_rot, this_sigma)
				if this_offspring is not None:
					offsprings.append(this_offspring)
					sum_sigmas += this_sigma
				"""
			
			this_optim = max(offsprings, key = self.fitness_compare)
			#print("Iter: {} sigma = {:.4f} optimal_contour = {:.4f} mean_contour = {:.4f}".format(i, sum_sigmas / self.num_offsprings, this_optim[1][-1][0], sum_optims / self.num_offsprings))
			avg_contour_history.append(sum_optims / self.num_offsprings)
			if self.get_fitness_compare()(this_optim, global_optim) > 0:
				global_optim = this_optim
			self.population = parents + offsprings

		rgb, dep = get_depth_map(self.plotter, *global_optim[1][ : 3], get_outputs = True)

		if return_history:
			return global_optim[1], rgb, dep, avg_contour_history
		else:
			return global_optim[1], rgb, dep
