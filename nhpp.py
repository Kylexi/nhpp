import numpy as np
from math import sqrt

def get_arrivals(knots):
	a = [0] # Arrival times for nonhomogeneous poisson process
	u = [0] # Arrival times for homogeneous poisson process
	n = 0   # 
	j = 0   # Counter to see which 'piece' of the integrated rate function we are in.
	s = []  # Holds the slope between each knot
	L = [0] # Holds values for integrated rate function.

	knot_times = list(knots.keys())
	knot_vals = list(knots.values())

	for i in range(1, len(knot_times)):
		L.append(L[-1] + 
			0.5 * (knot_vals[i] + knot_vals[i-1]) * 
			(knot_times[i] - knot_times[i-1])
			)
		s.append((knot_vals[i] - knot_vals[i-1]) / 
			(knot_times[i] - knot_times[i-1]))

	def inv_int_rate_func(u, j):
		res = 0
		if s[j] != 0:
			res = knot_times[j] + 2 * (u - L[j]) / (
				knot_vals[j] + sqrt(
					knot_vals[j]**2 + 2 * s[j] * (u - L[j])
					)
				)
		else:
			res = knot_times[j] + (u - L[j]) / knot_vals[j]
		return res

	while True:
		u_next = u[-1] + np.random.exponential(1.0)
		if u_next >= L[-1]:
			break
		while L[j+1] < u_next and j < len(knot_times):
			j += 1
		a_next = inv_int_rate_func(u_next, j)
		a.append(a_next)
		u.append(u_next)
		n += 1
	return a
