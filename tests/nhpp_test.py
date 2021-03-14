import nhpp
import math
import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize("test_input,expected", [
	({0: 1, 2: 1, 1: 0}, ([0, 1, 2], [1, 0, 1])),
	({0: 1, 3: 1, 2: 2}, ([0, 2, 3], [1, 2, 1])),
	])
def test_sorting(test_input, expected):
	assert nhpp.nhpp._get_sorted_pairs(test_input) == expected


@pytest.mark.parametrize("test_input,expected", [
	(0, 0),
	(1, 5),
	(0.5, 2.5),
	(3.5, 2.5)
	])
def test_piecewise_interp(test_input, expected):
	knot_times  = [0, 1, 2, 3, 5]
	knot_vals   = [0, 5, 1, 2, 4]
	knots = dict(zip(knot_times, knot_vals))
	assert nhpp.nhpp._get_piecewise_val(knots, test_input) == expected


def test_non_dict_error_catch():
	knots = [0, 1, 2]
	with pytest.raises(TypeError):
		nhpp.get_arrivals(knots)


def test_negative_rate_error_catch():
	knots = {0: 0, 1: -1, 2: 2, 3: 0}
	with pytest.raises(ValueError):
		nhpp.get_arrivals(knots)


def test_rate_slopes_error_catch():
	knot_times  = [0, 1, 2, 3, 4]
	knot_vals = [0, 0, 1, 2, 3]
	with pytest.raises(ValueError):
		nhpp.nhpp._get_rate_slopes(knot_times, knot_vals)


def get_epsilon(knots, bins, func=None, *args, **kwargs):
	knots = {0: 1, 1: 0, 2: 2}
	bins = 10
	data = []
	max_knot_val = max(knots.values())
	min_knot_dom = min(knots.keys())
	max_knot_dom = max(knots.keys())
	for i in range(100000):
		arrivals = nhpp.get_arrivals(knots)
		data.append(np.histogram(arrivals, bins, (min_knot_dom, max_knot_dom))[0])
	_, bin_measure = np.histogram(arrivals, bins, (min_knot_dom, max_knot_dom))
	df = pd.DataFrame(data)
	if func:
		check_against = [func(measure, *args, **kwargs) for measure in np.linspace(min_knot_dom, max_knot_dom, bins)]
	else:
		check_against = [nhpp.nhpp.	_get_piecewise_val(knots, measure) for measure in np.linspace(0, 2, bins)]
	max_val = max(check_against)
	check = max_val*df.sum()/df.sum().max()
	check, check_against = np.array(check), np.array(check_against)
	return np.sum(check - check_against)


def test_eps_no_func_1():
	knots = {0: 1, 1: 0, 2: 2}
	bins = 10
	assert(get_epsilon(knots, bins) < 1)


def test_eps_with_func_1():
	knots = {0: 3, math.pi/2: 9, math.pi: 3, 3*math.pi/2: 0, 2*math.pi: 3}
	bins = 10
	def test_func(t):
		return 3*np.sin(t) + 3
	assert(get_epsilon(knots, bins, test_func) < 1)


def test_eps_with_func_2():
	knots = {0: 0, 2.5: 8, 5: 0}
	bins = 10
	def test_func(t):
		return t*(5-t)
	assert(get_epsilon(knots, bins, test_func) < 1)


def test_non_dominating_piecewise():
	knots = {0: 0, 2.5: 6.25, 5: 0}
	bins = 10
	def test_func(t):
		return t*(5-t)
	with pytest.raises(ValueError):
		nhpp.get_arrivals(knots, test_func)
