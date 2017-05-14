#!/usr/loca/bin/python3
import csv
import sys
import math
import numpy

def elu(arr):
	return numpy.where(arr > 0, arr, numpy.exp(arr) - 1)

def make_layer(in_size, out_size):
	w = numpy.random.normal(scale=0.5, size=(in_size, out_size))
	b = numpy.random.normal(scale=0.5, size=out_size)
	return (w, b)

def forward(inpd, layers):
	out = inpd
	for layer in layers:
		w, b = layer
		out = elu(out @ w + b)
	return out

def gen_data(dim, layer_dims, N):
	layers = []
	data = numpy.random.normal(size=(N, dim))

	nd = dim
	for d in layer_dims:
		layers.append(make_layer(nd, d))
		nd = d
	w, b = make_layer(nd, nd)
	gen_data = forward(data, layers)
	gen_data = gen_data @ w + b
	return gen_data

def train():
	n = 20
	avgs = []
	layer_dims = [numpy.random.randint(60, 80), 100]
	for dim in range(1, 61):
		temp_avgs = []
		for r in range(n):
			data = gen_data(dim, layer_dims, numpy.random.randint(10000, 100000))
			temp_avgs.append(numpy.average(numpy.std(data, axis = 0)))
			print(str(dim) + ': ' + str(temp_avgs[-1]))
		avgs.append(numpy.average(temp_avgs))
	with open('model', 'w') as f:
		for i in range(60):
			f.write('%f\n' % round(avgs[i], 3))

def main():
	# train()
	avgs = []
	answers = []
	data = numpy.load(sys.argv[1])
	with open('model', errors = 'ignore') as f:
		for avg in csv.reader(f):
			avgs.append(float(avg[0]))
	
	for i in range(200):
		ans = 0
		min_diff = sys.maxsize
		avg = numpy.average(numpy.std(data[str(i)], axis = 0))
		for j in range(60):
			diff = abs(avg - avgs[j])
			if min_diff > diff:
				ans = j + 1
				min_diff = diff
		answers.append(ans)

	with open(sys.argv[2], 'w') as f:
		f.write('SetId,LogDim\n')
		for i in range(200):
			f.write('%d,%f\n' % (i, math.log(answers[i])))

if __name__ == '__main__':
	main()