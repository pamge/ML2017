#!/usr/loca/bin/python3
import csv
import sys
import math
import numpy

def main():
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