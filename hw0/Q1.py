#!/usr/bin/python
import sys
import numpy

def main(argv):
	assert(len(argv) == 3)
	matrixA = []
	matrixB = []
	# matrixA
	for line in open(sys.argv[1]):
		matrixA.append(line.strip('\n').split(','))
	# matrixB
	for line in open(sys.argv[2]):
		matrixB.append(line.strip('\n').split(','))
	# multiplication
	matrixA = numpy.matrix(matrixA, dtype=int)
	matrixB = numpy.matrix(matrixB, dtype=int)
	matrixC = [str(value) for value in sorted((matrixA * matrixB).reshape(-1).tolist()[0])]
	# print data
	print '\n'.join(matrixC)
	
if __name__ == '__main__':
	main(sys.argv)
