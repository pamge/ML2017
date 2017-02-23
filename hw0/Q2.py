#!/usr/bin/python
import sys
import numpy
from PIL import Image

def main(argv):
	assert(len(argv) == 3)
	lena = Image.open(argv[1])
	lenaModified = Image.open(argv[2])
	lenaPixel = lena.load()
	lenaModifiedPixel = lenaModified.load()
	im = Image.new('RGBA', lena.size)
	imPixel = im.load()
	for i in range(lena.size[0]):
		for j in range(lena.size[1]):
			if lenaPixel[i, j] == lenaModifiedPixel[i, j]:
				imPixel[i, j] = (0, 0, 0, 0)
			else:
				imPixel[i, j] = lenaModifiedPixel[i, j]
	im.save('ans_two.png')
	
if __name__ == '__main__':
	main(sys.argv)
