#!/usr/local/bin/python3
import sys
import numpy
import scipy
from PIL import Image
from scipy import misc

def load_data(folder_name, n):
    images = []
    for i in range(ord('A'), ord('J') + 1):
        for j in range(n):
            image = numpy.resize(misc.imread('%s/%c%02d.bmp' % (folder_name, chr(i), j), flatten = 1), 64 * 64)
            images.append(image)
    return numpy.array(images)

def get_averages(images):
    return numpy.mean(images, axis = 0)

def get_eigenvectors(images, averages):
    images = images - averages
    eigen_values, eigen_vectors = numpy.linalg.eig(numpy.dot(images.T, images))
    sorted_index = numpy.argsort(eigen_values)[::-1]
    return eigen_vectors[:, sorted_index]

def generate_average_face(path, averages):
    average_face = numpy.resize(averages, (64, 64))
    scipy.misc.imsave(path, numpy.resize(averages, (64, 64)))

def generate_eigenfaces(n, name, averages, eigenvectors):
    for i in range(n):
        eigenfaces = 255 * eigenvectors[:, i]
        out_images.append(numpy.resize(eigenfaces, (64, 64)))
        scipy.misc.imsave(name % i, numpy.resize(eigenfaces, (64, 64)))

def generate_projections(n, m, name1, name2, images, averages, eigenvectors):
    for i in range(ord('J') - ord('A') + 1):
        for j in range(n):
            projections = []
            for k in range(m):
                index = i * 10 + j
                length = numpy.dot(images[index, :].T - averages, eigenvectors[:, k])
                projection = length * eigenvectors[:, k]
                # scipy.misc.imsave(name1 % (chr(ord('A') + i), j, k), numpy.resize(projection, (64, 64)))
                projections.append(projection)
            projections = numpy.array(projections)
            projections_sum = numpy.sum(projections, axis = 0) * 1
            projections_sum = projections_sum + averages
            scipy.misc.imsave(name2 % (chr(ord('A') + i), j), numpy.resize(projections_sum, (64, 64)))

def main():
    images = load_data('faceExpressionDatabase', 10)
    averages = get_averages(images)
    eigenvectors = get_eigenvectors(images, averages)
    numpy.save('eigenvectors', eigenvectors)
    # eigenvectors = numpy.load('eigenvectors.npy')
    generate_average_face('Q1/1_average.bmp', averages)
    generate_eigenfaces(9, 'Q1/1_eigenface_%2d.bmp', averages, eigenvectors)
    generate_projections(10, 5, 'Q1/2_projection_%c%02d_%d.bmp', 'Q1/2_reconstruct_%c%02d.bmp', images, averages, eigenvectors)
    
    # combine anwers
    # grid_offset = 5
    # out_image = Image.new('RGB', (64 * 3 + grid_offset * (3 + 1), 64 * 3 + grid_offset * (3 + 1)), 'white')
    # for i in range(9):
    #     im = Image.open('Q1/1_eigenface_%2d.bmp' % i)
    #     r, c = i // 3, i % 3
    #     out_image.paste(im, ((c + 1) * grid_offset + c * 64, (r + 1) * grid_offset + r * 64))
    # out_image.save('1-1.eigenface.bmp')

    # combine anwers
    # out_image = Image.new('RGB', (64 * 3 + grid_offset * (3 + 1), 64 * 3 + grid_offset * (3 + 1)), 'white')
    # for i in range(9):
    #     im = Image.open('Q1/1_eigenface_%2d.bmp' % i)
    #     r, c = i // 3, i % 3
    #     out_image.paste(im, ((c + 1) * grid_offset + c * 64, (r + 1) * grid_offset + r * 64))
    # out_image.save('1-1.eigenface.bmp')

    # combine anwers
    # out_image1 = Image.new('RGB', (64 * 10 + grid_offset * (10 + 1), 64 * 10 + grid_offset * (10 + 1)), 'white')
    # out_image2 = Image.new('RGB', (64 * 10 + grid_offset * (10 + 1), 64 * 10 + grid_offset * (10 + 1)), 'white')
    # for i in range(100):
    #     im1 = Image.open('%s/%c%02d.bmp' % ('faceExpressionDatabase', chr(i // 10 + ord('A')), i % 10))
    #     im2 = Image.open('Q1/2_reconstruct_%c%02d.bmp' % (chr(i // 10 + ord('A')), i % 10))
    #     r, c = i // 10, i % 10
    #     out_image1.paste(im1, ((c + 1) * grid_offset + c * 64, (r + 1) * grid_offset + r * 64))
    #     out_image2.paste(im2, ((c + 1) * grid_offset + c * 64, (r + 1) * grid_offset + r * 64))
    # out_image1.save('1-2.original.bmp')
    # out_image2.save('1-2.reconstruct.bmp')
        
if __name__ == '__main__':
    main()
    