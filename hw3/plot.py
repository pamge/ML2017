import os
import argparse
from keras.models import load_model
from termcolor import colored,cprint
import keras.backend as K
from utils import *
import numpy as np
import matplotlib.pyplot as plt

import time

base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
img_dir = os.path.join(base_dir, 'image')
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
cmap_dir = os.path.join(img_dir, 'cmap')
if not os.path.exists(cmap_dir):
    os.makedirs(cmap_dir)
partial_see_dir = os.path.join(img_dir,'partial_see')
if not os.path.exists(partial_see_dir):
    os.makedirs(partial_see_dir)
model_dir = os.path.join(base_dir, 'model')

def read_test_data(test_path):
    test_features = []
    test_data = np.genfromtxt(test_path, delimiter = ',', skip_header = 1, dtype = None)
    for _, feature in test_data:
        test_features.append(feature.split(b' '))
        break
    test_features = np.array(test_features, dtype = 'float32') / 255
    return test_features.reshape(test_features.shape[0], 48, 48, 1)

def main():
    parser = argparse.ArgumentParser(prog='plot_saliency.py',
            description='ML-Assignment3 visualize attention heat map.')
    parser.add_argument('--epoch', type=int, metavar='<#epoch>', default=80)
    args = parser.parse_args()
    model_name = "best_model_%d" % time.time()
    emotion_classifier = load_model('best_model')
    print(colored("Loaded model from {}".format(model_name), 'yellow', attrs=['bold']))

    private_pixels = read_test_data('test.csv')

    input_img = emotion_classifier.input
    img_ids = [0]

    for idx in img_ids:
        pixels = private_pixels[idx].reshape(1, 48, 48, 1)

        val_proba = emotion_classifier.predict(pixels)
        pred = val_proba.argmax(axis=-1)
        target = K.mean(emotion_classifier.output[:, pred])
        grads = K.gradients(target, input_img)[0]
        fn = K.function([input_img, K.learning_phase()], [grads])

        grad_val = fn([pixels, 0])[0].reshape((48,48))
        heatmap = np.zeros((48,48))
        for i in range(48):
            for j in range(48):
                heatmap[i][j] = abs(grad_val[i][j])
        thres = 0.1
        see = pixels.reshape(48,48)*255

        see[np.where(heatmap <= thres)] = np.mean(see)

        plt.figure()
        plt.imshow(heatmap, cmap=plt.cm.jet)
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join(cmap_dir, 'privateTest', '{}.png'.format(idx)), dpi=100)

        plt.figure()
        plt.imshow(see,cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join(partial_see_dir, 'privateTest', '{}.png'.format(idx)), dpi=100)

if __name__ == "__main__":
    main()