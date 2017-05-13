#!/usr/local/bin/python3
import sys
import nltk
import numpy
import word2vec
import matplotlib.pyplot as pyplot
from adjustText import adjust_text
from sklearn.manifold import TSNE

def load_data(path):
    with open(path) as f:
        words = nltk.tokenize.word_tokenize(f.read())
        pos_tags = nltk.pos_tag(words)
        return words, dict(pos_tags)
    raise OSError(2, 'No such file or directory', path)

def get_most_frequent_words(k, model, pos_tags):
    words_vec = []
    words_table = []
    tags = ['JJ', 'NNP', 'NN', 'NNS']
    punctuations = [',', '.', ':', ';', '’', '!', '?', '—', '“']
    for word in model.vocab:
        flag = sum([1 if p in word else 0 for p in punctuations]) > 0
        if flag or word not in pos_tags or pos_tags[word] not in tags or len(word) <= 1:
            continue
        words_vec.append(model[word])
        words_table.append(word)
        if len(words_vec) >= k:
            break
    return words_table, numpy.array(words_vec)

def main():
    words, pos_tags = load_data('all.txt')
    word2vec.word2phrase('all.txt', 'word2phrase.txt', verbose = False)
    word2vec.word2vec('word2phrase.txt', 'word2vec.bin', alpha = 0.087, hs = 1, size = 100, verbose = False)
    model = word2vec.load('word2vec.bin')
    words_table, words_vec = get_most_frequent_words(500, model, pos_tags)
    tsne = TSNE(n_components = 2, random_state = 87)
    words_t_vec = tsne.fit_transform(words_vec)
    # show 
    figure = pyplot.figure(figsize = (12, 6), dpi = 150)
    pyplot.scatter(words_t_vec[:, 0], words_t_vec[:, 1], c = 'b', alpha = 0.2, s = 15)
    texts = []
    for vec, text in zip(words_t_vec, words_table):
        texts.append(pyplot.text(vec[0], vec[1], text, size = 5))
    adjust_text(texts, arrowprops = dict(arrowstyle = '-', color = 'k', lw = 0.5))
    pyplot.show()
    figure.savefig('figure.png')

if __name__ == '__main__':
    main()
