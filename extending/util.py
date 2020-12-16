import re
from collections import defaultdict

import numpy as np
from tqdm import tqdm


def make_set(lat_chunk):
    return {x.split(' ')[2] for x in lat_chunk if len(x.split(' ')) > 3}


def split_iter(string):
    return (x.group(0) for x in re.finditer(r"([^\n]+\n)*\n", string))


def get_file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def get_set_error(lat_word_set, correct_sentence):
    error_rate = 0
    err_by_length = defaultdict(int)
    for word in correct_sentence.split(' '):
        if word not in lat_word_set:
            err_by_length[len(word)] += 1
            error_rate += 1
            # print(word)
    # print(sorted(err_by_length.items()))
    return error_rate / len(correct_sentence.split(' '))


def get_unigrams(fname, top=0):
    unigrams = {}
    with open(fname, 'r') as f:
        for line in tqdm(f, total=get_file_len(fname)):
            line_split = line.strip().split(' ')
            if int(line_split[0]) < top:
                break
            unigrams[line_split[1].lower()] = int(line_split[0])
    return unigrams


def get_bigrams(fname, top=0):
    bigrams = defaultdict(dict)
    with open(fname, 'r') as f:
        for line in tqdm(f, total=get_file_len(fname)):
            spilt_line = line.strip().split(' ')
            if int(spilt_line[0]) < top:
                break
            bigrams[spilt_line[1].lower()][spilt_line[2].lower()] = int(spilt_line[0])
    return bigrams


def get_word2vec(fname):
    word2vec = {}
    with open(fname, 'r') as f:
        for line in tqdm(f, total=get_file_len(fname)):
            split_line = line.strip().split(' ')
            if len(split_line) > 50:
                word2vec[split_line[0].lower()] = np.array(split_line[1:]).astype(float)
    return word2vec


def create_unigrams_file(fname):
    with open(fname, 'r') as file:
        unigram_string = re.sub("[^A-za-z0-9 ęĘóÓąĄśŚłŁżŻźŹćĆńŃ\n]|\[|\]|_|\\\\|`", '', file.read())
    unigrams = defaultdict(int)
    for line in tqdm(unigram_string.split("\n")):
        splt = line.split()
        if len(splt) == 2:
            unigrams[splt[1]] += int(splt[0])
    return unigrams


def create_bigrams_file(fname):
    with open(fname, 'r') as file:
        bigram_string = re.sub("[^A-za-z0-9 ęĘóÓąĄśŚłŁżŻźŹćĆńŃ\n]|\[|\]|_|\\\\|`", '', file.read())

    bigrams = defaultdict(lambda: defaultdict(int))
    for line in tqdm(bigram_string.split("\n")):
        splt = line.split()
        if len(splt) == 3:
            bigrams[splt[1]][splt[2]] += int(splt[0])

    with open('bigrams', 'w') as f:
        f.write(
            '\n'.join(
                [
                    "%s %s %s" % (x[2], x[0], x[1])
                    for x in sorted(
                    [
                        (bigram[0], *second)
                        for bigram in bigrams.items()
                        for second in bigram[1].items()
                    ],
                    key=lambda _arg: _arg[2],
                    reverse=True
                )
                ]
            )
        )
    # return bigrams
