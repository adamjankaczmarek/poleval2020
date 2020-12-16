from typing import Set, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from lat_extending.graph import DirectedGraph
from lat_extending.lat import Lattice


class PageRank:
    graph: Union[None, DirectedGraph]
    words_set: Set[str]
    lattice: Union[None, Lattice]

    def __init__(
        self,
        unigrams,
        bigrams,
        word2vec,
        lattice: Lattice = None,
        graph: DirectedGraph = None
    ):

        print('creating set')
        self.words_set = set()
        if lattice is not None:
            self.words_set |= lattice.words_set
        if graph is not None:
            self.words_set |= graph.get_set()
        self.words_set &= set(unigrams.index) & set(word2vec.index)

        print('set created')

        self.unigrams = unigrams.loc[self.words_set]
        print('unigrams')
        self.word2vec = word2vec.loc[self.words_set]
        print('word2vec')
        self.bigrams = \
            bigrams.join(self.word2vec, 'word1', 'inner', rsuffix='aa').join(self.word2vec, 'word2', 'inner',
                                                                             rsuffix='bb')[
                ['bigram_score']]
        print('bigrams')
        self.uni_sum = self.unigrams.sum()

        self.vec_len = np.vstack(self.word2vec['word_vector'])
        self.vec_len_2 = np.sqrt(np.sum(self.vec_len * self.vec_len, axis=1))
        print('vec_len')

        self.graph = graph

    def run(self, rounds=20, d=0.85, recreate=False, vec_const=(5 / 3) * 100, bi_const=-2 / 5 * 100) -> DirectedGraph:
        if self.graph is None or recreate:
            print("Creating graph")
            self._create_graph(vec_const=vec_const, bi_const=bi_const, edge_weights=False)

        elif recreate:
            print("Adding edge weights to existing graph")
            self._fill_graph(vec_const=vec_const, bi_const=bi_const)

        else:
            print("Creating graph skipped")

        for node in self.graph:
            node.value = float(self.unigrams.loc[node.word, 'uni_value']) if node.word in self.unigrams.index else 1.

        for _ in tqdm(range(rounds)):
            self._round(d=d)

        return self.graph

    def _create_graph(self, vec_const=(5 / 3) * 100, bi_const=-2 / 5 * 100, edge_weights=True):
        self.graph = DirectedGraph()

        for w1 in tqdm(self.words_set):
            score = self._get_neighbours_scores(w1, bi_const, vec_const)

            word_list = list(
                filter(lambda x: type(x) == str, score.index)
            )
            word_scores = [
                float(score[x])
                if edge_weights else 1. for x in word_list
            ]

            if len(word_list) > 0:
                self.graph.add_node(w1, word_list, word_scores)

    def _fill_graph(self, vec_const, bi_const):
        for node in tqdm(self.graph):
            score = self._get_neighbours_scores(
                node.word, bi_const, vec_const,
                list({edge.node.word for edge in node.out_edges}),
                cut_too_small=False
            )

            for edge in node.out_edges:
                edge.set_weight(node, float(score.loc[edge.node.word]) if edge.node.word in score.index else 0.1)

    def _get_neighbours_scores(self, w1, bi_const, vec_const, neighbours=None, cut_too_small=True):

        if w1 in self.word2vec.index:

            vec_scores = np.sum(
                self.word2vec.loc[w1, 'word_vector'] * self.vec_len,
                axis=1
            ) \
                         / (
                             self.vec_len_2[self.word2vec.index.get_loc(w1)] * self.vec_len_2
                         ) * vec_const
            _neighbours = set(self.word2vec.index)
            if neighbours is not None:
                _neighbours &= set(neighbours)
        else:
            vec_scores = np.zeros(len(self.word2vec.index))
            _neighbours = []
        vec_scores_frame = pd.DataFrame(vec_scores, self.word2vec.index, ['vec_score']).loc[_neighbours]

        if w1 in self.bigrams.index.get_level_values(0):
            _neighbours = set(self.bigrams.loc[w1].index)
            if neighbours is not None:
                _neighbours &= set(neighbours)
            bi_scores = self.bigrams.loc[w1].loc[_neighbours] * bi_const
        else:
            bi_scores = self.bigrams.iloc[:0]
        score = bi_scores.join(vec_scores_frame, 'word2', 'outer').fillna(0.1)
        if 'word2' not in score.columns:
            score['word2'] = score.index
        score = score.reset_index(drop=True).set_index('word2')

        final_score = 1 / (score['vec_score'] * score['bigram_score'])
        if cut_too_small:

            if type(final_score) == pd.Series:
                return final_score[final_score > 1]
            return final_score[final_score['score'] > 1]
        return final_score

    def _round(self, d=0.85):
        for node in tqdm(self.graph):
            if node.word in self.unigrams.index:
                node_supp = sum([
                    in_edge.node.value * in_edge.weight / in_edge.node.out_sum if in_edge.node.out_sum != 0 else
                    (print(node), 0)[1]

                    for in_edge in node.in_edges
                ])
                node.value = float(d * self.unigrams.loc[node.word, "uni_value"] / self.uni_sum + d * node_supp)

    def _connection(self, word1, word2):
        val = np.log(self.bigrams[word1][word2] / self.unigrams[word1]) / -7. if word2 in self.bigrams[word1] else 3
        val *= cosine_dist(self.word2vec[word1], self.word2vec[word2]) * 5 / 3. \
            if word1 in self.word2vec and word2 in self.word2vec else 3
        return val < 1


def cosine_dist(u, v):
    return 1 - np.dot(u.T, v) / (np.sqrt(np.dot(u.T, v)) * np.sqrt(np.dot(u.T, v)))
