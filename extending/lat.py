import itertools
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple, Union

import Levenshtein
import numpy as np
import pybktree
from jiwer import wer
from tqdm import tqdm

from lat_extending.graph import DirectedGraph
from lat_extending.util import get_file_len


class Lattice:
    bk_tree: Union[None, pybktree.BKTree]
    words_set: Set[str]
    _head: str
    _edges: Dict[Any, set]
    _nodes: Dict[Any, list]
    graph: DirectedGraph

    @staticmethod
    def _get_node_score(node):

        node_scores = '\n' \
                      + '\n(' \
                      + ") | (".join({node_info['lang'] for node_info in node}) \
                      + ')\n(' \
                      + ") | (".join({node_info['acu'] for node_info in node}) \
                      + ')'
        return node_scores

    @staticmethod
    def get_random_lat_chunk(fname):
        lines = []
        random_start_line = np.random.rand(1) * get_file_len(fname)
        with open(fname, 'r') as file:
            second_nl = False
            for i, line in enumerate(file):
                if i < random_start_line and random_start_line >= 1:
                    continue

                if random_start_line < 1 or second_nl:
                    lines.append(line)
                if second_nl and line == "\n":
                    break
                elif line == "\n":
                    second_nl = True
        return lines

    def __init__(self, chunk='', random=False, fname=''):
        self._nodes = defaultdict(list)
        self._edges = defaultdict(set)
        self._head = ''
        if random:
            chunk = self.get_random_lat_chunk(fname)

        self.create_from_chunk(chunk)
        self.graph = DirectedGraph(
            nodes=self._nodes,
            edges=self._edges,
            head=self._head
        )

        self.words_set = self.get_word_set('base')

        # self.bk_extension_function = lambda f: (
        #     lambda x: set(
        #         map(
        #             lambda x: x[1],
        #             self.bk_tree.find(
        #                 x,
        #                 f(x)
        #             )
        #         )
        #     )
        # )
        self.bk_tree = None

    def bk_extension_function(self, f):
        def find_in_tree(word):
            return set(
                map(
                    lambda x: x[1],
                    self.bk_tree.find(
                        word,
                        f(word)
                    )
                )
            )

        return find_in_tree

    def create_from_chunk(self, chunk):
        lines = chunk.split('\n') if type(chunk) == str else chunk
        if len(lines) < 1:
            return

        self._head = lines[0].replace(' ', '').replace('\n', '')
        for line in lines[1:]:
            if line != '' and line != '\n':
                node = line.split(' ')
                if len(node) < 4:
                    continue
                scores = node[3].split(",")
                self.add(
                    parent_node=int(node[0]),
                    child_node=int(node[1]),
                    lang_score=scores[0],
                    acu_score=scores[1],
                    word=node[2].replace('<', '[').replace('>', ']').lower(),
                    hops_str=scores[2]
                )

    def add(self, parent_node, child_node, lang_score, acu_score, word, hops_str):
        self._nodes[child_node].append(dict(
            lang=lang_score,
            acu=acu_score,
            word=word,
            parent=parent_node,
            hops_str=hops_str,
            extended=set(),
            reduced=set(),
        ))
        self._edges[parent_node].add(child_node)

    def show_graph(self):
        return self.graph

    def extend_bk(self, f):
        if self.bk_tree is not None:
            self.extend(self.bk_extension_function(f), True)
        else:
            raise ValueError("bk tree is not set yet")

    def set_bk_tree(self, unigrams=None, tree=None):
        if tree is not None:
            self.bk_tree = tree
        else:
            if type(unigrams) == dict:
                unigrams = unigrams.keys()
            self.bk_tree = pybktree.BKTree(Levenshtein.distance, unigrams)

    def extend(self, extend_func, override=False, func_kwargs=None, ):
        if func_kwargs is None:
            func_kwargs = {}
        for node in tqdm(self.graph):
            extend_word_set = extend_func(node.word, **func_kwargs)
            node.extend(extend_word_set, override=override)

        self.words_set = self.get_word_set('extended')

    def reduce_with_pr(self, graph: DirectedGraph, top=10):
        for node in self.graph:
            extension_scores = {word: graph[word].value for word in node.extension if
                                word in graph and word != node.word}
            node.reduce(
                set(
                    itertools.islice(
                        sorted(
                            extension_scores.keys(),
                            key=lambda x: extension_scores[x],
                            reverse=True
                        ),
                        top
                    )
                )
            )

    def print_extension(self):
        for node in self.graph:
            print(node.extension)

    def get_word_set(self, mode: str = 'base') -> Set[str]:

        word_set = {node.word for node in self.graph}
        if mode == 'base':
            return word_set
        if mode == 'extended':
            word_set.update(*[node.extension for node in self.graph])
        elif mode == "reduced":
            word_set.update(*[node.reduced for node in self.graph])
        return word_set

    def _repr_svg_(self):
        return self.graph._repr_svg_()

    def oracle_best(self, reference_file, mode='reduced'):
        reference = reference_file[self._head].split()
        cache = {}
        edge_graph = list(self.graph.get_edge_form(mode).values())[0]

        def dist(i, n):
            if (i, n) in cache:
                return cache[i, n]
            if i == len(reference):
                return 10000000, ['aaaa'] * 10  # TODO
            if n not in edge_graph:
                return len(reference) - 1 - i, []

            wa = reference[i]
            candidates = []

            for node_info in edge_graph[n]:
                node = node_info['to']
                wb = node_info['word']
                # print (wa, wb)
                if wa == wb:
                    d, p = dist(i + 1, node)
                    candidates.append((d, [wb] + p))
                    continue

                d1, p1 = dist(i + 1, n)
                d2, p2 = dist(i, node)
                d3, p3 = dist(i + 1, node)

                candidates.append((d1 + 1, p1))
                candidates.append((d2 + 1, [wb] + p2))
                candidates.append((d3 + 1, [wb] + p3))

            # print ('candidates=', candidates)
            m = min(candidates)
            cache[i, n] = m
            return m

        x = dist(0, 0)
        # print(x[0])
        return x[1]

    def oracle_best_error(self, reference_file, mode='reduced'):
        return wer(
            reference_file[self._head],
            ' '.join(self.oracle_best(reference_file, mode))
        )

    def add_pr_lattice(self, new_edges: List[Tuple[int, int, str]]):
        max_id = max(self.graph.nodes.keys())[0]
        lat_dict = defaultdict(lambda: defaultdict(set))
        edges = defaultdict(set)
        nodes = defaultdict(list)
        for (parent, child, word) in new_edges:
            # lat_dict[parent + max_id + 1][child + max_id + 1].add(word)
            edges[parent].add(child)
            nodes[child].append({
                'lang': -1,
                'acu': -1,
                'word': word,
                'parent': parent,
                'hops_str': '',
                'extended': set(),
                'reduced': set(),
            })

        graph = DirectedGraph(nodes=nodes, edges=edges, head=self._head)
        return graph
        # nodes
        #
        # edges = {parent: {lat_dict[parent].keys()} for parent in lat_dict.keys()}
        # nodes =
        # self.graph.append(lat_dict)
