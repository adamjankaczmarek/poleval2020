# from __future__ import annotations

import datetime
import itertools
import operator
from collections import defaultdict
from queue import Queue
from typing import Any, Dict, List, Set, Tuple, Union

import Levenshtein
from graphviz import Digraph, Graph
from tqdm import tqdm

from utils.polish_text_to_phonemes import say_sentence


class _Graph:
    nodes: Dict[Union[str, Tuple[int, int]], "Node"]
    head: str

    def __getitem__(self, item: Union["Node", Any]):
        if type(item) == Node:
            return self.nodes[item.get_id()]
        else:
            return self.nodes[item]

    def __init__(self, nodes: Dict[Any, List[Dict[Any, Any]]]):
        info_keys = ['lang', 'acu', "parent", 'hops_str']
        self.head = str(datetime.datetime.now())
        self.nodes = {}
        for _id, node in nodes.items():
            for i, word in enumerate(node):
                info = {x: word[x] for x in info_keys}
                info['hops'] = len(info['hops_str'].split('_'))
                self.nodes[(_id, i)] = Node(
                    nodeid=(_id, i),
                    word=word['word'],
                    info=info
                )

    def __iter__(self):
        return iter(self.nodes.values())

    def __len__(self):
        return len(self.nodes)

    def __contains__(self, item):
        if type(item) == Node:
            return item.get_id() in self.nodes
        else:
            return item in self.nodes

    def add_node(self, node: Union["Node", str], neighbours, id=None, neighbours_ids=None):
        if type(node) != Node:
            node = Node(node if id is None else id, node)
        if node.get_id() not in self.nodes:
            self.nodes[node.get_id()] = node
        node = self.nodes[node.get_id()]
        for i, n in enumerate(neighbours):
            if type(n) != Node:
                _n = Node(n if neighbours_ids is None else neighbours_ids[i], n)
            else:
                _n = n
            if _n.get_id() not in self.nodes:
                self.nodes[_n.get_id()] = _n
            neighbours[i] = self.nodes[_n.get_id()]
        return node, neighbours

    def get_set(self, mode: str = '') -> Set[str]:
        res = {node.word for node in self}
        if mode == 'extended':
            res = res.union(*{node.extension for node in self})
        elif mode == 'reduced':
            res = res.union(*{node.reduced for node in self})
        return res

    def get_edge_form(self, mode="reduced") -> Dict[str, Dict[int, List[Dict[str, Union[int, float, str]]]]]:
        """
        Converts a graph from node form to edge form (in edge form words are on edges)
        Returns: dict with one key (head of a graph) containing a dict:
            key: id of a parent
            value: list of edges; each edge is a dict with keys:
                'to': id of child
                 'word' - a word, str
                 'acus' - acoustic cost
                 'lang' - language cost
                 'edit' - phonetic editing cost
                 'hops' - number of 'hops'

        """
        res = defaultdict(list)
        for node in self:
            if 'parent' in node.info:
                node_info = node.get_info(mode)
                for word_info in node_info:
                    if word_info not in res[node.info['parent']]:
                        res[node.info['parent']].append(word_info)
        return {self.head: dict(sorted(res.items(), key=lambda x: x[0]))}

    def _create_dot_graph(self):
        dot = Graph(strict=True)

        for node in self:
            dot.node(str(node.get_id()), str(node.word), label=str(node.value) if node.value is not None else '')
            for edge in node:
                dot.edge(
                    str(node.get_id()),
                    str(edge.node.get_id()),
                    label=f"{edge.weight :.2f}" if edge.weight is not None else ''
                )
        return dot

    def render(self, *args, **kwargs):
        return self._create_dot_graph().render(*args, **kwargs)

    def _repr_svg_(self):
        return self._create_dot_graph()._repr_svg_()


class UndirectedGraph(_Graph):
    def __init__(self, nodes={}, edges={}):
        nodes = nodes.copy()
        super().__init__(nodes)

        for parent_id, sub_nodes in edges.items():
            for p_num, p_node in enumerate(nodes[parent_id]):
                parent_node = self[(parent_id, p_num)]
                for node_group_id in sub_nodes:
                    for i, word in enumerate(nodes[node_group_id]):
                        current = self[(node_group_id, i)]
                        self[parent_node].add_neighbour(current)
                        current.add_neighbour(parent_node)

    def add_node(self, node, neighbours, weights=None):
        if weights is None or len(weights) != len(neighbours):
            weights = defaultdict(lambda: None)
        node, neighbours = super().add_node(node, neighbours)
        for i, n in enumerate(neighbours):
            self.nodes[node.get_id()].add_neighbour(n, weights[i])
            self.nodes[n.get_id()].add_neighbour(node, weights[i])


class DirectedGraph(_Graph):
    head_node: "Node"
    head: str

    def __init__(self, nodes={}, edges={}, head: str = ''):
        nodes = nodes.copy()
        super().__init__(nodes)
        if head != '':
            self.head_node = Node(
                nodeid=(0, 0),
                word="[start=%s]" % head
            )
            self.nodes[(0, 0)] = self.head_node
            nodes[0] = [{"word": '[start]'}]
        self.head = head
        for parent_id, sub_nodes in edges.items():
            for p_num, p_node in enumerate(nodes[parent_id]):
                parent_node = self[(parent_id, p_num)]
                for node_group_id in sub_nodes:
                    for i, word in enumerate(nodes[node_group_id]):
                        current = self[(node_group_id, i)]
                        self[parent_node].add_child(current)
                        current.add_parent(parent_node)

    def _create_dot_graph(self):
        dot = Digraph(strict=True)

        leveled_nodes = [node.get_id() for node in self if type(node.get_id()) == tuple]
        unleveled_nodes = [node for node in self if type(node.get_id()) != tuple]
        for node in tqdm(unleveled_nodes):

            dot.node(str(node.get_id()), f"{node.word}\n{node.value if node.value is not None else '':5f}")

            for edge in node:
                if type(edge) != InEdge:
                    dot.edge(
                        str(node.get_id()),
                        str(edge.node.get_id()),
                        label=f"{edge.weight :.2f}" if edge.weight is not None else ''
                    )
                else:
                    dot.edge(
                        str(edge.node.get_id()),
                        str(node.get_id()),
                        label=f"{edge.weight :.2f}" if edge.weight is not None else ''
                    )

        grouped_nodes = accumulate(leveled_nodes)
        for level in tqdm(grouped_nodes):
            label = " | ".join({self[(level, subnode)].word for subnode in grouped_nodes[level]})
            node_values = {
                self[(level, subnode)].value
                for subnode in grouped_nodes[level]
                if self[(level, subnode)].value is not None
            }
            if len(node_values):
                label += "\n" + " | ".join(
                    {
                        f"{self[(level, subnode)].value:5f}"
                        if self[(level, subnode)].value is not None
                        else ' '
                        for subnode in grouped_nodes[level]
                    }
                )
            dot.node(str(level), str(label))

            for edge in self[(level, grouped_nodes[level][0])]:

                if type(edge) == OutEdge:
                    dot.edge(
                        str(level),
                        str(edge.node.get_id()[0]),
                        label=f"{edge.weight :.2f}" if edge.weight is not None else ''
                    )
                elif type(edge) == InEdge:
                    dot.edge(
                        str(edge.node.get_id()[0]),
                        str(level),
                        label=f"{edge.weight :.2f}" if edge.weight is not None else ''
                    )
                else:
                    dot.edge(
                        str(edge.node.get_id()[0]), str(level),
                        label=f"{edge.weight :.2f}" if edge.weight is not None else ''
                    )
                    dot.edge(
                        str(level), str(edge.node.get_id()[0]),
                        label=f"{edge.weight :.2f}" if edge.weight is not None else ''
                    )

        return dot

    def add_node(self, word: Union["Node", str], neighbours=[], weights=None, head=False, id=None, neighbours_ids=None):
        if weights is None or len(weights) != len(neighbours):
            weights = defaultdict(lambda: None)
        node, neighbours = super().add_node(word, neighbours, neighbours_ids=neighbours_ids, id=id)
        if head:
            self.head_node = node
        for i, n in enumerate(neighbours):
            self.nodes[node.get_id()].add_child(n, weights[i])
            self.nodes[n.get_id()].add_parent(node, weights[i])

    @property
    def donuts(self):
        return self._donut_gen(self.head_node)

    def _donut_gen(self, start_node):
        while len(start_node.out_edges) != 0:

            res = self.get_a_donut(start_node)
            if res is not None:
                graph, start_node = res
                yield graph
            else:
                start_node = start_node.out_edges[0].node

    def _test(self):
        for i in range(5):
            yield i

    @staticmethod
    def get_a_donut(start_node: "Node") -> Union[None, Tuple["DirectedGraph", "Node"]]:
        graph = DirectedGraph()
        graph.add_node(start_node, head=True)
        if len(start_node.out_edges) < 2:
            return None
        paths = set()
        nodes_queue: Queue[Node] = Queue()
        visited = defaultdict(lambda: 0)
        visited[start_node.get_id()] = 1
        node_paths = defaultdict(set)
        for edge in start_node.out_edges:
            nodes_queue.put(edge.node)
            paths.add(edge.node.get_id())
            node_paths[edge.node.get_id()].add(edge.node.get_id())

        current_node: Union[Node, None] = None
        while not nodes_queue.empty() and (current_node is None or node_paths[current_node.get_id()] != paths):
            # TODO: krawędź z poza pączka
            current_node: Node = nodes_queue.get()
            if not visited[current_node.get_id()]:
                graph.add_node(current_node)

            visited[current_node.get_id()] += 1

            if visited[current_node.get_id()] >= len(current_node.in_edges):
                for i, out_edge in enumerate(current_node.out_edges):
                    nodes_queue.put(out_edge.node)
                    if i > 0:
                        paths.add(out_edge.node.get_id())
                        node_paths[out_edge.node.get_id()].add(out_edge.node.get_id())
                    node_paths[out_edge.node.get_id()] |= node_paths[current_node.get_id()]
        return graph, current_node

    def get_max_subtree(self):
        return self.head_node.get_max_subtree()

    def append(self, lat_dict: Dict[int, Dict[int, Union[Set[str], List[str]]]]):
        edges = {parent: {lat_dict[parent].keys()} for parent in lat_dict.keys()}
        nodes = []
        # for parent in sorted(lat_dict.keys()):
        #     for node in sorted(lat_dict[node].keys()):
        #         for i, word in enumerate(lat_dict[node][child]):
        #             self.add_node(
        #                 word,
        #                 id=(node, i),
        #                 neighbours=[neighbour_word for neighbour_word in lat_dict[child]]
        #             )

        pass


def accumulate(l):
    it = itertools.groupby(l, operator.itemgetter(0))
    return {
        key: list([item[1] for item in subiter])
        for key, subiter in it
    }


class Node:
    _node_id: Union[Tuple[int, int], str]
    word: str
    value: Union[float, None]
    # children: List[Node]
    # parents: List[Node]
    edges: Dict[Any, Union["OutEdge", "InEdge", "UndirectedEdge"]]
    extension: Set[str]
    reduced: Set[str]
    info: Dict[Any, Any]
    max_tree: Union[None, Tuple[float, List[str]]]
    greedy_max_tree: Union[None, List[str]]

    def __init__(self, nodeid: Any, word: str, info=None):
        if info is None:
            info = {}
        self._node_id = nodeid
        self.word = word
        self.info = info
        self.extension = set()
        self.reduced = set()
        self.edges = {}
        self.value = None
        self.max_tree = None
        self.greedy_max_tree = None
        self._out_sum = None

    def extend(self, extend_word_set: Set[str], override: bool = False):
        if self.word in extend_word_set:
            extend_word_set.remove(self.word)
        if override:
            self.extension = extend_word_set
        else:
            self.extension.update(extend_word_set)

    def reduce(self, reduce_word_set: Set[str]):
        self.reduced = reduce_word_set

    def __str__(self):
        all_neighbours = ', '.join([x.__class__.__name__ + ': ' + x.node.word for x in self])
        neighbours = "(Neighbours: " + all_neighbours + ')' if all_neighbours != '' else '(No neighbours)'
        return self.word + neighbours

    def add_child(self, child, weight=None):
        self.edges[child.get_id()] = OutEdge(child, weight)

    def add_parent(self, parent, weight=None):
        self.edges[parent.get_id()] = InEdge(parent, weight)

    def add_neighbour(self, neighbour, weight=None):
        self.edges[neighbour.get_id()] = UndirectedEdge(neighbour, weight)

    def get_id(self):
        return self._node_id

    def __iter__(self):
        return iter(self.edges.values())

    def get_max_subtree(self, len_factor=1) -> Tuple[float, List[str]]:
        cache = {}

        def _max_subtree(node: "Node", sum_value: float, sentence: List[str]):
            if node.get_id() in cache:
                return cache[node.get_id()]

            if 'acu' in node.info:
                sum_value += float(node.info['acu'])
                new_sent = sentence.copy() + [node.word]
            else:
                new_sent = sentence.copy()
            if len(node.out_edges) == 0:
                cache[node.get_id()] = (sum_value, new_sent)
                return (sum_value, new_sent)

            cache[node.get_id()] = min(
                [_max_subtree(edge.node, sum_value, new_sent) for edge in node.out_edges],
                key=lambda x: x[0],
            )
            return cache[node.get_id()]

        # if self.max_tree is None:
        #     if len(self.out_edges) > 0:
        #         min_val, words = max(
        #             [
        #                 edge.node.get_max_subtree()
        #                 for edge in self.out_edges
        #             ],
        #             key=lambda x: (-int(len(x[1]) * len_factor), x[0])
        #         )
        #     else:
        #
        #         min_val = 0
        #         words = []
        #     self.max_tree = min_val + self.value, [self.word] + words

        return _max_subtree(self, 0., [])

    def get_greedy_max_tree(self) -> List[str]:
        if self.greedy_max_tree is None:
            if len(self.out_edges) > 0:
                max_edge = max([edge for edge in self.out_edges], key=lambda x: x.node.value)
                word_list = max_edge.node.get_greedy_max_tree()
            else:
                word_list = []
            self.greedy_max_tree = [self.word] + word_list

        return self.greedy_max_tree

    @property
    def out_edges(self) -> List["Edge"]:
        return [x for x in self if type(x) != InEdge]

    @property
    def in_edges(self) -> List["Edge"]:
        return [x for x in self if type(x) != OutEdge]

    @property
    def out_sum(self) -> Union[float, int]:
        if self._out_sum is None:
            self._out_sum = sum(x.weight for x in self.out_edges)
        return self._out_sum

    @property
    def in_sum(self) -> Union[float, int]:
        return sum(x.weight for x in self.in_edges)

    def get_info(self, mode='reduced') -> List[Dict[str, Union[str, float, int]]]:
        """
        Returns information about this node and all the extending nodes
        Returns: list which for each word in self.reduced and self.word contains a dict with keys:
            'to': id of child
             'word' - a word, str
             'acus' - acoustic cost
             'lang' - language cost
             'edit' - phonetic editing cost
             'hops' - number of 'hops'
        """
        words_list: List[str] = [self.word]
        if mode == 'reduced':
            words_list.extend(list(self.reduced))
        if mode == 'extended':
            words_list.extend(list(self.extension))

        return [
            {
                'to': self.get_id()[0] if type(self.get_id()) == tuple else self.get_id(),
                'word': word,
                'acus': float(self.info.get('acu', -1)),
                'lang': float(self.info.get('lang', -1)) if not i else -1,
                'edit': Levenshtein.distance(say_sentence([self.word]), say_sentence([word])),
                'hops': self.info.get("hops", 0),
            }
            for i, word in enumerate(words_list)
        ]


class Edge:
    id: int
    weight: float
    node: Node

    def __init__(self, node, weight=None):
        self.node = node
        self.weight = weight

    def set_weight(self, start_node, weight):
        self.weight = weight
        self.node.edges[start_node.get_id()].weight = weight

    def __str__(self):
        return f"{self.node.word}({self.weight})"


class OutEdge(Edge):
    pass


class InEdge(Edge):
    pass


class UndirectedEdge(Edge):
    pass
