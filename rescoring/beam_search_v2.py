import time
import heapq
import gc
import models
import pickle

def line_type (line):
	if line == "\n":
		return "Break"
	elif 3 == len(line.split(",")):
		return "Edge"
	elif 3 == len(line.split("-")) or 4 == len(line.split('-')) or ((line[:2] == "EN" or line[:2] == "PL") and line[-5:] == "_pl \n" and len(line) == 11):
		return "Title"
	elif line in ["%d \n"%d for d in range(5500)]:
		return "Dropped"
	else:
		print("Unexpected line [%s]"%line[:-1])

def parse (lines):
	cost_to_use = 1
	graph = {}
	key = ""
	for line in lines:
		type_ = line_type(line)
		if type_ == "Break":
			key = ""
		elif type_ == "Title":
			key = line.strip()
			graph[key] = {}
		elif type_ == "Dropped":
			pass
		elif type_ == "Edge":
			a, acustic, x = line.split(",")
			try:
				start, end, word, language = a.split(" ")
				start = int(start)
				end = int(end)
				graph[key].setdefault(start, [])
				graph[key][start].append({"to" : end, "word" : word, "acus" : float(acustic), "lang" : float(language), "steps" : len(x.split('_')), "x":x})
			except:
				try:
					start, language = a.split(" ")
					start = int(start)
					graph[key].setdefault(start, [])
					graph[key][start].append({"to" : -1, "word" : "", "acus" : float(acustic), "lang" : float(language), "steps" : len(x.split('_')), "x":x})
				except:
					pass
		else:
			print ("[%s]"%type_)
			raise "Not expected type [%s]"%type_
	return graph

def wer(sentence, reference):
	sentence_words = list(filter (lambda x : len(x) > 0, list(sentence.split(' '))))
	reference_words = list(reference.split(' '))

	reference_len = len(reference_words)
	sentence_len = len(sentence_words)

	costs = [ [ reference_len + sentence_len + 123 for i in range(reference_len + 1) ] for i in range (sentence_len + 1) ]
	costs[ 0 ][ 0 ] = 0

	
	for i in range(sentence_len):
		costs[ i + 1 ][ 0 ] = i + 1
	for i in range(reference_len):
		costs[ 0 ][ i + 1 ] = i + 1

	for i in range(sentence_len):
		for j in range(reference_len):
			costs [ i + 1 ][ j + 1 ] = 1 + min (costs[ i ][ j ], min (costs[ i ][ j + 1], costs[ i + 1 ][j]))
			if sentence_words[ i ] == reference_words[ j ]:
				costs[ i + 1][ j + 1] = min(costs[ i + 1 ][ j + 1 ], costs[ i ][ j ])
	return costs[sentence_len][reference_len]*1.0/(reference_len)

def gridify(graph):
	queue = [(0, 0)]
	times = {0:0}

	while queue != []:
		t, p = heapq.heappop(queue)
		if p in graph:
			for edge in graph[p]:
				new_t = t + (edge['steps'] if 'steps' in edge else edge['hops'])
				new_p = edge['to']
				if new_p not in times:
					times[new_p] = new_t
					heapq.heappush(queue, (new_t, new_p))
	return times

def collapse_grid(graph):
	i_size = { 0:0 }
	for node in graph:
		for edge in graph[node]:
			if edge['to'] in i_size:
				i_size[edge['to']] += 1
			else:
				i_size[edge['to']] = 1

	m_time = { key:0 for key in i_size }
	m_time[0] = 0
	
	queue = [ node for node in i_size if i_size[node] == 0]

	while queue != []:
		node = queue[0]
		queue = queue[1:]
		if node not in graph:
			continue

		for edge in graph[node]:
			m_time[edge['to']] = max(m_time[node] + (edge['steps'] if 'steps' in edge else edge['hops']), m_time[edge['to']])
			i_size[edge['to']] -= 1
			if i_size[edge['to']] == 0:
				queue.append(edge['to'])
	return graph, m_time
		
	result_graph = {}
	for node in graph:
		if node not in m_time:
			continue

		t_node = m_time[node]

		if t_node not in result_graph:
			result_graph[t_node] = []

		for edge in graph[node]:
			t_edge = {key:edge[key] for key in edge.keys()}
			t_edge['to'] = m_time[edge['to']]
			if t_node == t_edge['to']:
				print ('cycle', edge)
				exit(1)
			result_graph[t_node].append(t_edge)
	times = {}
	for time in m_time:
		times[m_time[time]] = m_time[time]
	return result_graph, times
	
def any_extensions(state, graph):
		for (_, pos, _) in state:
				if pos in graph:
						return True
		return False

def all_extensions(state, graph, acustic_multiplier, model):
		for (score, (ac, lm), pos, sentence) in state:
				if pos in graph:
						for edge in graph[pos]:
								new_sentence = (sentence + ' ' + edge['word']).strip(' ')
								new_lm = model.update_state(lm, edge['word'])
								new_ac = ac + edge['acus']
								new_score = model.current_score(new_lm) + new_ac*acustic_multiplier
								yield (new_score,  (new_ac, new_lm), edge['to'], new_sentence)
				else:
					pass

def run_beamsearch(model, graph, beam_size, acustic_multiplier):
		if True:
			graph, times = collapse_grid(graph)
		else:	
			times = gridify(graph)

		state = {0:[(0, (0, model.new_state()), 0, '')]}

		time_queue = [0]
		res = []

		while(time_queue != []):
				t = heapq.heappop(time_queue)

				for  (_, (ac_score, lm_state), pos, sen) in state[t]:
					if pos not in graph:
						res.append ((model.final_score(lm_state) + acustic_multiplier * ac_score, (ac_score, model.current_score(lm_state)), sen))
						if len(res) > beam_size:
							res = sorted(res)[:beam_size]
				
				touched_times = set()
				for candidate in (all_extensions(state[t], graph, acustic_multiplier, model)):
					(score, (ac, lm), pos, sen) = candidate
					c_t = times[pos]
					if c_t not in state:
						state[c_t] = []
						heapq.heappush(time_queue, c_t)

					state[c_t].append(candidate)
					touched_times.add(c_t)
				for tt in touched_times:
					if len(state[tt]) > beam_size:
						state[tt] = sorted(state[tt])[:beam_size]

				state[t] = []

		return res

def parallel_beamserch(x):
		model, beam_size, acustic_multiplier, elements = x
		model.start()

		res = []
		for (key, graph, reference) in	elements:
				results = run_beamsearch(model, graph, beam_size, acustic_multiplier)
				results = sorted(results)
				(score, pos, best_sentence) = results[0]

				if reference is not None:
					wer_score = (wer(best_sentence, reference))

					options = [(sentence, ac_score, wer(sentence, reference)) for (_, (ac_score, _), sentence) in results]

					res.append((key, wer_score, min([wer for (_,_,wer) in options]), best_sentence, len(reference.split(' ')), options))
				else:
					res.append((key, [(sentence, ac_score) for (_, (ac_score, _), sentence) in results]))

		return res


if __name__ == "__main__":
	import argparse
	import kenlm
	import multiprocessing
	import numpy as np

	parser = argparse.ArgumentParser ()

	parser.add_argument ('-lattice', help='File with lattice in the same format as poleval data', required=True)
	parser.add_argument ('-kenlm-model', help='File with kenlm model')
	parser.add_argument ('-flair-model', help='flair model name')
	parser.add_argument ('-reference', help='File with reference sentences with keys')
	#parser.add_argument ('-one-best', help='File with 1best sentences with keys', required=True)
	parser.add_argument ('-proc', help='Number of processes to use', required=True, type=int)
	parser.add_argument ('-beam-size', help='Beam size', required=True, type=int)
	parser.add_argument ('-acustic-multiplier', help='Multiplier of accustic cost', required=True)
	parser.add_argument ('-save-beams-to', help='If provided save all final sentences to a file')

	args = parser.parse_args ()
	
	if args.kenlm_model is None and args.flair_model is not None:
		model = models.Flair_model(args.flair_model)
	elif args.kenlm_model is not None and args.flair_model is None:
		model = models.Kenlm_model(args.kenlm_model)
	else:
		print ("Exactly one of flair-model or -kenlm-model must be provided")
		exit(1)

	graphs = parse([line for line in open(args.lattice, 'r')])
	
	references_dict = {}
	if args.reference is not None:
		references = [line.strip() for line in open(args.reference, 'r')]

		for reference in references:
			words = reference.split (' ')
			key = words[0]
			reference = ' '.join(words[1:])
			references_dict[key] = reference

	"""one_bests_dict = {}
	for one_best in	[line.strip() for line in open(args.one_best, 'r')]:
		words = one_best.split (' ')
		key = words[0]
		one_best = ' '.join(words[1:])
		one_bests_dict[key] = one_best"""
	
	keys_list = list(graphs.keys())
	keys_split = [[] for i in range(args.proc)]
	for i in range(len(keys_list)):
		keys_split[i%args.proc].append(keys_list[i])
	pool = multiprocessing.Pool(args.proc)

	for key in keys_list:
		if key not in references_dict:
			references_dict[key] = None

	inputs = [ (model, args.beam_size, float(args.acustic_multiplier.replace(',', '.')), [(key, graphs[key], references_dict[key]  ) for key in keys]) for keys in keys_split]
	results = pool.map(parallel_beamserch, inputs)
	
	
	if args.reference is not None:
		scores = [score for inner_result in results for (key, score, best_score, sentence, ref_len, options) in inner_result ]
		best_scores = [best_score for inner_result in results for (key, score, best_score, sentence, ref_len, options) in inner_result ]
		#scores_b1 = [score_b1 for inner_result in results for (key, score, score_b1, sentence, ref_len) in inner_result ]
		ref_len = [ref_len for inner_result in results for (key, score, best_score, sentence, ref_len, options) in inner_result ]
		print ("beam scores:", np.average(scores, weights = ref_len))
		print ("ORECLE BEST beam scores:", np.average(best_scores, weights = ref_len))
		#print ("1-best:", np.average(scores_b1, weights = ref_len))

	
	if args.save_beams_to is not None:
		if args.reference is not None:
			beams = [(key, options) for inner_result in results for (key, score, best_score, sentence, ref_len, options) in inner_result ]
		else:
			beams = [(key, options) for inner_result in results for (key, options) in inner_result]
		print (beams)
		with open(args.save_beams_to, 'wb') as fl:
			pickle.dump(beams, fl)



