def line_type (line):
	if line == "\n":
		return "Break"
	elif 3 == len(line.split(",")):
		return "Edge"
	elif 3 == len(line.split("-")) or 4 == len(line.split('-')):
		return "Title"
	elif line in ["%d \n"%d for d in range(5500)]:
		return "Dropped"
	else:
		print("Unexpected line" , line)

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
			a, acustic, _ = line.split(",")
			try:
				start, end, word, language = a.split(" ")
				start = int(start)
				end = int(end)
				graph[key].setdefault(start, [])
				graph[key][start].append({"to" : end, "word" : word, "acus" : float(acustic), "lang" : float(language)})
			except:
				pass
		else:
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
	return costs[sentence_len][reference_len]
	
start_sentence = ""
def go_graph(sentence, graph,pos = 0,  score = 0.0):
	sentence = list(filter(lambda x : len(x) != 0, sentence))
	if len(sentence) == 0:
		return score
	
	global start_sentence
	if pos == 0:
		start_sentence = sentence

	if pos not in graph:
		print ("Pos not in graph")
		print (pos)
		print ("sentence", sentence)
		print ("start sentence", start_sentence)
		print (graph)
	for edge in graph[pos]:
		if edge['word'] == sentence[0]:
			return (go_graph(sentence[1:], graph, edge['to'], score + edge['acus']))
	print ("I was not able to finish", pos, graph, sentence)
	exit(1)

if __name__ == "__main__":
	import argparse
	import kenlm

	parser = argparse.ArgumentParser ()

	parser.add_argument ('-sentences', help='File with sentences to score with keys. Keys should be of format key-NUM, were key is used to find lattice and reference, and NUM can be any number', required=True)
	parser.add_argument ('-lattice', help='File with lattice in the same format as poleval data', required=True)
	parser.add_argument ('-model', help='File with kenlm model', required=True)
	parser.add_argument ('-reference', help='File with reference sentences with keys', required=True)

	args = parser.parse_args ()

	model = kenlm.Model(args.model)
	graphs = parse([line for line in open(args.lattice, 'r')])
	
	references = [line.strip() for line in open(args.reference, 'r')]

	references_dict = {}
	for reference in references:
		words = reference.split (' ')
		key = words[0]
		reference = ' '.join(words[1:])
		references_dict[key] = reference

	print ("key, lm, acc, wer")
	for sentence in open(args.sentences, 'r'):
		words = sentence.strip().split(' ')
		key = words[0]
		master_key ='-'.join( key.split('-')[:-1])
		sentence = ' '.join(words[1:])

		wer_score = wer(sentence, references_dict[master_key])
		lm_score = model.score(sentence,bos=True, eos=True)
		acc_score = go_graph(sentence.split(' '), graphs[master_key])
		print("%s, %f, %f, %f"%(key, lm_score, acc_score, wer_score))
