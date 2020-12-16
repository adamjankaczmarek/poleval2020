import os
import pickle
import models
import time
import argparse
import persisted_dictionary
import filelock
import socket

parser = argparse.ArgumentParser()

parser.add_argument('-cap-word-score', help='If provided will cap per word score at this value', required = True)
parser.add_argument('-ignore-symbols', help='If provided will ignore some symbols', required = True)
parser.add_argument('-lock', required=True)
parser.add_argument('-to-rescore', required=True)

args = parser.parse_args()

next_candidate_lock = filelock.SoftFileLock(args.lock+'.lock')
my_name = socket.gethostname() + os.environ.get("CUDA_VISIBLE_DEVICES")
my_done = set()

def next_candidate():
	used = set()
	my_used = set()
	with next_candidate_lock:
		try:
			with open(args.lock, 'r') as f:
				for line in f:
					uid, name = line.strip().split(' ')
					if name != my_name:
						used.add(int(uid))
					else:
						my_used.add(int(uid))
		except:
			print('New lock')

		for i in range(250):
			if i not in used and i not in my_done:
				res = i
				break
		
		if res not in my_used:
			with open(args.lock, 'a+') as f:
				f.write(str(i) + ' ' + my_name + '\n')
		my_done.add(res)
		return res

if args.ignore_symbols.lower() == "false":
	print('with symbols included')
	find_ignored = False
	file_suff = ""
elif args.ignore_symbols.lower() == "true":
	print('ignoring symbols')
	find_ignored = True
	file_suff = "-no-symbols"
else:
	print ('unrecognized ignore_sybmols parameter')
	exit(0)

if args.cap_word_score is not None:
	file_suff += "-keep-per-word-score"
	rescore_suff = "-capped-at-" + args.cap_word_score
	cap_word_score = float(args.cap_word_score.replace(',','.'))
	print(f"Capped score: {cap_word_score}")
else:
	rescore_suff = ""
	
	
beams_file = args.to_rescore
with open(beams_file, 'rb') as bf:
	beams = pickle.load(bf)
	

rescored = persisted_dictionary.Persisted_dict(args.to_rescore+'-bert_rescore-'+socket.gethostname(), 1)
model = models.Polbert_model('dkleczek/bert-base-polish-uncased-v1')
model.start()
model.max_batch_size = 10
failed = 0

while(True):
	i = next_candidate()
	print(i)
	iii = -1
	for (key, options) in beams:
		iii += 1
		if len(options) <= i:
			continue

		(sen, ac) = options[i]
		print('layer:', i, 'example: %d/%d'%(iii,len(beams)), "sentence with %d chars and %d words"%(len(sen), len(sen.split(' '))), 'failed:%d'%failed)
		if (key, i) not in rescored:
			words = sen.strip().split(' ')
			examples = []
			for xxx in range(len(words)):
				examples.append ((' '.join(words[:xxx]), words[xxx], ' '.join(words[xxx+1:] )))

			if len(examples) > 350:
				failed += 1
				continue
			b_score = model.score_words(examples)
			rescored[(key, i)] = ({'ac':ac, 'bert':b_score, 'sen':sen})

