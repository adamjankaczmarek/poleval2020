import pickle
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('-rescored', help="File with pickled rescored dictionary", required=True)
parser.add_argument('-references', help="File with reference. Used to weight average by length", required=True)
parser.add_argument('-lattice', help="file with lattice", required=True)

args = parser.parse_args()

lattice = list(open(args.lattice, 'r'))
lattice_keys = [line.strip() for line in lattice if len(line.strip().split(' ')) == 1 and line.strip().split('-')[-1][:4] == "file" ]
reshuffle = [[] for i in range(40)]
for i in range(len(lattice_keys)):
	reshuffle[i%40].append(lattice_keys[i])

lattice_keys = [e for part in reshuffle for e in part]

print (len(lattice_keys))
weights_dict = {line.strip().split(' ')[0]:(len(line.strip().split(' '))-1) for line in (open(args.references, 'r'))}
weights = [weights_dict[key] for key in lattice_keys]
rescores = pickle.load(open(args.rescored, 'rb'))

def fst (x):
	return x[0]['wer']

def min_wer(x):
	return min([y['wer'] for y in x[:90]])

def best_by_score(alpha, name):
	def f(x):
		return min([(y['ac']*alpha + y[name], y['wer']) for y in x[:90]])[1]
	return f

def average(f):
	return (np.average(np.array(np.array(list(map(f, rescores)))), weights = weights))

print ("kenlm-best: ", average(fst))
print ("optimal: ", average(min_wer))
print ("flair 0.5: ", average(best_by_score(0.5, 'flair')))

a = 0.001
b = 1.001
k = 200
res = (1, 0)
for i in range(0, k+1):
	alpha = a + i*(b-a)*1.0/k
	res = min(res, (average(best_by_score(alpha, 'flair')), alpha))
print ('flair res:', res)
