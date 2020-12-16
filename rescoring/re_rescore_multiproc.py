import pickle
import models
import time
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool, cpu_count, current_process, Lock

import flair, torch
flair.device = torch.device('cpu') 

beams_file = '5k-beams-with-ac-score'


#rescored = []
#iii = 0
#for (_, options) in tqdm(beams):
#	print(len(options))
#	print(iii)
#	iii += 1
#	rescored.append([])
#	batch_size = 125
#	batched = [[]]
#	for i in range(len(options)):
#		if len(batched[-1]) == batch_size:
#			batched.append([])
#		batched[-1].append(options[i])
#	
#	print("split for batches", len(batched))
#
#	for batch in batched:
#		scores = f_model.score_multi([sen for (sen,_,_) in batch])
#		for (f_score, (_, ac, wer)) in zip(scores, batch):
#			rescored[-1].append({'ac':ac, 'wer':wer, 'flair':f_score})


def mp_rescore_beams(beams):
    n_cores = cpu_count() // 2
    pool = Pool(n_cores)
    beams_lengths = list(map(len, np.array_split(beams, n_cores)))
    beams_ends = np.cumsum(beams_lengths)
    print(beams_ends)
    starts_ends = list(zip(([0] + list(beams_ends[:-1])), beams_ends))
    print(starts_ends)
    print(len(beams))
    dfs = pool.map(score_beams, starts_ends)
    pool.close()
    pool.join()
    print(dfs)
    return dfs


def score_beams(params):
    print(params)
    start, end = params
    f_model = load_flair_model()
    beams = load_beams()
    beams = beams[start:end]
    batch_size = 128
    rescored = []
    batched = [[]]

    for _, options in tqdm(beams, desc=f"Beams in range [{start} - {end}]:  "):
        rescored.append([])
        for i in range(len(options)):
            if len(batched[-1]) == batch_size:
                batched.append([])
            batched[-1].append(options[i])

        for batch in batched:
            scores = f_model.score_multi([sen for sen, _, _ in batch])
            for f_score, (_, ac, wer) in zip(scores, batch):
                rescored[-1].append({'ac':ac, 'wer':wer, 'flair':f_score})

    return rescored


def load_flair_model():
    f_model = models.Flair_model('pl-opus-forward')
    f_model.start()
    return f_model

def load_beams():
    with open(beams_file, 'rb') as bf:
        return pickle.load(bf)

def save_rescored_beams(rescored):
    with open(beams_file+'-rescored', 'wb') as bf:
        pickle.dump(rescored, bf)


if __name__ == "__main__":
    beams = load_beams()
    rescored = mp_rescore_beams(beams)
    save_rescored_beams(rescored)
