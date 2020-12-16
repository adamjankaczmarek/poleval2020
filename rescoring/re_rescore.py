import pickle
import models
import time
from tqdm import tqdm

beams_file = '5k-beams-with-ac-score'
with open(beams_file, 'rb') as bf:
	beams = pickle.load(bf)


#b_model = models.Polbert_model('dkleczek/bert-base-polish-uncased-v1')
#b_model.start()

f_model = models.Flair_model('pl-opus-forward')
f_model.start()

rescored = []
iii = 0
for (_, options) in tqdm(beams):
	print(len(options))
	print(iii)
	iii += 1
	rescored.append([])
	batch_size = 128
	batched = [[]]
	for i in range(len(options)):
		if len(batched[-1]) == batch_size:
			batched.append([])
		batched[-1].append(options[i])
	
	print("split for batches", len(batched))

	for batch in batched:
		scores = f_model.score_multi([sen for (sen,_,_) in batch])
		for (f_score, (_, ac, wer)) in zip(scores, batch):
			rescored[-1].append({'ac':ac, 'wer':wer, 'flair':f_score})

print (rescored)

with open(beams_file+'-rescored', 'wb') as bf:
	pickle.dump(rescored, bf)
