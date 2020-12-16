import pickle
import models
import time
from tqdm import tqdm
import sys

beams_file = '5k-beams-with-ac-score'
MAX_LEN = 2171

def load_flair_model():
    f_model = models.Flair_model('pl-opus-forward')
    f_model.start()
    return f_model

def rescore(beams, min_batch_len):
    f_model = load_flair_model()
    rescored = []
    iii = 0
    for (_, options) in tqdm(sorted(beams, key=lambda b: -len(b[1][0][0]))):
        print(f"#Options = {len(options)}")
        print(f"Beam length = {len(options[0][0])}")
        max_sample_len = max([len(options[x][0]) for x in range(len(options))])
        print(f"Max sample length = {max_sample_len}")
        print(f"iii = {iii}")
        iii += 1
        rescored.append([])
        
        batch_size = min(len(options), (MAX_LEN * min_batch_len) // max_sample_len)
        batched = [[]]
        for i in tqdm(range(len(options)), desc="Batching"):
        	if len(batched[-1]) == batch_size:
        		batched.append([])
        	batched[-1].append(options[i])
        
        print("split for batches", len(batched))
        
        for batch in tqdm(batched, desc="Batches in current utterance"):
            try:
                print(len(batch))
                print(f"#Chars in batch[0][0] = {len(batch[0][0])}")
                scores = f_model.score_multi([sen for (sen,_,_) in batch])
            except RuntimeError as e:
                print(len(batch))
                print(batch[0][0])
                raise e
            for (f_score, (_, ac, wer)) in zip(scores, batch):
                rescored[-1].append({'ac':ac, 'wer':wer, 'flair':f_score})
                with open("incremental-rescored", 'a') as irf:
                    irf.writeline(str({'ac':ac, 'wer':wer, 'flair':f_score}))


        
    print(rescored)
    return rescored





if __name__ == "__main__":
    start, end, batch = map(int, sys.argv[1:4])
    print([start, end, batch])
    with open(beams_file, 'rb') as bf:
        beams = pickle.load(bf)

    beams = beams[start:end]

    rescored = rescore(beams, batch)

    with open(beams_file+'-rescored', 'wb') as bf:
        pickle.dump(rescored, bf)

