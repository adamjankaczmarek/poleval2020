import os
import pickle
import argparse
from flair.data import Dictionary
from flair.models import LanguageModel
from flair.embeddings import FlairEmbeddings
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus

params = {
    "hidden_size": 2048,
    "nlayers": 1,
    "sequence_length": 250,
    "mini_batch_size": 256,
    "max_epochs": 10
}


def train_flair_model(corpus_path, is_forward_lm, fine_tune):
    # load the default character dictionary | AK: load 'chars-large' for pl
    dictionary: Dictionary = Dictionary.load('chars-large')
    direction = "forward" if is_forward_lm else "backward"
    # instantiate your language model, set hidden size and number of layers
    if fine_tune:
        language_model = FlairEmbeddings(f"{fine_tune}-{direction}").lm
        dictionary = language_model.dictionary
    else:
        language_model = LanguageModel(dictionary, is_forward_lm, hidden_size=params["hidden_size"], nlayers=params["nlayers"])
    
    print("LM initialized")
    
    # get your corpus, process forward and at the character level
    corpus_cache = f"sejm-textcorpus-{fine_tune}-{direction}.smallval.pickle"
    if not os.path.exists(corpus_cache):
        corpus = TextCorpus(corpus_path, dictionary, is_forward_lm, character_level=True)
        print("corpus initialized")
        with open(corpus_cache, 'wb') as corpus_cache_file:
            pickle.dump(corpus, corpus_cache_file)
    else:
        with open(corpus_cache, 'rb') as corpus_cache_file:
            corpus = pickle.load(corpus_cache_file)

    # train your language model
    trainer = LanguageModelTrainer(language_model, corpus)
    print("trainer initialized")
    trainer.train(f"resources/lm/{fine_tune}-sejm-{direction}",
                  sequence_length=params["sequence_length"],
                  mini_batch_size=params["mini_batch_size"],
                  max_epochs=params["max_epochs"])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str)
    parser.add_argument("--fine-tune", type=str, default=None)
    parser.add_argument("--backward",  action="store_true")
    args = parser.parse_args()
    print(args)

    train_flair_model(args.corpus, not args.backward, args.fine_tune)

