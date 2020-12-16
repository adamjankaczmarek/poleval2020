import kenlm
import flair
import math
import torch
import transformers
import time

class Kenlm_model(object):
	def __init__(self, filename):
		self.filename = filename

	def start(self):
		self.model = kenlm.Model(self.filename)
	
	def new_state(self):
		state = kenlm.State()
		self.model.BeginSentenceWrite(state)
		return (0, state)
	
	def score(self, sentence):
		return -self.model.score(sentence)

	def update_state(self, ext_state, word):
		(score, state) = ext_state
		tmp_state = kenlm.State()
		tmp_score = score - self.model.BaseScore(state, word, tmp_state)
		return (tmp_score, tmp_state)
	
	def current_score(self, ext_state):
		(score, state) = ext_state
		return score

	def final_score(self, ext_state):
		(score, state) = ext_state
		tmp_state = kenlm.State()
		return score - self.model.BaseScore(state, "</s>", tmp_state)


class Flair_model(object):
	def __init__(self, model_name):
		self.tokens = {}
		self.model_name = model_name
	
	def start(self):
		with torch.no_grad():
			self.model = flair.embeddings.FlairEmbeddings(self.model_name).lm

	def _to_tokens(self, sentence):
		with torch.no_grad():
			sentence = sentence
			tokens = [self.model.dictionary.get_idx_for_item(char) for char in sentence]
			return torch.tensor(tokens).unsqueeze(1).to(flair.device)

	def _pred_to_score(self, sentence, predictions):
			sentence = sentence
			predictions = torch.narrow(predictions, 0, 0, len(sentence))
			tokens = [self.model.dictionary.get_idx_for_item(char) for char in sentence]
			predictions = predictions.view(-1, len(self.model.dictionary))
			target_tensor = torch.tensor(tokens).to(flair.device)
			return torch.nn.CrossEntropyLoss(reduction='sum')(predictions, target_tensor).item()
		

	def score(self, sentence):
		with torch.no_grad():
			word_tensor = self._to_tokens(' ' + sentence + ' ')
			predictions, _, _ = self.model.forward(word_tensor, self.model.init_hidden(1))	

			return self._pred_to_score(sentence + ' .', predictions)

	def score_multi(self, sentences):
		size = len(sentences)
		total_length = max([len(sentence) for sentence in sentences])
		padded_sentences = [(' ' + sentence + ' .' + (12+total_length)*"#")[:total_length+12] for sentence in sentences]
		tokens = torch.cat(list(map(self._to_tokens, padded_sentences)), dim=1)
		predictions, _, _ = self.model.forward(tokens, self.model.init_hidden(size))
		
		score_m = [self._pred_to_score(sentences[i] + ' .', predictions.narrow(1, i, 1)) for i in range(size)]
		#score_s = [self.score(sentence) for sentence in sentences]

		return score_m

	def new_state(self):
		with torch.no_grad():
			tokens = [self.model.dictionary.get_idx_for_item(char) for char in ' ']
			word_tensor = torch.tensor(tokens).unsqueeze(1).to(flair.device)
			predictions, _, state2 = self.model.forward(word_tensor, self.model.init_hidden(1))
			predictions = predictions.view(-1, len(self.model.dictionary))
			return (state2, 0, predictions)
	
	def update_state(self, ext_state, word):
		with torch.no_grad():
			(state, score, predictions) = ext_state
			word += ' '
			if word in self.tokens:
				word_tensor, target_tensor = self.tokens[word]
			else:
				new_tokens = [self.model.dictionary.get_idx_for_item(char) for char in word]
				word_tensor = torch.tensor(new_tokens).unsqueeze(1).to(flair.device)
				target_tensor = torch.tensor(new_tokens).to(flair.device)
				self.tokens[word] = (word_tensor, target_tensor)

			new_predictions, _, state2 = self.model.forward(word_tensor, state)
			new_predictions = new_predictions.view(-1, len(self.model.dictionary))
			all_predictions = torch.cat([predictions, new_predictions], dim = 0)
			
			word_predictions = torch.narrow(all_predictions, 0, 0, len(word))
			next_predictions = torch.narrow(all_predictions, 0, len(word), 1)

			new_score = score + torch.nn.CrossEntropyLoss(reduction='sum')(word_predictions, target_tensor).item()

			return (state2, new_score, next_predictions)

	def update_many_states(self, pre_states, words):
		with torch.no_grad():
			if len(pre_states) != len(words):
				raise "Length of states and words must be equal"
			len_and_key = []
			for i in range(len(words)):
				len_and_key.append((len(words[i]), i))
			
			states = []
			scores = [0 for _ in len_and_key]
			pred = []
			
			s0s = []
			s1s = []
			ps = []
			ws = []
			ks = []
			
			for (_, key) in sorted(len_and_key):
				try:
					(s0,s1), sc, p = pre_states[key]
				except:
					print (key)
					print (pre_states[key])
					raise "I dont know how to use this"
					
				w = words[key]
				
				s0s.append(s0)
				s1s.append(s1)
				ps.append(p.view(1,1,-1))
				ws.append(w)
				ks.append(key)
				pred.append(0)
				scores[key] = sc
				states.append(0)
			
			s0s = torch.cat(s0s, dim=1)
			s1s = torch.cat(s1s, dim=1)
			ps_t = torch.cat(ps, dim=1)
			
			while ws != []:
				if ws[0] == "":
					raise "This was not expected at all"

				sl = len(ws[0])
				words = [ word[:sl] for word in ws]
            
				tokens = torch.tensor([[self.model.dictionary.get_idx_for_item(char) for char in word] for word in words]).to(flair.device).transpose(0, 1)
				new_pred, _, (s0s, s1s) = self.model.forward(tokens, (s0s, s1s))
   
				e_pred_t = torch.cat([ps_t, new_pred], dim = 0)
            
				magic = e_pred_t[:-1,:,:].transpose(0,1).reshape(-1, len(self.model.dictionary))
				CEL = torch.nn.CrossEntropyLoss(reduction='none')(magic, tokens.transpose(0,1).reshape(-1)).reshape(-1,sl).sum(dim=1)
				for i in range(len(ws)): 
					 scores[ks[i]] +=  CEL[i]
					 ws[i] = ws[i][sl:]
				ps_t = e_pred_t[-1:,:,:]

				while ws != [] and ws[0] == "":
					 ws = ws[1:]
					 states[ks[0]] = (s0s[:,:1,:], s1s[:,:1,:])
					 s0s = s0s[:,1:,:]
					 s1s = s1s[:,1:,:]
					 pred[ks[0]] = ps_t[:,0,:]
					 ks = ks[1:]
					 ps_t = ps_t[:,1:,:]
			return [(states[i], scores[i].item(), pred[i]) for i in range(len(states))]

	def final_score(self, ext_state):
		with torch.no_grad():
			(state, score, predictions) = ext_state
			tokens = [self.model.dictionary.get_idx_for_item(char) for char in '.']
			final_tokens = torch.tensor(tokens).to(flair.device)
			return (torch.nn.CrossEntropyLoss(reduction='sum')(predictions, final_tokens).item() + score)

	def current_score(self, ext_state):
		with torch.no_grad():
			(state, score, predictions) = ext_state
			return score



class FlairBi_model:

    def __init__(self, model_name):
        self.forward_model = Flair_model(model_name + "-forward")
        self.backward_model = Flair_model(model_name + "-backward")


    def score_multi(self, sentences):
    	size = len(sentences)
    	total_length = max([len(sentence) for sentence in sentences])
    	padded_sentences = [(' ' + sentence + ' .' + (12+total_length)*"#")[:total_length+12] for sentence in sentences]
    	tokens = torch.cat(list(map(self._to_tokens, padded_sentences)), dim=1)
    	fwd_predictions, _, _ = self.forward_model.forward(tokens, self.forward_model.init_hidden(size))
    	bwd_predictions, _, _ = self.backward_model.backward(tokens, self.backward_model.init_hidden(size))

    	score_m = [self._pred_to_score(sentences[i] + ' .', predictions.narrow(1, i, 1)) for i in range(size)]
    	#score_s = [self.score(sentence) for sentence in sentences]
    
    	return score_m
    
class Polbert_model(object):
	def __init__(self, model_name):
		self.name = model_name
		self.max_batch_size = 30

	def start(self, find_ignored = False):
		with torch.no_grad():
			self.model = transformers.BertForMaskedLM.from_pretrained(self.name)
			self.tokenizer = transformers.BertTokenizer.from_pretrained(self.name)
			self.nlp = transformers.pipeline('fill-mask', model=self.model, tokenizer=self.tokenizer, device=0)

			if not find_ignored:
				self.ignored = []
			else:
				def keep(x):
					if x in ['[UNUSED_%d]'%i for i in range(256)]:
						return True
					if all([(c in 'qazxswedcvfrtgbnhyujmkiolp1234567890ążźęśćńół') for c in x ]):
						return True
					if x[:2] == '##' and all([(c in 'qazxswedcvfrtgbnhyujmkiolp1234567890ążźęśćńół') for c in x[2:] ]):
						return True
					if x in ['[PAD]', '[UNK]', '[CLS]', '[MASK]', '[SEP]']: return True
					return False
				dic = self.nlp.tokenizer.get_vocab()
				self.ignored = [dic[key] for key in dic.keys() if not keep(key)]
				

	def per_word_score(self, sentence):
		with torch.no_grad():
			sentence = sentence.strip().lower().replace('<unk>', '[UNK]')
			if sentence == "":
				return []
			tokens = self.nlp._parse_and_tokenize([sentence])
			n_tokens = len(tokens['input_ids'][0])
			keys = tokens.keys()

			def empty_batch():
				r = {key:[] for key in keys}
				r['masked_id'] = []
				r['masked_token'] = []
				return r

			batches = [empty_batch ()]

			for i in range(n_tokens)[1:-1]:
				if len(batches[-1]['masked_id']) == self.max_batch_size:
					batches.append(empty_batch ())

				for key in keys:
					batches[-1][key].append(tokens[key][:1])
				batches[-1]['masked_id'].append(i)
				batches[-1]['masked_token'].append(tokens['input_ids'][0][i].item())
				batches[-1]['input_ids'][-1] = torch.cat([ tokens['input_ids'][:1,:i], torch.tensor([[ self.nlp.tokenizer.convert_tokens_to_ids(self.nlp.tokenizer.mask_token) ]]), tokens['input_ids'][:1,i+1:]], dim=1)
	
			def finalize_batch(batch):
				for key in keys:
					batch[key] = torch.cat(batch[key], dim=0)
				return batch
			batches = [finalize_batch(batch) for batch in batches]

			score = []
			for batch in batches:
				masked_tokens = batch['masked_token']
				masked_ids = batch['masked_id']
				outputs = self.nlp._forward({key:batch[key] for key in keys}, return_tensors=True)

				for b in range(len(masked_ids)):
					logits = outputs[b, masked_ids[b], :].to('cpu')
					logits[self.ignored] = -1e12
					probs = logits.softmax(dim=0)
					target_prob = probs[masked_tokens[b]]
					try:
						token_score = math.log(target_prob)/math.log(10)
					except:
						print ('error. Target_prob = %f. masked_token_id = %d, masked_token "%s"'%(target_prob, masked_tokens[b], self.nlp.tokenizer.convert_ids_to_tokens(masked_tokens[b])))
						token_score = math.log(target_prob)/math.log(10)
					score.append(-token_score)
					#print (token_score, '', masked_tokens[b], self.nlp.tokenizer.convert_ids_to_tokens(masked_tokens[b]))

			return score			

	def score(self, sentence):
		return sum(self.per_word_score(sentence))

	def score_words(self, examples):
		with torch.no_grad():
			info = []
			sentences = []
			score = [ 0 for i in range(len(examples))]
		
			example_ind = -1
			for (pref, word, suf) in examples:
				pref = pref.strip().lower().replace('<unk>', '[UNK]')
				suf = suf.strip().lower().replace('<unk>', '[UNK]')
				word = word.strip().lower().replace('<unk>', '[UNK]')
				example_ind += 1

				pref_len = len(self.nlp.tokenizer.tokenize(pref))
				word_len = len(self.nlp.tokenizer.tokenize(word))

				for i in range(word_len):
					info.append(((pref_len+i), example_ind))

				for i in range(word_len):
					sentence = pref + ' ' + word + ' ' + suf
					sentences.append(sentence)

			while sentences != []:
				batch = sentences[:self.max_batch_size]
				batch_info = info[:self.max_batch_size]
				info = info[self.max_batch_size:]
				sentences = sentences[self.max_batch_size:]

				tokens = self.nlp._parse_and_tokenize(batch)
				try:
					expected = [ tokens['input_ids'][i, 1+batch_info[i][0]].item() for i in range(len(batch_info)) ]
				except:
					print (tokens)
					raise "What to put here anyway??"

				for i in range(len(batch_info)):
					tokens['input_ids'][i, 1+batch_info[i][0]] = self.nlp.tokenizer.convert_tokens_to_ids(self.nlp.tokenizer.mask_token)

				outputs = self.nlp._forward(tokens, return_tensors=True)

				for i in range(len(batch_info)):
					logits = outputs[i, 1+batch_info[i][0],:]
					logits[self.ignored] = -1e12
					probs = logits.softmax(dim=0)
					target_prob = probs[expected[i]]

					token_score = math.log(target_prob)/math.log(10)

					score[batch_info[i][1]] -= token_score

			return score


		

if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser ()
	parser.add_argument ('-reference', help='File with reference sentences with keys')
	parser.add_argument ('-score-flair', help='File with reference sentences with keys', type = bool)
	args=parser.parse_args()

	print ("hello")
	sentence = "pan minister chyba wie jaka ustawa jest najlepsza dla populacji jeleni"
	print(sentence)

	polbert = Polbert_model('dkleczek/bert-base-polish-uncased-v1')
	polbert.start()
	t = time.time()
	print("polbert", polbert.score(sentence))
	print (time.time() - t)

	polbert = Polbert_model('dkleczek/bert-base-polish-uncased-v1')
	polbert.start(find_ignored=True)
	t = time.time()
	print("polbert", polbert.score(sentence))
	print (time.time() - t)

	#flair_pl = Flair_model('pl-forward')
	#flair_pl.start()
	#print('flair', flair_pl.score(sentence))

	if args.reference is not None:
		text = [' '.join(line.strip().split(' ')[1:]) for line in open(args.reference, 'r')]
		
		if args.score_flair is not None:
			st = time.time()
			x = [ flair_pl.score(sentence) for sentence in text ]
			print (time.time() - st)

		st = time.time()
		x = [ polbert.score(sentence) for sentence in text[:5]]
		print (time.time() - st)
		print (x)
		

