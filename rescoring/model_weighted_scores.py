if __name__ == "__main__":
	import argparse
	import numpy as np
	
	parser = argparse.ArgumentParser()

	parser.add_argument ('-scores', help="File with scores", required=True)

	args = parser.parse_args ()
	
	all_lines = [line.strip() for line in open(args.scores)]

	if all_lines[0] != "key, lm, acc, wer":
		raise NameError("Order of columns is wrong: " + all_lines[0])

	source = np.zeros((500, (len(all_lines)-1)//500))
	print (source.shape)

	keys, lms, accs, wers = [], [], [], []
	for line in all_lines [1:]:
		key, lm, acc, wer = line.split(',')
		keys.append(key)
		lms.append(float(lm))
		accs.append(float(acc))
		wers.append(float(wer))
	
	keys = np.reshape (np.array(keys), (-1, 500))
	lms = np.reshape (np.array(lms), (-1, 500))
	accs = np.reshape (np.array(accs), (-1, 500))
	wers = np.reshape (np.array(wers), (-1, 500))

	print (np.average(np.min(wers, axis=1)))

	row_indexes = np.array([i for i in range(np.shape(keys)[0])])
	ans = (123123123, -1)
	for a in [ x * 0.001 for x in range(-1000000, 1000001, 25)]:
		res = np.average(wers[row_indexes, np.argmin(lms*a + accs, axis=1)])
		print (a, res)
		ans = min(ans, (res, a))
	print (ans)
		
