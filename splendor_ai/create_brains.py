import os

def combine(x):
	so_far = []
	for i in x[0]:
		if len(x) > 1:
			for j in combine(x[1:]):
				so_far.append(' '.join([i, j]))
		else:
			so_far.append(i)
	return so_far

args = [
	['100'],
	['0'],
	['0.5', '1'],
	['0.4', '0.8'],
	['0.97', '0.99']
]

for idx, arg in enumerate(combine(args)):
	os.system('python3 train_model.py {} {}'.format(idx, arg))