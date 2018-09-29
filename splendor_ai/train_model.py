import sys
import os
abspath = '/'.join(os.path.abspath(__file__).split('/')[:-1]) + '/'
under_abspath = '/'.join(os.path.abspath(__file__).split('/')[:-2])

sys.path.insert(0, under_abspath)
from environment import splendor

import json
import numpy as np
import itertools as it
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.engine import InputLayer
from keras.layers import Dense
from keras.models import model_from_json

import json
import time
import math
from copy import deepcopy
from pprint import pprint
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if not os.path.isdir(abspath + 'brains'):
	os.mkdir(abspath + 'brains')

if not os.path.isfile(abspath + 'brains.json'):
	file = open(abspath + 'brains.json', 'w')
	file.write('{}')
	file.close()
else:
	try:
		file = open(abspath + 'brains.json', 'r')
		json.loads(file.read())
		file.close()
	except Exception as e:
		print('could not read brains.json error: ' + str(e))
		file = open(abspath + 'brains.json', 'w')
		file.write('{}')
		file.close()

pick_tokens = {}
pick_tokens['3'] = list(it.combinations([0,1,2,3,4], 3))
pick_tokens['2'] = list(it.combinations([0,1,2,3,4], 2))
pick_tokens['doubles'] = [0,1,2,3,4]
pick_tokens['1'] = [0,1,2,3,4]
#pick_tokens['0'] = [-1]

input_nodes = sum([
	#========
	#BOARD
	#========
	6,			# Tokens
	3 * 4 * 7,	# Cards (3 tiers, 4 cards each, 7 nodes per card)
	5 * 5,		# Nobles
	3,			# Hidden cards
	#========
	#PLAYERS
	#========
	4 * sum([
		6,		# Tokens
		5,		# Cards
		3 * 7,	# Reservations
		1		# Score
	])
])

def print_players(players):
	for idx in range(len(players)):
		print('player' + str(idx+1))
		pprint(players[idx])
		print()

def print_state(state, clear=True):
	if clear:
		os.system('clear')

	columns_to_show = [
		'value',
		'type'
	] + env.colors
	print(str(state['tier3'][columns_to_show]).splitlines()[0])
	for i in str(state['tier3'][columns_to_show]).splitlines()[1:]: print(i)
	for i in str(state['tier2'][columns_to_show]).splitlines()[1:]: print(i)
	for i in str(state['tier1'][columns_to_show]).splitlines()[1:]: print(i)
	print(	'T1 hidden cards: ' + str(state['hidden_t1']) + 
			'\tT2 hidden cards: ' + str(state['hidden_t2']) + 
			'\tT3 hidden cards: ' + str(state['hidden_t3']))
	print('Boards tokens: ' + str(state['tokens']))
	print('Nobels:\n' + state['nobels'].to_string(index=False))
	print()
	print_players(state['players'])

def state_to_nodes(state):
	return_nodes = np.zeros(input_nodes)

	sorted_tokens = [value for (key, value) in sorted(list(state['tokens'].items()))]
	return_nodes[:6] = sorted_tokens
	
	tier1 = [i[1:] for i in state['tier1'].values]
	if len(tier1) > 0:
		tier1 = np.concatenate(tier1, axis=0)
		return_nodes[6:6+len(tier1)] = tier1
	#return_nodes[6:34] = np.concatenate(tier1, axis=0)

	tier2 = [i[1:] for i in state['tier2'].values]
	if len(tier2) > 0:
		tier2 = np.concatenate(tier2, axis=0)
		return_nodes[34:34+len(tier2)] = tier2
	#return_nodes[34:62] = 
	
	tier3 = [i[1:] for i in state['tier3'].values]
	if len(tier3) > 0:
		tier3 = np.concatenate(tier3, axis=0)
		return_nodes[62:62+len(tier3)] = tier3
	#90
	
	nobles = state['nobels'].values
	if len(nobles) > 0:
		nobles = np.concatenate(nobles, axis=0)
		return_nodes[90:90+len(nobles)] = nobles

	return_nodes[115] = state['hidden_t1']
	return_nodes[116] = state['hidden_t2']
	return_nodes[117] = state['hidden_t3']

	for player in range(4):
		begin = 118 + player*33
		
		sorted_tokens = [value for (key, value) in sorted(list(state['players'][player]['tokens'].items()))]
		return_nodes[begin: begin+6] = sorted_tokens

		sorted_tokens = [value for (key, value) in sorted(list(state['players'][player]['cards'].items()))]
		return_nodes[begin+6: begin+11] = sorted_tokens

		for idx, i in enumerate(state['players'][player]['reservations']):
			this_begin = begin+11+idx*7
			this_reservation = i.values.tolist()[0][1:]
			return_nodes[this_begin: this_begin+7] = this_reservation

		return_nodes[begin+32] = state['players'][player]['score']

	return return_nodes

def combination_to_tokens(combination):
	to_pick = {c: 0 for c in COLORS}
	for color_idx in combination:
		to_pick[COLORS[color_idx]] += 1
	return to_pick

def step(move, state, env, print_move=False):
	if move in range(30):
		if move in range(10):
			combination = pick_tokens['3'][move]
			tokens = combination_to_tokens(combination)

		elif move in range(10, 20):
			combination = pick_tokens['2'][move%10]
			tokens = combination_to_tokens(combination)

		elif move in range(20, 25):
			combination = pick_tokens['doubles'][move%5]
			tokens = {COLORS[combination]: 2}

		elif move in range(25, 30):
			combination = pick_tokens['1'][move%5]
			tokens = {COLORS[combination]: 1}

		move = {'pick': tokens}

	elif move in range(30, 54):
		move -= 30

		if move in range(8):
			idx = int(move/2)
			card = state['tier1'][idx: idx + 1]
		elif move in range(8, 16):
			move -= 8
			idx = int(move/2)
			card = state['tier2'][idx: idx + 1]
		elif move in range(16, 24):
			move -= 16
			idx = int(move/2)
			card = state['tier3'][idx: idx + 1]

		if move % 2 == 0:
			# Buy card
			move = {'buy': card}
		else:
			# Reserve card
			move = {'reserve': card}

	elif move in range(54, 57):
		card = state['players'][0]['reservations'][move-54]
		move = {'buy': card}

	elif move in range(57, 87):
		if move in range(57, 67):
			combination = pick_tokens['3'][move-57]
			tokens = combination_to_tokens(combination)

		elif move in range(67, 77):
			combination = pick_tokens['2'][(move-57)%10]
			tokens = combination_to_tokens(combination)

		elif move in range(77, 82):
			combination = pick_tokens['doubles'][(move-57)%5]
			tokens = {COLORS[combination]: 2}

		elif move in range(82, 87):
			combination = pick_tokens['1'][(move-57)%5]
			tokens = {COLORS[combination]: 1}

		move = {'return': tokens}

	elif move == 87:
		move = {'pick': {}}		

	else:
		print('invalid move: ' + str(move))
	
	if print_move:
		print(move)
		return
	return env.move(move)

def choose_action(s, model, env):
	aval_vector = env.avaliable_outputs()
	if sum(aval_vector) != 1:
		aval_vector[-1] = 0

	if np.random.random() < this_brain['eps']:
		raw_prediction = np.random.rand(1, output_nodes)

	else:
		x = np.array(state_to_nodes(s)).reshape(1, input_nodes)
		raw_prediction = model.predict(x)
				
	prediction = raw_prediction * aval_vector
	a = np.argmax(prediction[0])
	if prediction[0][a] <= 0:
		not0 = prediction != 0
		a = np.argmax(not0)

	return a

def reward(state, last_state):
	me = state['players'][0]
	last_me = last_state['players'][0]
	score_difference = me['score'] - last_me['score']
	cards_difference = sum(me['cards'].values()) - sum(last_me['cards'].values())
	good_boy = cards_difference + score_difference**2
	if good_boy != 0:
		return good_boy
	return -.4

def dump_model(model, model_name):
	# serialize model to JSON
	#model_json = model.to_json()
	#with open(abspath + model_name + ".json", "w") as json_file:
	#	json_file.write(model_json)
	
	# serialize weights to HDF5
	model.save_weights(abspath + 'brains/' + model_name + ".h5")
	print("Saved model to disk {}".format(abspath + 'brains/' + model_name))

def load_model(model_name):
	# load json and create model
	#print("Loading model from disk {}".format(abspath + model_name))
	#json_file = open(abspath + model_name + '.json', 'r')
	#loaded_model_json = json_file.read()
	#json_file.close()
	#loaded_model = model_from_json(loaded_model_json)
	hidden_layer_size = int((input_nodes+output_nodes) / 2)
	loaded_model = Sequential()
	loaded_model.add(InputLayer(batch_input_shape=(1, input_nodes)))
	loaded_model.add(Dense(hidden_layer_size, activation='sigmoid'))
	loaded_model.add(Dense(hidden_layer_size, activation='sigmoid'))
	loaded_model.add(Dense(output_nodes, activation='linear'))
	# load weights into new model
	loaded_model.load_weights(abspath + 'brains/' + model_name + ".h5")
	loaded_model.compile(loss='mse', optimizer='adam', metrics=['mae'])
	#print("Loaded model from disk")

	return loaded_model

def create_model(hidden_layer_size, output_nodes):
	model = Sequential()
	model.add(InputLayer(batch_input_shape=(1, input_nodes)))
	model.add(Dense(hidden_layer_size, activation='sigmoid'))
	model.add(Dense(hidden_layer_size, activation='sigmoid'))
	model.add(Dense(output_nodes, activation='linear'))
	model.compile(loss='mse', optimizer='adam', metrics=['mae'])
	return model

def save(brains_json):
	file = open(abspath + 'brains.json', 'w')
	file.write(json.dumps(brains_json, indent=4))
	file.close()


if __name__ == '__main__':
	env = splendor.Splendor()

	COLORS = env.colors
	output_nodes = env.output_nodes

	hidden_layer_size = int((input_nodes+output_nodes) / 2)

	if len(sys.argv) == 1:
		model_name = 'model'
	else:
		model_name = sys.argv[1]
	
	file = open(abspath + 'brains.json', 'r')
	brains_info = json.loads(file.read())
	file.close()

	if model_name in brains_info and os.path.isfile(abspath + 'brains/' + model_name + '.h5'):
		print('loading model "{}"'.format(model_name))
		h5_path = brains_info[model_name]['h5_path']

		model = load_model(model_name)
		dumb_models = [load_model(model_name) for _ in range(4)]

	else:
		print('creating model "{}"'.format(model_name))
		brains_info[model_name] = {
			'h5_path': abspath + 'brains/' + model_name,
			'num_episodes': 100,
			'episodes_done': 0,
			'y': 0.95,
			'eps': 0.4,
			'decay_factor': 0.98,
			'renew_dumb_ai': 400
		}

		if len(sys.argv) > 2:
			argument_keys = ['num_episodes', 'episodes_done', 'y', 'eps', 'decay_factor', 'renew_dumb_ai']
			for idx, argument in enumerate(sys.argv[2:]):
				arg_idx = argument_keys[idx]
				current_argument = brains_info[model_name][arg_idx]

				brains_info[model_name][arg_idx] = type(current_argument)(argument)

		model = create_model(hidden_layer_size, output_nodes)
		dumb_models = [create_model(hidden_layer_size, output_nodes) for _ in range(4)]
		save(brains_info)

	this_brain = brains_info[model_name]
	
	r_avg_list = []
	games = []


	num_episodes = this_brain['num_episodes'] - this_brain['episodes_done']

	training_s = 15*num_episodes
	training_m = int(training_s/60)
	training_h = int(training_m/60)

	if training_h > 0:
		training_time = str(training_h) + 'h:' + str(training_m%60) + 'm'
	else:
		training_time = str(training_m) + 'm'

	finish_time = datetime.utcfromtimestamp(60*60*2 + time.time() + training_s).strftime('%Y-%m-%d %H:%M:%S')

	print('starting training at: ' + datetime.utcfromtimestamp(60*60*2 + time.time()).strftime('%Y-%m-%d %H:%M:%S'))
	print('estimated training time: ' + training_time + ', finish prediction: ' + finish_time)

	for i in range(this_brain['episodes_done'], this_brain['num_episodes']):
		s = deepcopy(env.reset())
		this_brain['eps'] *= this_brain['decay_factor']
		if i % 100 == 0:
			print("Episode {} of {}".format(i + 1, num_episodes))
		done = False
		r_sum = 0
		r = 0
		game_start = time.time()
		game_round = 0

		while not done:
			game_round += 1
			mr_87 = 0

			while True:
				a = choose_action(s, model, env)
				try:
					new_s = step(a, s, env)
				except:
					print_state(s)
					step(a, s, env, True)
					1/0

				if not new_s['return_tokens']:
					break

			if a == 87:
				mr_87 += 1

			for dumb_model in dumb_models:
				while True:
					dumb_a = choose_action(new_s, dumb_model, env)
					try:
						new_s = step(dumb_a, new_s, env)
					except:
						print_state(new_s)
						step(dumb_a, new_s, env, True)
						1/0

					if not new_s['return_tokens']:
						break
				if dumb_a == 87:
					mr_87 += 1

			r = reward(new_s, s)

			done = new_s['end']
			aval_vector = env.avaliable_outputs()
			new_x = np.array(state_to_nodes(new_s)).reshape(1, input_nodes)
			target = r + this_brain['y'] * np.max(model.predict(new_x)*aval_vector)
			target_vec = model.predict(new_x)[0]
			target_vec[a] = target
			model.fit(new_x, target_vec.reshape(-1, output_nodes), epochs=1, verbose=0)
			s = deepcopy(new_s)
			r_sum += r

			if mr_87 == 4 or game_round > 60:
				break

		if any(i['score'] >= 15 for i in s['players']):
			#print('final score: ' + str([i['score'] for i in s['players']]))
			if s['players'][0]['score'] >= 15:
				games.append(1)
				#print('game ' + str(i+1) + ' status: won')
			else:
				games.append(0)
				#print('game ' + str(i+1) + ' status: lost')

			game_time = int(time.time()-game_start)
			finish_time = str(i+1) + ' game finished in: ' + str(game_time) + 's'

			last_20 = games[-min(len(games), 20):]
			games20_summary = str(sum(last_20)) + '/' + str(len(last_20))
			games20_average = str(int(sum(last_20)/len(last_20)*100)) + '%'
			
			games_summary = str(sum(games)) + '/' + str(len(games))
			games_average = str(int(sum(games)/len(games)*100)) + '%'
			
			final_score = str([i['score'] for i in s['players']])
			
			print(finish_time + ', ' + \
				games20_summary + ' of last 20 games won that is ' + games20_average + \
				', ' + games_summary + ' games won that is ' + games_average + \
				', final score: ' + final_score)
		
		else:
			#print_state(new_s, False)
			print(str(i+1) + ' game finished because of 4 empty moves')
		
		if i%this_brain['renew_dumb_ai'] == 0 and i != 0:
			dump_model(model, model_name)
			dumb_models = [load_model(model_name) for _ in range(4)]
			this_brain['episodes_done'] = i
			save(this_brain)
		
		r_avg_list.append(r_sum)

		if os.path.isfile('stop'):
			print('stopping')
			break

	chunk = 10
	r_mean = []
	for i in range(int(math.ceil(len(r_avg_list)/chunk))):
		r_mean.append(np.mean(r_avg_list[chunk*i: min(chunk*(i+1), len(r_avg_list))]))

	plt.plot([i*chunk for i in range(len(r_mean))], r_mean)
	plt.show()

	dump_model(model, model_name)
	this_brain['episodes_done'] = i
	save(this_brain)