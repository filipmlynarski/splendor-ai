import sys
sys.path.insert(0, '..')

from environment import splendor

import numpy as np
import itertools as it
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.engine import InputLayer
from keras.layers import Dense
from keras.models import model_from_json

from pprint import pprint
from copy import deepcopy
import time
from datetime import datetime
import math

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

pick_tokens = {}
pick_tokens['3'] = list(it.combinations([0,1,2,3,4], 3))
pick_tokens['2'] = list(it.combinations([0,1,2,3,4], 2))
pick_tokens['doubles'] = [0,1,2,3,4]
pick_tokens['1'] = [0,1,2,3,4]
pick_tokens['0'] = [-1]

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
output_nodes = sum([
	#TOKEN PICKING
	sum(len(i) for i in pick_tokens.values()),
	
	#BUYING AND RESERVING BOARD CARDS
	3 * 4 * 2,
	
	#BUYING RESERVED CARDS
	3
])

COLORS = ['green', 'white', 'blue', 'black', 'red']

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

def avaliable_outputs(state, env):
	return_nodes = np.zeros(output_nodes)

	player_tokens = state['players'][0]['tokens']

	if sum(player_tokens.values()) <= 7:
		for idx, combination in enumerate(pick_tokens['3']):
			if all(state['tokens'][COLORS[i]] >= 1 for i in combination):
				return_nodes[idx] = 1

	if sum(player_tokens.values()) <= 8:
		for idx, combination in enumerate(pick_tokens['2']):
			if all(state['tokens'][COLORS[i]] >= 1 for i in combination):
				return_nodes[10 + idx] = 1

		for idx, combination in enumerate(pick_tokens['doubles']):
			if state['tokens'][COLORS[combination]] >= 2:
				return_nodes[20 + idx] = 1

	if sum(player_tokens.values()) <= 9:
		for idx, combination in enumerate(pick_tokens['1']):
			if state['tokens'][COLORS[combination]] >= 1:
				return_nodes[25 + idx] = 1

	player_reservations = state['players'][0]['reservations']
	can_reserve = len(player_reservations) < 3 and state['tokens']['gold'] > 0

	for card in range(len(state['tier1'])):
		this_card = state['tier1'][card: card+1]
		if env.can_afford(this_card):
			return_nodes[30 + card*2] = 1
		if can_reserve:
			return_nodes[31 + card*2] = 1

	for card in range(len(state['tier2'])):
		this_card = state['tier2'][card: card+1]
		if env.can_afford(this_card):
			return_nodes[38 + card*2] = 1
		if can_reserve:
			return_nodes[39 + card*2] = 1

	for card in range(len(state['tier3'])):
		this_card = state['tier3'][card: card+1]
		if env.can_afford(this_card):
			return_nodes[46 + card*2] = 1
		if can_reserve:
			return_nodes[47 + card*2] = 1

	for idx, reservation in enumerate(player_reservations):
		if env.can_afford(reservation):
			return_nodes[54+idx] = 1

	return_nodes[-1] = 1

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

	elif move == 57:
		move = {'pick': {}}

	else:
		print('invalid move: ' + str(move))
	
	if print_move:
		print(move)
		return
	return env.move(move)

def reward(state, a, last_state):
	me = state['players'][0]
	last_me = last_state['players'][0]
	score_difference = me['score'] - last_me['score']
	cards_difference = sum(me['cards'].values()) - sum(last_me['cards'].values())
	good_boy = cards_difference + score_difference**2
	if good_boy != 0:
		return good_boy
	return -.2

def dump_model(model, model_name):
	# serialize model to JSON
	model_json = model.to_json()
	with open(model_name + ".json", "w") as json_file:
		json_file.write(model_json)
	
	# serialize weights to HDF5
	model.save_weights(model_name + ".h5")
	print("Saved model to disk")

def load_model(model_name):
	# load json and create model
	json_file = open(model_name + '.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	
	# load weights into new model
	loaded_model.load_weights(model_name + ".h5")
	loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	#print("Loaded model from disk")

	return loaded_model

def create_model(hidden_layer_size, output_nodes):
	model = Sequential()
	model.add(InputLayer(batch_input_shape=(1, input_nodes)))
	model.add(Dense(hidden_layer_size, activation='sigmoid'))
	model.add(Dense(output_nodes, activation='linear'))
	model.compile(loss='mse', optimizer='adam', metrics=['mae'])
	return model


if __name__ == '__main__':
	env = splendor.Splendor()
	COLORS = env.colors

	hidden_layer_size = int((input_nodes+output_nodes) / 2)

	if len(sys.argv) == 1:
		model_name = 'model'
	else:
		model_name = sys.argv[1]
	
	if os.path.isfile(model_name + '.h5') and os.path.isfile(model_name + '.json'):
		print('loading model "{}"'.format(model_name))
		model = load_model(model_name)
		dumb_model = load_model(model_name)
	else:
		print('creating model "{}"'.format(model_name))
		model = create_model(hidden_layer_size, output_nodes)
		dumb_model = create_model(hidden_layer_size, output_nodes)

	num_episodes = 4000
	y = 0.95
	eps = 0.95
	decay_factor = 0.999
	r_avg_list = []
	games = []
	renew_dumb_ai = 500

	training_s = 11*num_episodes
	training_m = int(training_s/60)
	training_h = int(training_m/60)

	if training_h > 0:
		training_time = str(training_h) + 'h:' + str(training_m%60) + 'm'
	else:
		training_time = str(training_m) + 'm'

	finish_time = datetime.utcfromtimestamp(60*60*2 + time.time() + training_s).strftime('%Y-%m-%d %H:%M:%S')

	print('starting training at: ' + datetime.utcfromtimestamp(60*60*2 + time.time()).strftime('%Y-%m-%d %H:%M:%S'))
	print('estimated training time: ' + training_time + ', finish prediction: ' + finish_time)

	for i in range(num_episodes):
		s = deepcopy(env.reset())
		eps *= decay_factor
		if i % 100 == 0:
			print("Episode {} of {}".format(i + 1, num_episodes))
		done = False
		r_sum = 0
		r = 0
		game_start = time.time()
		inv = False

		while not done:
			mr_57 = 0
			aval_vector = avaliable_outputs(s)
			if sum(aval_vector) != 1:
				aval_vector[-1] = 0

			if np.random.random() < eps:
				raw_prediction = np.random.rand(1, output_nodes)

			else:
				x = np.array(state_to_nodes(s)).reshape(1, 250)
				raw_prediction = model.predict(x)
				
			prediction = raw_prediction * aval_vector
			#prediction[0][-1] = -100
			a = np.argmax(prediction[0])
			#print('not zeros:', sum(prediction[0] != 0))
			#print('first action:', a)
			if prediction[0][a] <= 0:
				not0 = prediction != 0
				a = np.argmax(not0)
				#a = np.argmax(_)
			#print('scecord action:', a)

			if a == 57:
				mr_57 += 1

			try:
				new_s = step(a, s)
			except Exception as e:
				print(1)
				print(str(e))
				print(s)
				print(a)
				print(step(a, s, print_move=True))
				#sys.exit()
				break


			for dumb_move in range(3):
				aval_vector = avaliable_outputs(new_s)
				if sum(aval_vector) != 1:
					aval_vector[-1] = 0

				if np.random.random() < eps:
					raw_prediction = np.random.rand(1, output_nodes)

				else:
					x = np.array(state_to_nodes(new_s)).reshape(1, 250)
					raw_prediction = dumb_model.predict(x)

				prediction = raw_prediction * aval_vector
				#prediction[0][-1] = -100
				dumb_a = np.argmax(prediction[0])
				#print('not zeros:', sum(prediction[0] != 0))
				#print('first action:', dumb_a)
				if prediction[0][dumb_a] <= 0:
					not0 = prediction != 0
					dumb_a = np.argmax(not0)
				#print('second action:', dumb_a)

				if dumb_a == 57:
					mr_57 += 1

				try:
					new_s = step(dumb_a, new_s)
				except Exception as e:
					print(2)
					print(str(e))
					print(new_s)
					print(dumb_a)
					print(step(dumb_a, new_s, print_move=True))
					#sys.exit()
					inv = True
					break

			if inv:
				break

			r = reward(new_s, a, s)
			#print(r)

			done = new_s['end']
			aval_vector = avaliable_outputs(new_s)
			new_x = np.array(state_to_nodes(new_s)).reshape(1, 250)
			target = r + y * np.max(model.predict(new_x)*aval_vector)
			target_vec = model.predict(new_x)[0]
			target_vec[a] = target
			model.fit(new_x, target_vec.reshape(-1, output_nodes), epochs=1, verbose=0)
			s = deepcopy(new_s)
			r_sum += r

			if mr_57 == 4:
				break

		if any(i['score'] >= 15 for i in s['players']):
			#print('final score: ' + str([i['score'] for i in s['players']]))
			if s['players'][0]['score'] >= 15:
				games.append(1)
				#print('game ' + str(i+1) + ' status: won')
			else:
				games.append(0)
				#print('game ' + str(i+1) + ' status: lost')

			finish_time = str(i+1) + ' game finished in: ' + str(int(time.time()-game_start)) + 's'

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
		
		if i%renew_dumb_ai == 0 and i != 0:
			dump_model(model, model_name)
			dumb_model = load_model(model_name)
		
		r_avg_list.append(r_sum)

		if os.path.isfile('stop'):
			break

	chunk = 5
	r_mean = []
	for i in range(int(math.ceil(len(r_avg_list)/chunk))):
		r_mean.append(np.mean(r_avg_list[chunk*i: min(chunk*(i+1), len(r_avg_list))]))

	plt.plot([i*chunk for i in range(len(r_mean))], r_mean)
	plt.show()

	dump_model(model, model_name)