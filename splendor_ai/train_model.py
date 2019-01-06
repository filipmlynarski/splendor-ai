import sys
import os
abspath = '/'.join(os.path.abspath(__file__).split('/')[:-1]) + '/'
under_abspath = '/'.join(os.path.abspath(__file__).split('/')[:-2])

sys.path.insert(0, under_abspath)
from environment import splendor
from print_board import PrintBoard

import json
import numpy as np
import itertools as it
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.engine import InputLayer
from keras.layers import Dense
from keras.models import model_from_json
from keras.layers import LeakyReLU

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
	3 * 4 * 11,	# Cards (3 tiers, 4 cards each, 11 nodes per card)
	5 * 5,		# Nobles
	3,			# Hidden cards
	#========
	#PLAYERS
	#========
	4 * sum([
		6,		# Tokens
		5,		# Cards
		3 * 11,	# Reservations
		1		# Score
	])
])
#input_nodes = 966
input_nodes = 1312

output_nodes = 88
output_nodes = 13

'''
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
	print('Nobels:\n' + state['nobles'].to_string(index=False))
	print()
	print_players(state['players'])
'''

def state_to_nodes(state):
	return_nodes = np.zeros(input_nodes)

	sorted_tokens = [value for (key, value) in sorted(list(state['tokens'].items()))]
	return_nodes[:6] = sorted_tokens
	
	tier1 = [i[1:] for i in state['tier1'].values]
	if len(tier1) > 0:
		tier1 = np.concatenate(tier1, axis=0)
		tier1 = tier_to_vector2(tier1)

		return_nodes[6:6+len(tier1)] = tier1

	tier2 = [i[1:] for i in state['tier2'].values]
	if len(tier2) > 0:
		tier2 = np.concatenate(tier2, axis=0)
		tier2 = tier_to_vector2(tier2)

		return_nodes[34:34+len(tier2)] = tier2
		return_nodes[50:50+len(tier2)] = tier2
	
	tier3 = [i[1:] for i in state['tier3'].values]
	if len(tier3) > 0:
		tier3 = np.concatenate(tier3, axis=0)
		tier3 = tier_to_vector2(tier3)

		return_nodes[62:62+len(tier3)] = tier3
		return_nodes[94:94+len(tier3)] = tier3
	
	nobles = state['nobles'].values
	if len(nobles) > 0:
		nobles = np.concatenate(nobles, axis=0)
		return_nodes[138:138+len(nobles)] = nobles

	return_nodes[163] = state['hidden_t1']
	return_nodes[164] = state['hidden_t2']
	return_nodes[165] = state['hidden_t3']

	for player in range(4):
		begin = 166 + player*45
		
		sorted_tokens = [value for (key, value) in sorted(list(state['players'][player]['tokens'].items()))]
		return_nodes[begin: begin+6] = sorted_tokens

		sorted_tokens = [value for (key, value) in sorted(list(state['players'][player]['cards'].items()))]
		return_nodes[begin+6: begin+11] = sorted_tokens

		for idx, i in enumerate(state['players'][player]['reservations']):
			this_begin = begin+11+idx*11
			this_reservation = i.values.tolist()[0][1:]
			this_reservation = card_to_vector2(this_reservation)

			return_nodes[this_begin: this_begin+11] = this_reservation

		return_nodes[begin+44] = state['players'][player]['score']

	'''print_state(state)
	
	print('GAME TOKENS')
	print(return_nodes[:6])

	print('CARDS')
	for node_card in range(int(84/7)):
		print(return_nodes[6 + node_card*7: 6 + (node_card+1) * 7])

	print('NOBELS')
	print(return_nodes[90:115])

	print('HIDDEN CARDS')
	print(return_nodes[115:118])

	print('PLAYERS')
	print(return_nodes[118:150])
	1/0'''
	return return_nodes

def state_to_nodes2(state):
	return_nodes = np.zeros(input_nodes)

	return_nodes[:40] = tokens_to_vector(state['tokens'])

	players_cards = []
	for player in state['players']:
		tokens_with_keys = sorted(player['tokens'].items(), key=lambda x: x[0])
		players_cards.append([i[1] for i in tokens_with_keys if i[0] != 'gold'])
	players_cards[0] = [1,1,1,1,1]
	tier1 = [i[1:] for i in state['tier1'].values]
	if len(tier1) > 0:
		tier1 = np.concatenate(tier1, axis=0)
		#tier1 = subtract_cards(tier1, players_cards[0])
		tier1 = tier_to_vector(tier1)

		return_nodes[6:6+len(tier1)] = tier1

	tier2 = [i[1:] for i in state['tier2'].values]
	if len(tier2) > 0:
		tier2 = np.concatenate(tier2, axis=0)
		#tier2 = subtract_cards(tier2, players_cards[0])
		tier2 = tier_to_vector(tier2)

		return_nodes[170:170+len(tier2)] = tier2
	
	tier3 = [i[1:] for i in state['tier3'].values]
	if len(tier3) > 0:
		tier3 = np.concatenate(tier3, axis=0)
		#tier3 = subtract_cards(tier3, players_cards[0])
		tier3 = tier_to_vector(tier3)

		return_nodes[334:334+len(tier3)] = tier3

	nobles = state['nobles'].values
	if len(nobles) > 0:
		nobles = np.concatenate(nobles, axis=0)
		#nobles = subtract_cards(nobles, players_cards[0], nobles=True)
		nobles = nobles_to_vector(nobles)

		return_nodes[498:498+len(nobles)] = nobles

	for player in range(4):
		begin = 662 + player*76
		
		return_nodes[begin: begin+40] = tokens_to_vector(state['players'][player]['tokens'])
		return_nodes[begin+40: begin+75] = tokens_to_vector(state['players'][player]['cards'])

		'''
		for idx, i in enumerate(state['players'][player]['reservations']):
			this_begin = begin+75+idx*37
			this_reservation = i.values.tolist()[0][1:]
			this_reservation = card_to_vector(this_reservation)

			return_nodes[this_begin: this_begin+37] = this_reservation
		'''
		return_nodes[begin+75] = state['players'][player]['score']

	return return_nodes

def state_to_nodes3(state):
	return_nodes = np.zeros((input_nodes))

	sorted_tokens = [value for (key, value) in sorted(list(state['players'][0]['tokens'].items()))]
	return_nodes[:6] = sorted_tokens

	tier1 = [i[1:] for i in state['tier1'].values]
	tier1 = np.concatenate(tier1, axis=0)
	tier1 = tier_to_vector(tier1)

	return_nodes = tier1[6:13]
	print(return_nodes)
	return return_nodes

def subtract_cards(tier, cards, nobles=False):
	zeros = np.zeros((len(tier)))
	piece_size, start = 5, 2
	if nobles:
		piece_size, start = 5, 0

	for i in range(0, len(tier), piece_size+start):
		tier[i+start: i+start+piece_size] -= cards
	tier = np.maximum(tier, zeros, casting='unsafe', dtype=int)
	
	return tier

def nobles_to_vector(nobles):
	return_nobles = np.zeros((5 * 7 * 5))

	for i in range(5):
		nobel = nobles[i*5: i*5 + 5]

		nobel_idx = i*35
		return_nobles[nobel_idx: nobel_idx+35] = tokens_to_vector(nobel)

	return return_nobles

def card_to_vector(card):
	return_card = np.zeros((37))
	return_card[:2] = card[:2]
	return_card[2:37] = tokens_to_vector(card[2:])

def card_to_vector2(card):
	return_card = np.zeros((11))
	return_card[0] = card[0]
	return_card[card[1]] = 1
	return_card[6:] = card[2:]

	return return_card

def tier_to_vector2(tier):
	return_tier = np.zeros((11 * 4))

	for i in range(len(tier)//7):
		card = tier[i*7: i*7 + 7]
		card_idx = i*11
		return_tier[card_idx] = card[0]
		return_tier[card_idx + card[1]] = 1
		return_tier[card_idx+6: card_idx+11] = card[2:]

	return return_tier

def tier_to_vector(tier):
	return_tier = np.zeros((41 * 4))

	for i in range(len(tier)//7):
		card = tier[i*7: i*7 + 7]
		card_idx = i*41
		return_tier[card_idx] = card[0]
		return_tier[card_idx + card[1]] = 1
		return_tier[card_idx+6: card_idx+41] = tokens_to_vector(card[2:])

	return return_tier

def tokens_to_vector(tokens):
	if type(tokens) is dict:
		if 'gold' in tokens:
			return_tokens = np.zeros((5 * 7 + 5))
		else:
			return_tokens = np.zeros(5 * 7)

		so_far = 0
		for (key, value) in sorted(list(tokens.items())):
			return_tokens[so_far: so_far + value] = 1

			if key == 'gold':
				so_far += 5
			else:
				so_far += 7
	else:
		return_tokens = np.zeros(5 * 7)
		for idx, value in enumerate(tokens):
			return_tokens[idx*7: idx*7 + value] = 1

	return return_tokens

def combination_to_tokens(combination):
	COLORS = ['green', 'white', 'blue', 'black', 'red']
	to_pick = {c: 0 for c in COLORS}
	for color_idx in combination:
		to_pick[COLORS[color_idx]] += 1
	return to_pick

def step(move, state, env, print_move=False):
	COLORS = env.colors

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

def step3(move, state, env, print_move=False):
	COLORS = env.colors

	if move in range(10):
		combination = pick_tokens['3'][move]
		tokens = combination_to_tokens(combination)
		move = {'pick': tokens}

	elif move == 10:
		card = state['tier1'][:1]
		move = {'buy': card}

	else:
		card = state['tier1'][:1]
		move = {'reserve': card}

	return env.move(move)

def avaliable_outputs3(state, env):
	return_nodes = np.zeros(output_nodes)
	player_tokens = state['players'][0]['tokens']

	if not state['return_tokens']:
		if sum(player_tokens.values()) <= 7:
			for idx, combination in enumerate(pick_tokens['3']):
				if all(state['tokens'][env.colors[i]] >= 1 for i in combination):
					return_nodes[idx] = 1

	this_card = state['tier1'][:1]
	if self.can_afford(this_card):
		return_nodes[10] = 1
	return_nodes[11] = 1
	return_nodes[12] = 1

	return return_nodes

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

	if aval_vector[a] == 0:
		a = len(aval_vector) - 1

	return a

def reward(state, last_state):
	me = state['players'][0]
	last_me = last_state['players'][0]
	score_difference = me['score'] - last_me['score']
	cards_difference = sum(me['cards'].values()) - sum(last_me['cards'].values())
	#good_boy = cards_difference + score_difference**2
	good_boy = sum(me['cards'].values()) + me['score']**2
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

	# Input Layer
	#loaded_model.add(InputLayer(batch_input_shape=(1, input_nodes)))
	loaded_model.add(Dense(input_nodes, activation='linear', input_dim=input_nodes))

	# Hidden layers
	loaded_model.add(Dense(input_nodes, activation='sigmoid'))
	#loaded_model.add(LeakyReLU(alpha=.001))
	loaded_model.add(Dense(hidden_layer_size, activation='sigmoid'))
	loaded_model.add(Dense(hidden_layer_size, activation='sigmoid'))

	# Output layer
	loaded_model.add(Dense(output_nodes, activation='linear'))

	# load weights and compile model
	loaded_model.load_weights(abspath + 'brains/' + model_name + ".h5")
	loaded_model.compile(loss='mse', optimizer='adam', metrics=['mae'])

	return loaded_model

def create_model(hidden_layer_size, output_nodes):
	model = Sequential()
	
	# Input Layer
	#model.add(InputLayer(batch_input_shape=(1, input_nodes)))
	model.add(Dense(input_nodes, activation='linear', input_dim=input_nodes))
	
	# Hidden Layers
	model.add(Dense(input_nodes, activation='sigmoid'))
	#model.add(LeakyReLU(alpha=.001))
	model.add(Dense(hidden_layer_size, activation='sigmoid'))
	model.add(Dense(hidden_layer_size, activation='sigmoid'))
	
	# Output layer
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
					PrintBoard.print_state(s, game_round, 0)
					step(a, s, env, True)
					exit('exiting')

				if not new_s['return_tokens']:
					break

			if a == 87:
				mr_87 += 1

			for idx, dumb_model in enumerate(dumb_models):
				while True:
					dumb_a = choose_action(new_s, dumb_model, env)
					try:
						new_s = step(dumb_a, new_s, env)
					except:
						PrintBoard.print_state(new_s, game_round, idx)
						step(dumb_a, new_s, env, True)
						exit('exiting')

					if not new_s['return_tokens']:
						break
				if dumb_a == 87:
					mr_87 += 1

			done = new_s['end']

			aval_vector = env.avaliable_outputs()
			new_x = np.array(state_to_nodes(new_s)).reshape(1, input_nodes)
			next_state_pred = model.predict(new_x)
			
			r = reward(new_s, s)
			target = r + this_brain['y'] * np.max(next_state_pred*aval_vector)
			target_vec = next_state_pred[0]
			#print('ai prediction:', target_vec[a])
			#print('correct:', target)
			#print()
			target_vec[a] = target
			model.fit(new_x, target_vec.reshape(-1, output_nodes), epochs=1, verbose=0)
			s = deepcopy(new_s)

			r_sum += r

			if mr_87 == 4 or game_round > 60:
				break

		PrintBoard.print_state(s, game_round, 0)

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

			brains_info[model_name] = this_brain
			save(brains_info)
		
		r_avg_list.append(r_sum)

		if os.path.isfile('stop'):
			print('stopping')
			break

	dump_model(model, model_name)
	this_brain['episodes_done'] = i
	
	brains_info[model_name] = this_brain
	save(brains_info)

	chunk = 10
	r_mean = []
	for i in range(int(math.ceil(len(r_avg_list)/chunk))):
		r_mean.append(np.mean(r_avg_list[chunk*i: min(chunk*(i+1), len(r_avg_list))]))

	plt.plot([i*chunk for i in range(len(r_mean))], r_mean)
	plt.show()
