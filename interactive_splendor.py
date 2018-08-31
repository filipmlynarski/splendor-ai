from environment import splendor
import os
import sys
import numpy as np
sys.path.insert(0, 'splendor_ai')

from pprint import pprint
from train_model import load_model, avaliable_outputs, state_to_nodes, step

env = splendor.Splendor()
s = env.return_state()

os.system('clear')
print('USAGE')
print('\tpick tokens:')
print('\t\tp[5 integers corresponding to amount of tokens of each color]')
print('\t\texample: „p10011“ will pick 1 green, 1 black and 1 red')
print('\tbuy card:')
print('\t\tb[index of card to pick]')
print('\t\texample: „b36“')
print('\treserve card:')
print('\t\tr[index of card to reserve]')
print('\t\texample: „r17“')
print('\tto end game simply type one of following: „end“, „q“, „bye“, CTRIL-C or CTRIL-D')
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

def print_players(players):
	for idx in range(len(players)):
		print('player' + str(idx+1))
		pprint(players[idx])
		print()

def get_card(card_idx):
	player_reservations = s['players'][player_index]['reservations']

	if card_idx in s['tier1'].index:
		idxs = s['tier1'].index.tolist()
		idx = idxs.index(card_idx)
		return s['tier1'][idx:idx+1]
	elif card_idx in s['tier2'].index:
		idxs = s['tier2'].index.tolist()
		idx = idxs.index(card_idx)
		return s['tier2'][idx:idx+1]
	elif card_idx in s['tier3'].index:
		idxs = s['tier3'].index.tolist()
		idx = idxs.index(card_idx)
		return s['tier3'][idx:idx+1]
	elif any(card_idx in i.index for i in player_reservations):
		reservation = [i for i in player_reservations if card_idx in i.index][0]
		return reservation

players = [0, 0, 0, 0]

if len(sys.argv) != 1:
	model_name = sys.argv[1]
	h5 = ''.join(['splendor_ai/' + model_name, '.h5'])
	json = ''.join(['splendor_ai/' + model_name, '.json'])
	if not all(map(os.path.isfile, [h5, json])):
		print('supplied model "{}" doesn\'t exist'.format(model_name))
		sys.exit()

	model = load_model('splendor_ai/' + model_name)
	players[0] = 1

print_state(s, False)
player_index = 0

while True:
	if players[player_index] == 0:
		bot = False
		sys.stdout.write('player' + str(player_index+1) + ' >> ')
	else:
		bot = True
		sys.stdout.write(model_name + ' >> ')
		aval_vector = avaliable_outputs(s, env)

		if sum(aval_vector) != 1:
			aval_vector[-1] = 0

		x = np.array(state_to_nodes(s)).reshape(1, 250)
		raw_prediction = model.predict(x)
		prediction = raw_prediction * aval_vector
		a = np.argmax(prediction[0])
		if prediction[0][a] <= 0:
			not0 = prediction != 0
			a = np.argmax(not0)

		step(a, s, env, True)
		sys.stdout.write('Press ENTER to confirm')
		input()
		s = step(a, s, env)

	if not bot:
		try:
			user_input = input()
		except:
			print()
			break

		if user_input in ['end', 'q', 'bye']:
			break

		if user_input == '' or user_input == '0':
			move = {'pick': {}}

		elif user_input[0] == 'p':
			try:
				move = {'pick': {env.colors[i]: int(user_input[1+i]) for i in range(len(env.colors))}}
			except Exception as e:
				print('cant pick, error: ' + str(e))
				continue

		elif user_input[0] == 'b':
			card_idx = int(user_input[1:])
			card = get_card(card_idx)

			move = {'buy': card}

		elif user_input[0] == 'r':
			card_idx = int(user_input[1:])
			card = get_card(card_idx)

			move = {'reserve': card}

		else:
			print('invalid action, avaliable actions: p b r')
			continue

		try:
			s = env.move(move)
		except Exception as e:
			print('wrong move, error: ' + str(e))
			continue

	s = env.return_state(False)
	print_state(s)
	player_index = (player_index+1) % 4

print('Hellon\'t')
