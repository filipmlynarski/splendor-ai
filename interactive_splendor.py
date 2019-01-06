from environment import splendor
from print_board import PrintBoard

import os
import time
import sys
import numpy as np
sys.path.insert(0, 'splendor_ai')

from pprint import pprint
from train_model import load_model, state_to_nodes, step

def is_int(x):
	try:
		int(x)
		return True
	except:
		return False

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
	for idx, player_type in enumerate(sys.argv[1:]):
		if player_type != 'p':
			model_name = player_type
			h5 = ''.join(['splendor_ai/brains/' + model_name, '.h5'])
			if not os.path.isfile(h5):
				print('supplied model "{}" doesn\'t exist'.format(model_name))
				sys.exit()

			model = load_model(model_name)
			players[idx] = 1


if __name__ == '__main__':

	os.system('clear')
	print('USAGE')
	print('\tpick tokens:')
	print('\t\tp[5 integers corresponding to amount of tokens of each color]')
	print('\t\texample: „gkr“ will pick 1 green, 1 black and 1 red, „bb“ will pick 2 blue tokens')
	print('\tbuy card:')
	print('\t\tb[index of card to pick]')
	print('\t\texample: „b36“')
	print('\treserve card:')
	print('\t\tr[index of card to reserve]')
	print('\t\texample: „r17“')
	print('\tto end game simply type one of following: „end“, „q“, „bye“, CTRIL-C or CTRIL-D')
	print()

	env = splendor.Splendor()
	s = env.return_state()

	short = {'g': 'green', 'w': 'white', 'b': 'blue', 'k': 'black', 'r': 'red'}
	player_index = 0
	game_round = 1
	anty_nystagmus = 0
	input('Press ENTER to start game')
	PrintBoard.print_state(s, game_round, player_index)

	while True:
		if players[player_index] == 0:
			bot = False
			sys.stdout.write('player' + str(player_index+1) + ' >> ')
		else:
			thinking_start = time.time()
			
			sys.stdout.write('player' + str(player_index+1) + ' [' + model_name + '] >> ')
			s_shifted = env.return_state()
			bot = True
			aval_vector = env.avaliable_outputs()

			if sum(aval_vector) != 1:
				aval_vector[-1] = 0

			x = np.array(state_to_nodes(s_shifted)).reshape(1, 346)
			raw_prediction = model.predict(x)
			prediction = raw_prediction * aval_vector
			a = np.argmax(prediction[0])
			if prediction[0][a] <= 0:
				not0 = prediction != 0
				a = np.argmax(not0)

			step(a, s_shifted, env, True)
			s = step(a, s_shifted, env)

			#sys.stdout.write('Press ENTER to confirm')
			#input()
			
			thinking_time = time.time() - thinking_start
			if thinking_time < anty_nystagmus:
				time.sleep(anty_nystagmus - thinking_time)

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

			elif user_input[0] == 'b' and is_int(user_input[1:]):
				card_idx = int(user_input[1:])
				card = get_card(card_idx)

				move = {'buy': card}

			elif user_input[0] == 'r' and is_int(user_input[1:]):
				card_idx = int(user_input[1:])
				card = get_card(card_idx)

				move = {'reserve': card}

			elif all(map(lambda x: x in short, user_input)):
				try:
					to_pick = {i: 0 for i in env.colors}
					for color_letter in user_input:
						to_pick[short[color_letter]] += 1

					if s['return_tokens']:
						move = {'return': to_pick}
					else:
						move = {'pick': to_pick}

				except Exception as e:
					if s['return_tokens']:
						print('can\'t return, error: ' + str(e))
					else:
						print('can\'t pick, error: ' + str(e))
					continue

			else:
				print('invalid action, avaliable actions: p b r')
				continue

			try:
				s = env.move(move)
			except Exception as e:
				print('wrong move, error: ' + str(e))
				continue

		s = env.return_state(False)
		player_index = s['player_index']

		if player_index == 0 and not s['return_tokens'] and not s['end']:
			game_round += 1

		PrintBoard.print_state(s, game_round, player_index)
		if s['end']:
			break

	print('Hellon\'t')