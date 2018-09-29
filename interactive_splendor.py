from environment import splendor

import os
import time
import sys
import numpy as np
sys.path.insert(0, 'splendor_ai')

from termcolor import colored, colored
from pprint import pprint
from train_model import load_model, avaliable_outputs, state_to_nodes, step

env = splendor.Splendor()
term_colors = env.colors.copy()
term_colors[3] = 'magenta'
term_colors.append('yellow')
s = env.return_state()

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

def is_int(x):
	try:
		int(x)
		return True
	except:
		return False

def color_card(card):
	card = card.split()
	
	card[0] = ' '*(3-len(card[0])) + colored(card[0], attrs=['underline'])
	card[1] = ' '*5 + colored(card[1], attrs=['bold'])
	card[2] = ' '*4 + colored(u'\u25AE', term_colors[int(card[2])-1]) + ' '
	for idx, color in enumerate(term_colors[:-1]):
		if card[idx+3] == '0':
			card[idx+3] = ''
		else:
			card[idx+3] = colored(u'\u25C9', color) * int(card[idx+3])

	for card_idx in range(len(card)):
		if card[7-card_idx] == '':
			del card[7-card_idx]

	return(' '.join(card))

def color_nobel(nobel):
	return '  '.join(colored(u'\u25AE', term_colors[idx]) * int(i) for idx ,i in enumerate(nobel.split()) if i != '0')

def color_tokens(tokens, card=False):
	if card:
		mark = u'\u25AE'
	else:
		mark = u'\u25C9'

	return_tokens = []
	for color in term_colors[:len(tokens.keys())]:
		if color == 'magenta':
			if tokens['black'] > 0:
				return_tokens.append(colored(mark, color) * int(tokens['black']))
		elif color == 'yellow':
			if tokens['gold'] > 0:
				return_tokens.append(colored(mark, color) * int(tokens['gold']))
		else:
			if tokens[color] > 0:
				return_tokens.append(colored(mark, color) * int(tokens[color]))
	return ' '.join(return_tokens)

def print_player(player):
	if player['score'] >= 15:
		print('score:  ' + colored(str(player['score']), 'yellow', attrs=['bold']))
	else:
		print('score:  ' + colored(str(player['score']), attrs=['bold']))
	print('cards:  ' + color_tokens(player['cards'], True))
	print('tokens: ' + color_tokens(player['tokens']) + ' ')
	print('nobles: ' + str(player['nobles']))
	columns_to_show = [
		'value',
		'type'
	] + env.colors
	if len(player['reservations']) > 0:
		sys.stdout.write('reservations: ')
	#	print(colored('idx', attrs=['underline']) + '  ' + colored('value', attrs=['bold']) + '  type  price')
		colored_reservations = []
		for reservation in player['reservations']:
			colored_reservation = color_card(reservation[columns_to_show].to_string().splitlines()[1:][0]).split()
			colored_reservation.insert(1, '  ')
			colored_reservation.insert(3, '  ')
			colored_reservation.insert(5, '  ')
			colored_reservations.append(' '.join(colored_reservation))
		sys.stdout.write(u'   \u2503  '.join(colored_reservations) + '\n')

def print_state(state, clear=True):
	if clear:
		os.system('clear')

	columns_to_show = [
		'value',
		'type'
	] + env.colors
	print('Round: ' + str(game_round))
	
	print('Nobels'.center(47, '='))
	for i in state['nobels'].to_string(index=False).splitlines()[1:]: print(' '*17 + color_nobel(i))
	print()

	#print(str(state['tier3'][columns_to_show]).splitlines()[0])
	print(colored('idx', attrs=['underline']) + '  ' + colored('value', attrs=['bold']) + '  type  price')
	if len(state['tier3']) > 0:
		print('Tier3'.center(47, '='))
		for i in str(state['tier3'][columns_to_show]).splitlines()[1:]: print(color_card(i))
	if len(state['tier2']) > 0:
		print('Tier2'.center(47, '='))
		for i in str(state['tier2'][columns_to_show]).splitlines()[1:]: print(color_card(i))
	if len(state['tier1']) > 0:
		print('Tier1'.center(47, '='))
		for i in str(state['tier1'][columns_to_show]).splitlines()[1:]: print(color_card(i))
	
	print(	'T3 hidden cards: ' + str(state['hidden_t1']) + 
			'\tT2 hidden cards: ' + str(state['hidden_t2']) + 
			'\tT1 hidden cards: ' + str(state['hidden_t3']))
	print()

	print('Boards tokens: ' + color_tokens(state['tokens']))
	print()
	
	print_players(state['players'])

def print_players(players):
	for idx in range(len(players)):
		if idx == player_index:
			print('--> player' + str(idx+1))
		else:
			print('player' + str(idx+1))
		print_player(players[idx])
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
	for idx, player_type in enumerate(sys.argv[1:]):
		if player_type != 'p':
			model_name = player_type
			h5 = ''.join(['splendor_ai/brains/' + model_name, '.h5'])
			if not os.path.isfile(h5):
				print('supplied model "{}" doesn\'t exist'.format(model_name))
				sys.exit()

			model = load_model(model_name)
			players[idx] = 1

short = {'g': 'green', 'w': 'white', 'b': 'blue', 'k': 'black', 'r': 'red'}
player_index = 0
game_round = 1
anty_nystagmus = 0
input('Press ENTER to start game')
print_state(s)

while True:
	if players[player_index] == 0:
		bot = False
		sys.stdout.write('player' + str(player_index+1) + ' >> ')
	else:
		thinking_start = time.time()
		
		sys.stdout.write('player' + str(player_index+1) + ' [' + model_name + '] >> ')
		s_shifted = env.return_state()
		bot = True
		aval_vector = avaliable_outputs(s_shifted, env)

		if sum(aval_vector) != 1:
			aval_vector[-1] = 0

		x = np.array(state_to_nodes(s_shifted)).reshape(1, 250)
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

	print_state(s)
	if s['end']:
		break

print('Hellon\'t')