from splendor import Splendor
import os
import sys
from pprint import pprint

env = Splendor()
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

print_state(s, False)
player_index = 0

while True:
	sys.stdout.write('player' + str(player_index+1) + ' >> ')
	try:
		user_input = input()
	except:
		break

	if user_input in ['end', 'q', 'bye']:
		break

	if user_input == '' or user_input == '0':
		s = env.move({'pick': {}})

	elif user_input == 'xd':
		env.players[player_index]['score'] += 1
		s = env.move({'pick': {}})		

	elif user_input == 'XD':
		env.players[player_index]['score'] = 14

	elif user_input[0] == 'p':
		s = env.move({'pick': {env.colors[i]: int(user_input[1+i]) for i in range(len(env.colors))}})

	elif user_input[0] == 'b':
		card_idx = int(user_input[1:])
		card = get_card(card_idx)

		s = env.move({'buy': card})

	elif user_input[0] == 'r':
		card_idx = int(user_input[1:])
		card = get_card(card_idx)

		s = env.move({'reserve': card})

	else:
		print('invalid action, avaliable actions: p b r')
		continue

	print_state(s)
	player_index = (player_index+1) % 4

print('Hellon\'t')