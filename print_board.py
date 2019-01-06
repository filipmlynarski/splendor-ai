import os
from termcolor import colored
from environment import splendor

class PrintBoard(object):

	env = splendor.Splendor()
	term_colors = env.colors.copy()
	term_colors[3] = 'magenta'
	term_colors.append('yellow')
	colors = env.colors

	def color_card(card):
		card = card.split()
		
		card[0] = ' '*(3-len(card[0])) + colored(card[0], attrs=['underline'])
		card[1] = ' '*5 + colored(card[1], attrs=['bold'])
		card[2] = ' '*4 + colored(u'\u25AE', PrintBoard.term_colors[int(card[2])-1]) + ' '
		for idx, color in enumerate(PrintBoard.term_colors[:-1]):
			if card[idx+3] == '0':
				card[idx+3] = ''
			else:
				card[idx+3] = colored(u'\u25C9', color) * int(card[idx+3])

		for card_idx in range(len(card)):
			if card[7-card_idx] == '':
				del card[7-card_idx]

		return(' '.join(card))

	def color_nobel(nobel):
		return '  '.join(colored(u'\u25AE', PrintBoard.term_colors[idx]) * int(i) for idx ,i in enumerate(nobel.split()) if i != '0')

	def color_tokens(tokens, card=False):
		if card:
			mark = u'\u25AE'
		else:
			mark = u'\u25C9'

		return_tokens = []
		for color in PrintBoard.term_colors[:len(tokens.keys())]:
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
		print('cards:  ' + PrintBoard.color_tokens(player['cards'], True))
		print('tokens: ' + PrintBoard.color_tokens(player['tokens']) + ' ')
		print('nobles: ' + str(player['nobles']))
		columns_to_show = [
			'value',
			'type'
		] + PrintBoard.colors
		if len(player['reservations']) > 0:
			print('reservations: ', end='')
		#	print(colored('idx', attrs=['underline']) + '  ' + colored('value', attrs=['bold']) + '  type  price')
			colored_reservations = []
			for reservation in player['reservations']:
				colored_reservation = PrintBoard.color_card(reservation[columns_to_show].to_string().splitlines()[1:][0]).split()
				colored_reservation.insert(1, '  ')
				colored_reservation.insert(3, '  ')
				colored_reservation.insert(5, '  ')
				colored_reservations.append(' '.join(colored_reservation))
			print(u'   \u2503  '.join(colored_reservations) + '\n', end='')

	def print_state(state, game_round, player_index, clear=True):
		if clear:
			os.system('clear')

		columns_to_show = [
			'value',
			'type'
		] + PrintBoard.colors
		print('Round: ' + str(game_round))
		
		print('Nobels'.center(47, '='))
		for i in state['nobles'].to_string(index=False).splitlines()[1:]: print(' '*17 + PrintBoard.color_nobel(i))
		print()

		#print(str(state['tier3'][columns_to_show]).splitlines()[0])
		print(colored('idx', attrs=['underline']) + '  ' + colored('value', attrs=['bold']) + '  type  price')
		if len(state['tier3']) > 0:
			print('Tier3'.center(47, '='))
			for i in str(state['tier3'][columns_to_show]).splitlines()[1:]: print(PrintBoard.color_card(i))
		if len(state['tier2']) > 0:
			print('Tier2'.center(47, '='))
			for i in str(state['tier2'][columns_to_show]).splitlines()[1:]: print(PrintBoard.color_card(i))
		if len(state['tier1']) > 0:
			print('Tier1'.center(47, '='))
			for i in str(state['tier1'][columns_to_show]).splitlines()[1:]: print(PrintBoard.color_card(i))
		
		print(	'T3 hidden cards: ' + str(state['hidden_t1']) + 
				'\tT2 hidden cards: ' + str(state['hidden_t2']) + 
				'\tT1 hidden cards: ' + str(state['hidden_t3']))
		print()

		print('Boards tokens: ' + PrintBoard.color_tokens(state['tokens']))
		print()
		
		PrintBoard.print_players(state['players'], player_index)

	def print_players(players, player_index):
		for idx in range(len(players)):
			if idx == player_index:
				print('--> player' + str(idx+1))
			else:
				print('player' + str(idx+1))
			PrintBoard.print_player(players[idx])
			print()