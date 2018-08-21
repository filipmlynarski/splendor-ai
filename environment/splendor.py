import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import os
from copy import deepcopy
import pandas as pd


class Splendor:

	def __init__(self):
		self.load_cards()
		self.reset()
		self.avaliable_actions = ['buy', 'pick', 'reserve']
		self.colors = ['green', 'white', 'blue', 'black', 'red']

	def reset(self, return_state=True):
		self.end = False
		self.set_cards()
		self.create_players()
		self.place_tokens()
		
		if return_state:
			return self.return_state()

	def return_state(self):
		#shift players so that player thats next to move will be first in list
		shifted_players = self.players[self.current_player:] + self.players[:self.current_player]
		shifted_players = self.players

		shown_tier1 = self.tier1[-min(5, len(self.tier1)):]
		shown_tier2 = self.tier2[-min(5, len(self.tier2)):]
		shown_tier3 = self.tier3[-min(5, len(self.tier3)):]

		#need to check if some mutal objects aint copied in game
		game = {
			'players': shifted_players,
			'tokens': self.tokens.copy(),
			'tier1': shown_tier1,
			'tier2': shown_tier2,
			'tier3': shown_tier3,
			'hidden_t1': len(self.tier1) - len(shown_tier1),
			'hidden_t2': len(self.tier2) - len(shown_tier2),
			'hidden_t3': len(self.tier3) - len(shown_tier3),
			'nobels': self.nobles.copy(),
			'end': self.end
		}

		if self.end:
			self.reset(False)

		return game

	def move(self, move):
		action = list(move.keys())
		if len(action) != 1:
			assert False, 'move dict requires exactly one key'

		action = action[0]
		if action not in self.avaliable_actions:
			assert False, 'invalid action, avaliable actions: ' + str(self.avaliable_actions)

		if action == 'buy':
			requested_card = move[action]
			if not self.can_afford(requested_card):
				assert False, 'invalid action you can\'t buy this card'

			self.buy(requested_card)

		elif action == 'pick':
			requested_tokens = move[action]
			if not self.can_pick(requested_tokens):
				assert False, 'invalid action you can\'t pick this tokens'

			self.pick(requested_tokens)

		elif action == 'reserve':
			card_to_reserve = move[action]
			if not self.can_reserve(card_to_reserve):
				assert False, 'invalid action you can\'t reserve this card'

			self.reserve(card_to_reserve)

		else:
			assert False, 'invalied action, avaliable actions: ' + str(self.avaliable_actions)

		self.check_nobles()
		self.check_winners()

		self.current_player = (self.current_player + 1) % 4

		return self.return_state()

	def remove_card(self, card):
		if int(card['tier']) == 1:
			card_idx = card.index.tolist()[0]
			self.tier1 = self.tier1.drop(card_idx)

		if int(card['tier']) == 2:
			card_idx = card.index.tolist()[0]
			self.tier2 = self.tier1.drop(card_idx)

		if int(card['tier']) == 3:
			card_idx = card.index.tolist()[0]
			self.tier3 = self.tier1.drop(card_idx)

	def card_to_colors(self, card):
		return ','.join(str(int(card[c])) for c in self.colors)

	def show_cards(self):
		shown_tier1 = self.tier1[-min(5, len(self.tier1)):].reset_index(drop=True)
		shown_tier2 = self.tier2[-min(5, len(self.tier2)):].reset_index(drop=True)
		shown_tier3 = self.tier3[-min(5, len(self.tier3)):].reset_index(drop=True)

		str_tier1 = [self.card_to_colors(shown_tier1.ix[i]) for i in range(len(shown_tier1))]
		str_tier2 = [self.card_to_colors(shown_tier2.ix[i]) for i in range(len(shown_tier2))]
		str_tier3 = [self.card_to_colors(shown_tier3.ix[i]) for i in range(len(shown_tier3))]

		return str_tier1 + str_tier2 + str_tier3

	def show_reservations(self):
		reservations = self.players[self.current_player]['reservations']
		return [self.card_to_colors(reservations[i]) for i in range(len(reservations))]

	def can_afford(self, card):
		#check if requested card is in shown cards or current player reservations
		str_reservations = self.show_reservations()

		if not any(self.card_to_colors(card) in i for i in self.show_cards() + str_reservations):
			return False

		tokens = self.players[self.current_player]['tokens']
		cards = self.players[self.current_player]['cards']
		token_diff = [tokens[i] + cards[i] - card[i] for i in self.colors]
		missing_tokens = abs(sum(i.values[0] for i in token_diff if i.values[0] < 0))

		if self.players[self.current_player]['tokens']['gold'] >= missing_tokens:
			return True

		return False

	def buy(self, card):
		if self.card_to_colors(card) in self.show_reservations():
			#if this card was reserved
			self.players[self.current_player]['tokens']['gold'] += 1
			reservations = self.players[self.current_player]['reservations']
			idx = card.index.tolist()[0]
			idxes = [i.index.tolist()[0] for i in reservations]
			to_pop = idxes.index(idx)
			self.players[self.current_player]['reservations'].pop(to_pop)
		else:
			self.remove_card(card)

		for color in self.colors:
			#this player's cards of certain color
			this_cards = self.players[self.current_player]['cards'][color]

			#if player doesnt have more cards of this color than needed
			if int(card[color]) > this_cards:
				#subtract missing tokens of this color from player and add them to board
				necessary_tokens = int(card[color]) - this_cards
				self.players[self.current_player]['tokens'][color] -= necessary_tokens
				self.tokens[color] += necessary_tokens

				#tokens of this color after purchase
				this_tokens = self.players[self.current_player]['tokens'][color]
				
				#if amount of this color tokens is negative equalize it with gold tokens
				if this_tokens < 0:
					self.players[self.current_player]['tokens']['gold'] += this_tokens
					self.players[self.current_player]['tokens'][color] = 0

					#equlize situation on board
					self.tokens['gold'] -= this_tokens
					self.tokens[color] += this_tokens

		card_color = self.colors[int(card['type'])-1]
		self.players[self.current_player]['cards'][card_color] += 1
		self.players[self.current_player]['score'] += int(card['value'])

	def can_pick(self, tokens):
		values = tokens.values()
		if not 0 <= sum(values) <= 3 or any(i == 3 for i in values):
			return False

		if any(i == 2 for i in values) and sum(values) != 2:
			return False

		for color in tokens:
			if self.tokens[color] < tokens[color]:
				return False

		player_tokens = self.players[self.current_player]['tokens'].values()
		if sum(values) + sum(player_tokens) > 10:
			return False

		return True

	def pick(self, tokens):
		for color in tokens:
			self.players[self.current_player]['tokens'][color] += tokens[color]
			self.tokens[color] -= tokens[color]

	def can_reserve(self, card):
		#current player reservations
		this_reservations = self.players[self.current_player]['reservations']

		if len(this_reservations) == 3 or self.tokens['gold'] == 0:
			return False

		if self.card_to_colors(card) not in self.show_cards():
			return False

		return True

	def reserve(self, card):
		self.remove_card(card)
		self.players[self.current_player]['reservations'].append(card)
		self.tokens['gold'] -= 1

	def check_nobles(self):
		this_cards = self.players[self.current_player]['cards']
		for nobleman in range(len(self.nobles)):
			for color in self.colors:
				if this_cards[color] < self.nobles.ix[nobleman][color]:
					break
			else:
				self.players[self.current_player]['nobleman'] += 1
				self.players[self.current_player]['score'] += 3
				self.nobles.drop(nobleman.index.tolist())
				break

	def check_winners(self):
		if any(i['score'] >= 15 for i in self.players):
			self.end = True

	def load_cards(self):
		abspath = '/'.join(os.path.abspath(__file__).split('/')[:-1]) + '/'

		if not os.path.isfile(abspath + 'cards.csv'):
			assert False, 'cards.csv file does not exist'
		if not os.path.isfile(abspath + 'nobles.csv'):
			assert False, 'nobles.csv file does not exist'

		self.primary_cards = pd.read_csv(abspath + 'cards.csv')
		self.primary_nobles = pd.read_csv(abspath + 'nobles.csv')

	def place_tokens(self):
		self.tokens = {
			'green': 7,
			'white': 7,
			'blue': 7,
			'black': 7,
			'red': 7,
			'gold': 5
		}

	def set_cards(self):
		#shuffle all the cards together
		shuffled_cards = self.primary_cards.sample(frac=1)
		shuffled_nobles = self.primary_nobles.sample(frac=1)

		#organize cards in relation to their tier
		t1_idx = shuffled_cards['tier'] == 1
		t2_idx = shuffled_cards['tier'] == 2
		t3_idx = shuffled_cards['tier'] == 3
		self.tier1 = shuffled_cards.loc[t1_idx].reset_index(drop=True)
		self.tier2 = shuffled_cards.loc[t2_idx].reset_index(drop=True)
		self.tier3 = shuffled_cards.loc[t3_idx].reset_index(drop=True)

		self.nobles = shuffled_nobles[-5:].reset_index(drop=True)

	def create_players(self):
		#the player's index, which will move next
		self.current_player = 0

		primary_player = {
			'score': 0,
			'tokens': {
				'green': 0,
				'white': 0,
				'blue': 0,
				'black': 0,
				'red': 0,
				'gold': 0
			},
			'cards': {
				'green': 0,
				'white': 0,
				'blue': 0,
				'black': 0,
				'red': 0
			},
			'nobles': 0,
			'reservations': []
		}

		self.players = [deepcopy(primary_player) for _ in range(4)]

if __name__ == '__main__':
	env = Splendor()
	s = env.return_state()
	card = s['tier1'][:1]

	for i in range(2):
		env.move({'pick': {'green': 1, 'blue': 1, 'white': 1}})
		env.move({'pick': {'black': 1, 'red': 1}})

	env.move({'reserve': card})
	try:
		env.move({'buy': card})
	except:
		print('maybe next time')
	print(env.players[0]['cards'])