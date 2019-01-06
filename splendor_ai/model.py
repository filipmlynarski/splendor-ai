import os

import numpy as np
import itertools as it

from keras.models import Sequential
from keras.engine import InputLayer
from keras.layers import Dense

from keras import optimizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Model(object):

	def __init__(self, colors):
		self.colors = colors

		self.pick_tokens = {}
		self.pick_tokens['3'] = list(it.combinations([0,1,2,3,4], 3))
		self.pick_tokens['2'] = list(it.combinations([0,1,2,3,4], 2))
		self.pick_tokens['doubles'] = [0,1,2,3,4]
		self.pick_tokens['1'] = [0,1,2,3,4]
		
		# 5 nodes for card cost / 5 nodes for player tokens
		self.input_nodes = 5
		# picking just one token / buying card / empty move
		self.output_nodes = len(self.pick_tokens['1']) + 1 + 1

		self.y = 0.3
		self.eps = 1.2
		self.decay_factor = 0.99

		self.create_model()

	def create_model(self):
		self.model = Sequential()

		self.model.add(Dense(self.output_nodes, activation='linear', input_dim=self.input_nodes))
		#self.model.add(Dense(self.output_nodes, activation='linear'))
		
		adam = optimizers.adam(lr=0.05)
		self.model.compile(loss='mse', optimizer=adam, metrics=['mae'])

	def best_action(self, state, env):
		aval_vector = self.avaliable_outputs(state, env)
		if sum(aval_vector) != 1:
			aval_vector[-1] = 0

		if np.random.random() < self.eps:
			raw_prediction = np.random.rand(1, self.output_nodes)

		else:
			x = np.array(self.state_to_nodes(state)).reshape(1, self.input_nodes)
			raw_prediction = self.model.predict(x)
		
		prediction = raw_prediction * aval_vector
		a = np.argmax(prediction[0])
		if prediction[0][a] <= 0:
			not0 = prediction != 0
			a = np.argmax(not0)

		if aval_vector[a] == 0:
			a = len(aval_vector) - 1

		return a

	def update_weights(self, prev_state, new_state, action, env):
		aval_vector = self.avaliable_outputs(new_state, env)
		new_x = np.array(self.state_to_nodes(new_state)).reshape(1, self.input_nodes)
		next_state_pred = self.model.predict(new_x)

		r = self.reward(new_state)
		if sum(new_state['players'][0]['cards'].values()):
			print(r)
		#print(r)
		target = r + self.y * np.max(next_state_pred*aval_vector)
		target_vec = next_state_pred[0]
		#print('ai prediction:', target_vec[a])
		#print('correct:', target)
		#print()
		target_vec[action] = target
		#if action == 5:
			#print(action)
			#print(list(target_vec))
		#exit()
		self.model.fit(new_x, target_vec.reshape(-1, self.output_nodes), epochs=1, verbose=0)

	def reward(self, state):
		me = state['players'][-1]
		cards = sum(me['cards'].values())
		#print(me['cards'].values())
		if cards:
			return 2
		return -.1

	def state_to_nodes(self, state):
		return_nodes = np.zeros((self.input_nodes))

		card = state['tier1'][:1].values[0][3:]
		return_nodes[:5] = card

		#sorted_tokens = [value for (key, value) in sorted(list(state['players'][0]['tokens'].items())) if key != 'gold']
		#return_nodes[5:] = sorted_tokens

		return return_nodes

	def avaliable_outputs(self, state, env):
		return_nodes = np.zeros(self.output_nodes)
		player_tokens = state['players'][0]['tokens']

		if sum(player_tokens.values()) <= 9:
			for color_idx in self.pick_tokens['1']:
				if state['tokens'][env.colors[color_idx]] >= 1:
					return_nodes[color_idx] = 1

		this_card = state['tier1'][:1]

		if env.can_afford(this_card):
			return_nodes[5] = 1
		return_nodes[6] = 1

		return return_nodes

	def step(self, state, move, env):
		if move in range(5):
			combination = self.pick_tokens['1'][move]
			tokens = self.combination_to_tokens([combination])
			move = {'pick': tokens}

		elif move == 5:
			card = state['tier1'][:1]
			move = {'buy': card}

		else:
			move = {'pick': {}}

		return env.move(move)

	def combination_to_tokens(self, combination):
		COLORS = ['green', 'white', 'blue', 'black', 'red']
		to_pick = {c: 0 for c in COLORS}
		for color_idx in combination:
			to_pick[COLORS[color_idx]] += 1
		return to_pick

if __name__ == '__main__':
	model = Model(['green', 'white', 'blue', 'black', 'red'])