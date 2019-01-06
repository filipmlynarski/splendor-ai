import sys
import os
abspath = '/'.join(os.path.abspath(__file__).split('/')[:-1]) + '/'
under_abspath = '/'.join(os.path.abspath(__file__).split('/')[:-2])

sys.path.insert(0, under_abspath)
from environment import splendor
from print_board import PrintBoard
from model import Model

from copy import deepcopy


env = splendor.Splendor()
model = Model(env.colors)
model2 = Model(env.colors)

for game_number in range(1):
	state = deepcopy(env.reset())
	model.eps *= model.decay_factor

	done = False
	game_round = 0
	while not done:
		empty_moves = 0
		game_round += 1

		action = model.best_action(state, env)
		new_state = model.step(state, action, env)
		model.update_weights(state, new_state, action, env)
		if action == 6:
			break
			empty_moves += 1
		elif action == 5:
			print('bought!')
			break

		'''
		if new_state['return_tokens']:
			action = model.best_action(new_state, env)
			old_state = new_state
			new_state = model.step(new_state, action, env)
			model.update_weights(old_state, new_state, action, env)

			if not new_s['return_tokens']:
				break
		'''

		state = new_state
		if state['return_tokens']:
			break

		for i in range(3):
			state = env.move({'pick': {}})
			continue
			action = model2.best_action(state, env)
			new_state = model2.step(state, action, env)
			if action == 6:
				empty_moves += 1
			elif action == 5:
				print('other one bought')
				done = True


			'''
			if new_state['return_tokens']:
				action = model2.best_action(new_state, env)
				new_state = model2.step(new_state, action, env)

				if not new_s['return_tokens']:
					break

			'''
			state = new_state


		#PrintBoard.print_state(new_state, game_round, 0)
		if empty_moves == 4:
			print('breaking!')
			break
		print(env.return_state(False)['players'][0]['tokens'])
	#print(env.return_state(False)['players'][0]['cards'])
	#PrintBoard.print_state(new_state, game_round, 0)
	if not (game_number%100):
		#print(model.eps)
		print('game round: {}'.format(game_number))

for layer in model.model.layers:
	for i in layer.get_weights()[0]:
		print('  '.join(str([round(x, 2) for x in list(i)]).replace(',', ' ').split()))