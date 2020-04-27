import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Flatten
from keras.utils.np_utils import to_categorical

import matplotlib.pyplot as plt
from tetris import main, cnn_main, cols, rows, RIGHT, LEFT, UP, DOWN, NOTHING, values

import numpy as np
import random

from tetris_actions import store_actions
from tetris_binary_grids import store_grids

import pygame
from pygame.locals import *
from tetris import window


model = Sequential()
# 25 4x4 filters, because of each tiles define in a 4x4 gird
# 25 is randomy chosen
model.add(Conv2D(25, (4, 4), input_shape=(20,15,1)))
# 2x2 max pooling, 2x2 is randomly chosen
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(50, (4, 4)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Activation('softmax'))
model.add(Dense(units=10))
model.add(Activation('softmax'))
model.add(Dense(units=4))
model.compile(loss='mean_squared_error', optimizer='adagrad', metrics=['accuracy'])


class Wrapper():

	def __init__(self):
		while True:
			window.fill((0,0,0))
			font = pygame.font.SysFont('Jetbrains Mono', 30)
			text1 = font.render('Press G for generating data.', False, (255,255,255))
			text2 = font.render('Press T for training model and predict.', False, (255,255,255))
			window.blit(text1, (100, 200))
			window.blit(text2, (100, 300))
			pygame.display.update()
			for event in pygame.event.get():
				if event.type == QUIT:
					pygame.quit()
					quit()
				if event.type == KEYDOWN:
					if event.key == K_g:
						main(self, values)
					if event.key == K_t:
						x_train = np.array(store_grids)
						y_train = to_categorical(store_actions)
						model.fit(x_train, y_train, shuffle=True, epochs=10, verbose=1)
						cnn_main(self, values)
					if event.key == K_ESCAPE:
						pygame.quit()
						quit()

	def control(self, values):
		global model
		current_binary = np.array([values['current_binary']])
		prediction = model.predict_classes(current_binary)
		if prediction == RIGHT:
			return RIGHT
		elif prediction == LEFT:
			return LEFT
		elif prediction == UP:
			return UP
		elif prediction == DOWN:
			return DOWN
		# elif prediction == NOTHING:
		# 	return NOTHING
	
	def gameover(self, values):
		global model
		while True:
			window.filee((0,0,0))
			font = pygame.font.SysFont('Jetbrains Mono', 30)
			text1 = font.render('Press G for generating data.', False, (255,255,255))
			text2 = font.render('Press T for training model and predict.', False, (255,255,255))
			window.blit(text1, (120, 200))
			window.blit(text2, (120, 300))
			pygame.display.update()
			for event in pygame.event.get():
				if event.type == QUIT:
					pygame.quit()
					quit()
				if event.type == KEYDOWN:
					if event.key == K_g:
						main(self, values)
					if event.key == K_t:
						if values['score'] > 0:	
							x_train = np.array(store_grids)
							y_train = to_categorical(store_actions)
							model.fit(x_train, y_train, shuffle=True, epochs=10, verbose=1)
						cnn_main(self, values)
					if event.key == K_ESCAPE:
						pygame.quit()
						quit()

	def cnn_gameover(self, values):
		cnn_main(self, values)

if __name__ == '__main__':
	Wrapper()