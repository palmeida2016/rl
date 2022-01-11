import numpy as np
import time

# https://pythonprogramming.net/own-environment-q-learning-reinforcement-learning-python-tutorial/

class Piece:
	def __init__(self, name, x, y, size):
		'''
		Initialize Piece with its position
		'''
		self.name = name
		self.x = x
		self.y = y
		self.size = size

		self.key = {
		0: [1, 0],
		1: [-1, 0],
		2: [0, 1],
		3: [0, -1],
		}

	def __repr__(self):
		'''
		Function to print() object
		'''
		return 'Piece'

	def __str__(self):
		'''
		Function to print() object
		'''
		return f'{self.name} | ({self.x},{self.y})'

	def __sub__(self, otherPiece):
		'''
		Method to get difference in position of 2 pieces
		'''
		# Check if objects both are pieces
		assert(type(otherPiece) == type(self))

		# Calculate and return difference in positions
		return (self.x-otherPiece.x, self.y-otherPiece.y)

	def __eq__(self, otherPiece):
		'''
		Method to check if two pieces are in same pos
		'''
		# Check if objects both are pieces
		assert(type(otherPiece) == type(self))
	   
	   # Check if objects are in the same place
		return (self.x == otherPiece.x and self.y == otherPiece.y)
		

	def move(self, action):
		'''
		Update position of piece with delta x values
		'''
		# Convert from action to delta values
		(deltaX, deltaY) = self.key.get(action)
		
		# Check if move is possible
		if self.x+deltaX < 0 or self.x+deltaX > self.size-1:
			deltaX = 0

		if self.y+deltaY < 0 or self.y+deltaY > self.size-1:
			deltaY = 0
		
		# Update positions of piece with movement
		self.x += deltaX
		self.y += deltaY

	def getPos(self):
		'''
		Return the position of piece as ordered pair
		'''
		return (self.x,self.y)


class Env:
	def __init__(self, size = 4, nPolice = 1, nThief = 1, nGold = 1):
		# Define grid size
		self.size = size

		# Define Empty List of Pieces
		self.pieces = []

		# Stores the numbers given for pieces
		self.nPolice = nPolice
		self.nThief = nThief
		self.nGold = nGold

		# Define number of moves possible per piece (4 cardinal directions)
		self.moves = 4
  
		# Define Key for Layout
		self.key = {
			'empty' : ' ',
			'police' : 'P',
			'thief' : 'T',
			'gold' : 'G'}
		
		# Define constants for training
		self.episodes = 20000
		self.episode_length = 200
		self.epsilon = 0.9
		self.learning_rate = 0.1
		self.discount = 0.95

		# Define penalties
		self.move_reward = -1
		self.police_reward = -50
		self.gold_reward = 50

		# Define q-table dictionary for learning
		self.q_table = self.initializeQTable()

	def initializeQTable(self):
		'''
		Creates Q-Table to keep track of optimal moves

		Must be modified if default number of pieces is changed
		'''

		# Check if saved q_table already exists


		# Calculate required size for q_table
		size = [self.size] * 2 * (self.nPolice+self.nThief+self.nGold)
		size.append(self.moves)

		# Create q_table with required size
		q_table = np.random.rand(*size)
		q_table = q_table * (2*self.size) - self.size

		return q_table

	def train(self):
		'''
		Main script to keep train agent by modifying Q-Table
		'''
		for episode in range(self.episodes):
			self.initializePositions()

		pass		

	def save(self):
		pass

	def initializePositions(self, nPolice = 1, nThief = 1, nGold = 1):
		'''
		Randomly populates the grid with no overlap
		'''
		def populate(self, n, name):
			for _ in range(n):
				# Generate Random Pos for Object
				x = np.random.randint(self.size)
				y = np.random.randint(self.size)
				
				# Check if position is already occupied
				if (x,y) in [piece.getPos() for piece in self.pieces]:
					while (x,y) in [piece.getPos() for piece in self.pieces]: # Try until free
						x = np.random.randint(self.size)
						y = np.random.randint(self.size)
				
				# If position is accepted, add to pieces
				self.pieces.append(Piece(name,x,y,self.size))

		# Add all objects to the list
		populate(self, self.nThief, 'thief')
		populate(self, self.nGold, 'gold')
		populate(self, self.nPolice, 'police')
	
	def getDistanceThief(self):
		# Get thief object in list
		thief = next(piece for piece in self.pieces if piece.name == 'thief')
		
		# Return distance to each object
		return [thief-piece for piece in self.pieces if piece.name != 'thief']
		
	def display(self):
		'''
		Display current configuration of the grid
		'''
		for x in range(self.size):
			print((2*self.size+1)*'-')
			for y in range(self.size):
				print('|',end='')
				if (x,y) in [piece.getPos() for piece in self.pieces]: # If there is piece
					# Get piece at that spot
					p = next(piece for piece in self.pieces if piece.getPos() == (x,y))
					
					# Print object at that position with key
					print(self.key.get(p.name), end='')
				else:
					# If no object print empty space
					print(self.key.get('empty'), end='')
			print('|')
		print((2*self.size+1)*'-')

	def train(self):
		pass


# Main script to run everything
def main():
	env = Env()
	env.initializePositions()
	env.display()
	env.getDistanceThief()
	print(env.q_table)

if __name__ == '__main__':
	main()