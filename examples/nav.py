import numpy as np
import time

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
        4: [1, 1],
        5: [1, -1],
        6: [-1, 1],
        7: [-1, -1]
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
    def __init__(self, size = 8):
        # Define grid size
        self.size = size

        # Define Empty List of Pieces
        self.pieces = []
  
        # Define Key for Layout
        self.key = {
            'empty' : ' ',
            'police' : 'P',
            'thief' : 'T',
            'gold' : 'G'}

    def initializePositions(self, nPolice = 3, nThief = 1, nGold = 1):
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
        populate(self, nThief, 'thief')
        populate(self, nGold, 'gold')
        populate(self, nPolice, 'police')
    
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

if __name__ == '__main__':
    env = Env()
    env.initializePositions()
    env.display()
    env.getDistanceThief()
