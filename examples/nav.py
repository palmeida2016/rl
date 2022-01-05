import numpy as np

class Piece:
    def __init__(self, name, x, y):
        '''
        Initialize Piece with its position
        '''
        self.name = name
        self.x = x
        self.y = y

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
        
        return (self.x == otherPiece.x and self.y == otherPiece.y)

    def move(self, deltaX, deltaY):
        '''
        Update position of piece with delta x values
        '''
        self.x += deltaX
        self.y += deltaY

    def getPos(self):
        '''
        Return the position of piece as ordered pair
        '''
        return (self.x,self.y)

class Env:
    def __init__(self, size = 10):
        # Define Layout
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
                self.pieces.append(Piece(name,x,y))

        # Add all objects to the list
        populate(self, nPolice, 'police')
        populate(self, nThief, 'thief')
        populate(self, nGold, 'gold')


    def display(self):
        '''
        Display current configuration of the grid
        '''
        for x in range(self.size):
            print('--------------------')
            for y in range(self.size):
                print('|',end='')
                if (x,y) in [piece.getPos() for piece in self.pieces]: # If there is piece
                    # Get piece at that spot
                    p = next(piece for piece in self.pieces if piece.getPos() == (x,y))
                    
                    # Print object at that position with key
                    print(self.key.get(p.name), end='')
                else:
                    print(self.key.get('empty'),end='')
            print('')
        print('--------------------')

if __name__ == '__main__':
    env = Env()
    env.initializePositions()
    env.display()
