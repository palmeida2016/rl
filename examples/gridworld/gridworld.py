import numpy as np
import time

class Piece:
    # Key to map movements to delta X and delta Y
    key = {
    0: [1, 0],
    1: [-1, 0],
    2: [0, 1],
    3: [0, -1],
    }

    def __init__(self, name, x, y, size):
        '''
        Initialize Piece with its position
        '''
        # Define key properties for piece
        self.name = name
        self.x = x
        self.y = y
        self.size = size
        
    def __repr__(self):
        '''
        Function to print() object
        '''
        return f'({self.x},{self.y})'

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
        (deltaX, deltaY) = Piece.key.get(action)

        
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
    def __init__(self, size = 5, nPolice = 2, nThief = 1, nGold = 1):
        # Define grid size
        self.size = size

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

        # Option to keep grid same
        self.staticStartingLayout = True

        # Option to change starting position of thief every episode
        self.staticStartingThief = True

        # Define constants for training
        self.episodes = 100000 # Number of simulations to be ran
        self.episode_length = 50 # Maximum number of actions in each episode
        self.epsilon_start = 0.95 # Initial epsilon
        self.epsilon = self.epsilon_start # Percent chance of object to explore vs. exploit
        self.epsilon_min = 0.05 # Min epsilon to not drop below
        self.learning_rate = 0.3
        self.discount = 0.9
        self.decay = 0.9999 # Rate by which to decrease epsilon

        # Save rewards over episodes
        self.rewards = []

        # Define penalties
        self.move_reward = -1 # Move costs 1. Encourage optimal pathing
        self.police_reward = -100 # Negative reward for obstacle
        self.gold_reward = 50 # Positive reward for goal
        self.endless_reward = 0 # Negative reward for not reaching goal by the end
        
        # Define filename to store q_tablie
        self.filename = 'q_table.npy'

        # Define q-table dictionary for learning
        self.q_table = self.initializeQTable()

        
    def initializeQTable(self):
        '''
        Creates Q-Table to keep track of optimal moves initially randomized
        '''
        # Check if saved q_table already exists
        temp = self.load(self.filename)
        if temp is not False:
            return temp

        # Calculate required size for q_table
        size = [self.size] * 2 * (self.nPolice+self.nGold)
        size.append(self.moves)

        # Create q_table with required size
        q_table = np.random.rand(*size)
        q_table = q_table * -self.moves

        return q_table

    def train(self):
        '''
        Main script to keep train agent by modifying Q-Table
        '''
        if self.staticStartingLayout:
            import copy
            self.initializePositions()
            self.default = copy.deepcopy(self.pieces)

        for episode in range(self.episodes):
            # If using same grid, use stored grid for training, otherwise, generate new grid every time
            if self.staticStartingLayout: #and not episode in range(0,self.episodes,5000):
                self.resetToDefault()
                # If thief is to be randomized
                if self.staticStartingThief:
                    self.randomPlaceThief()
            else:
                print('restarting board')
                self.epsilon = self.epsilon_start
                self.initializePositions()
                self.default = copy.deepcopy(self.pieces)


            # Print current episode
            print(episode)

            # Initialize episode reward
            episode_reward = 0

            # Get thief piece
            thief = next(piece for piece in self.pieces if piece.name == 'thief')
            
            # Get Police pieces
            police = [piece for piece in self.pieces if piece.name == 'police']

            # Get Gold piece
            gold = [piece for piece in self.pieces if piece.name == 'gold']

            for step in range(self.episode_length):
                # Print Episode
                if episode in range(0,self.episodes,10000):
                    self.display()
                    time.sleep(1)


                # Get Observation as distance between police and gold to thief
                observation = self.getDistanceThief()
                
                # Convert Observation to index for q_tables
                obs = tuple(dist for i in observation for dist in i)
                
                if np.random.random() > self.epsilon:
                    # If greater than epsilon, get correct action from q_table
                    action = np.argmax(self.q_table[obs])
                else:
                    # If not greater than epsilon, randomly take action
                    action = np.random.randint(0,4)
                
                # Move thief according to action
                thief.move(action)
                
                # Check if thief is in the same position as gold or police
                if any([thief==piece for piece in police]):
                    # If thief is in the same square as police
                    reward = self.police_reward

                elif any([thief==piece for piece in gold]):
                    # If thief is in the same square as gold
                    reward = self.gold_reward

                else:
                    # If thief is not in the same square as police or gold, give move penalty
                    reward = self.move_reward

                # Get new observation after move
                new_observation = self.getDistanceThief()

                # Convert observation to index for q_tables
                new_obs = tuple(dist for i in new_observation for dist in i)

                # Calculate Q score
                q_future_max = np.max(self.q_table[new_obs])
                q_current = self.q_table[obs][action]
                
                
                # Update q_table with new score
                if reward == self.gold_reward:
                    q_new = self.gold_reward
                else:
                    q_new = (1-self.learning_rate) * q_current + self.learning_rate * (reward + self.discount * q_future_max)
                
                # Update the q_table with new observation
                self.q_table[obs][action] = q_new
                
                # Add reward for current step to total episode reward
                episode_reward += reward

                if reward == self.gold_reward: # If thief hit police, end the episode
                    print('G')
                    break
                elif reward == self.police_reward: # If thief hit gold, end the episode
                    print('P')
                    break

                if step == self.episode_length-1: # If thief is on last move, add negative reward for not finishing
                    reward = self.endless_reward
                    break
            
            # Keep track of rewards each episode
            self.rewards.append(episode_reward)
 
            # Lower epsilon (threshold for using q_table) exponentially as learning goes on
            self.epsilon *= self.decay
            
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

        self.save(self.filename)

    def test(self):
        # Test with n starting conditions
        n = 5
        for _ in range(n):
            # Load trained q_table
            self.q_table = self.load(self.filename)

            # Initialize board randomly
            self.resetToDefault()
            self.randomPlaceThief()

            # Initialize episode reward
            episode_reward = 0

            # Get thief piece
            thief = next(piece for piece in self.pieces if piece.name == 'thief')
            
            # Get Police pieces
            police = [piece for piece in self.pieces if piece.name == 'police']

            # Get Gold piece
            gold = [piece for piece in self.pieces if piece.name == 'gold']

            for step in range(self.episode_length):
                # Print Episode
                self.display()
                print(f'Step: {step}')
                time.sleep(0.5)

                # Get Observation as distance between police and gold to thief
                observation = self.getDistanceThief()
                
                # Convert Observation to index for q_tables
                obs = tuple(dist for i in observation for dist in i)
               
                # Choose best action from q_table) 
                action = np.argmax(self.q_table[obs])
                
                # Move thief according to action
                thief.move(action)
                
                # Check if thief is in the same position as gold or police
                if any([thief==piece for piece in police]):
                    # If thief is in the same square as police
                    reward = self.police_reward

                elif any([thief==piece for piece in gold]):
                    # If thief is in the same square as gold
                    reward = self.gold_reward

                else:
                    # If thief is not in the same square as police or gold, give move penalty
                    reward = self.move_reward

                # Add reward for current step to total episode reward
                episode_reward += reward

                if reward == self.gold_reward: # If thief hit police, end the episode
                    print('G')
                    break
                elif reward == self.police_reward: # If thief hit gold, end the episode
                    print('P')
                    break

                if step == self.episode_length-1: # If thief is on last move, add negative reward for not finishing
                    reward = self.endless_reward
                    break

            print(f'Reward: {episode_reward}')


    def save(self, filename):
        np.save(filename, self.q_table)

    def load(self,filename):
        from os.path import exists

        # If filename doesn't exist, return false
        if not exists(filename):
            return False
        
        # If filename exists, load it
        return np.load(filename)

    def initializePositions(self):
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
        
        # Initiate list for pieces
        self.pieces = []

        # Add all objects to the list
        populate(self, self.nThief, 'thief')
        populate(self, self.nGold, 'gold')
        populate(self, self.nPolice, 'police')

    def resetToDefault(self):
        import copy
        self.pieces = copy.deepcopy(self.default)

    def randomPlaceThief(self):
        # Remove starting thief from pieces
        self.pieces = [piece for piece in self.pieces if piece.name != 'thief']

        # Generate Random Pos for Object
        x = np.random.randint(self.size)
        y = np.random.randint(self.size)
              
        # Check if position is already occupied
        if (x,y) in [piece.getPos() for piece in self.pieces]:
            while (x,y) in [piece.getPos() for piece in self.pieces]: # Try until free
                x = np.random.randint(self.size)
                y = np.random.randint(self.size)
                
        # If position is accepted, add to pieces
        self.pieces.append(Piece('thief',x,y,self.size))

    def getDistanceThief(self):
        '''
        Calculates the distance between thief and other pieces as observation state
        '''
        # Get thief object in list
        thief = next(piece for piece in self.pieces if piece.name == 'thief')
        
        # Return distance to each object
        return [thief-piece for piece in self.pieces if piece.name != 'thief']
        
    def display(self):
        '''
        Display current configuration of the grid
        '''
        print()
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
        print()

# Main script to run everything
def main():
    import pandas as pd
    env = Env()
    
    env.train()
    #df = pd.DataFrame(env.rewards)
    #df.to_csv('test.csv')

    env.test()

if __name__ == '__main__':
    main()
