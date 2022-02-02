import numpy as np

class CartpoleEnv:
    def __init__(self):
        # Define Constants for Objects
        self.cart_mass = 1
        self.pole_mass = 0.1
        self.total_mass = self.cart_mass + self.pole_mass
        self.pole_length = 0.5
        self.force_magnitude = 10.0

        # Define Constants for simulation
        self.gravity = 9.81
        self.dt = 0.01

        # Define Simulation Ending Criteria
        self.x_limit = 2.5 # Cart can move 2.5 m in each direction
        self.theta_limit = 15 * (np.pi/180) # Pole can tip to maximum 15 degrees to each side

        # Initialize blank variables to store state information
        self.state = None

    def step(self, action):
        # Get cart information from state
        (x, x_dot, theta, theta_dot) = self.state

        # Get direction of force in cart from action
        if action == 0:
            force = self.force_magnitude
        else:
            force = -self.force_magnitude

        
        # Apply discretized integration
        
        # theta Acceleration
        theta_ddot = self.gravity * np.sin(theta) + np.cos(theta)*((-force - self.pole_mass*self.pole_length * theta_dot**2 * np.sin(theta)) / self.total_mass )  / \
        (self.pole_length * (4/3 - (self.pole_mass * np.cos(theta)**2)/(self.total_mass)))

        # x Acceleration
        x_ddot = ((force + self.pole_length * theta_dot**2 * np.sin(theta)) / self.total_mass) \
            - self.pole_length * theta_ddot * np.cos(theta) / self.total_mass

        # Update state values
        x = x + self.dt * x_dot
        x_dot = x_dot + self.dt * x_ddot
        theta = theta + self.dt * theta_dot
        theta_dot = theta + self.dt * theta_ddot

        # Check if new values are within simulation bounds
        done = (
            x < -self.x_limit or
            x > self.x_limit or
            theta < -self.theta_limit or
            theta > self.theta_limit
        )

        # Calculate reward
        if done:
            reward = 0.0
        else:
            reward = 1.0

        # Return new state
        self.state = (x, x_dot, theta, theta_dot)
        return (self.state, reward, done)

    def reset(self):
        '''
        Resets the environment for the next episode
        '''
        # Creates random state where pole is standing (x, x_dot, theta, theta_dot)
        self.state = np.random.random((4,)) * 0.1 - 0.05
        return self.state

def createAgent():
    from ann import ArticifialNeuralNetwork
    self.model = ArtificialNeuralNetwork()

def main():
    env = CartpoleEnv()
    env.createAgent()
    env.reset()
    for i in range(1000):
        state, reward, done = env.step(round(np.random.random()))
        print(state,reward,done)
        if done == True:
            print(i)
            break

if __name__ == '__main__':
    main()
