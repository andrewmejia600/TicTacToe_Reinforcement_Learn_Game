
#The following coded is heavily borrowed from
#https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter01/tic_tac_toe.py
#IMPLEMENTED QLEARN alogrithm to update the Bellman equation with a decay parameter

#######################################################################
# Copyright (C)                                                       #
# 2016 - 2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)           #
# 2016 Jan Hakenberg(jan.hakenberg@gmail.com)                         #
# 2016 Tian Jun(tianjun.cpp@gmail.com)                                #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
#######################################################################
# 2020 Andrew Mejia                                                   #
#######################################################################


import numpy as np
import random
import pickle

#Creating board game dimension space
BOARD_ROWS = 3
BOARD_COLS = 3
BOARD_SIZE = BOARD_ROWS * BOARD_COLS


class State:
    def __init__(self):
        # the board is represented by an n * n array,
        # 1 represents a chessman of the player who moves first,
        # -1 represents a chessman of another player
        # 0 represents an empty position
        #Board holder
        self.data = np.zeros((BOARD_ROWS, BOARD_COLS))
        #winner holder
        self.winner = None
        #hash holder
        self.hash_val = None
        #sate of board end holder
        self.end = None

    # Need a helper function to compute unique hash of one state
    def hash(self):
        #Initialize the hash value count
        if self.hash_val is None:
            self.hash_val = 0
            #iterate over the board matrix to a flattened space
            for i in np.nditer(self.data):
                #intialize self hash function and count by multiplying iterator by
                #3 and adding 1, since iterator will be 0 from matrix.
                self.hash_val = self.hash_val * 3 + i + 1
        #return the unique hash of the state of the game
        return self.hash_val

    # check whether a player has won the game, or it's a tie
    def is_end(self):
        #check to see if the end exists
        if self.end is not None:
            return self.end
        #Need a results list for all the board entries
        results = []
        # check board rows
        for i in range(BOARD_ROWS):
        # check by summing the rows
            results.append(np.sum(self.data[i, :]))
        # check columns by summing the columns
        for i in range(BOARD_COLS):
            results.append(np.sum(self.data[:, i]))

        # check diagonals of the board via trace procedure
        trace = 0
        reverse_trace = 0
        # for the length of the rows of the board
        for i in range(BOARD_ROWS):
        #the trace begins on the board with the row and column i and iterate through
            trace += self.data[i, i]
        #the reverese trace begins at 0 but is initialized at row i and goes from -i, the last position
            reverse_trace += self.data[i, BOARD_ROWS - 1 - i]
        results.append(trace)
        results.append(reverse_trace)


        for result in results:
            # if the row and column sums of the result is 3 then player 1 is the winner and the game ends
            if result == 3:
                self.winner = 1
                self.end = True
                return self.end
            # if the row and column sums of the result is -3 then player -1 is the winner and the game ends
            if result == -3:
                self.winner = -1
                self.end = True
                return self.end

        # whether it's a tie, meaning the absolute value sum of the board
        #positions is equal to the board size, where all positions
        #are taken, there is no winner and the game
        #ends
        sum_values = np.sum(np.abs(self.data))
        if sum_values == BOARD_SIZE:
            self.winner = 0
            self.end = True
            return self.end

        # game is still going on
        self.end = False
        return self.end

    # @symbol: 1 or -1
    # put chessman symbol in position (i, j)
    def next_state(self, i, j, symbol):
        #Generate a new state of the board for the next position move
        new_state = State()
        #First make a copy of the state of the board and save it as the new state instance
        new_state.data = np.copy(self.data)
        #The new state will have the position added at position [i,j]
        new_state.data[i, j] = symbol
        return new_state

    # print the board
    def print_state(self):
        #statrting with the rows print a | for spacing characters on the rows
        for i in range(BOARD_ROWS):
            print('-------------')
            out = '| '
            for j in range(BOARD_COLS):
                #if the row, column value is qual to 1, player 1, then mark *
                if self.data[i, j] == 1:
                    token = '*'
                # if the row, column value is equal to -1, player 2, then mark X
                elif self.data[i, j] == -1:
                    token = 'x'
                # for board positions that are open mark the baord 0.
                else:
                    token = '0'
                out += token + ' | '
            print(out)
        print('-------------')

# Helper function to get all states of the game it will take the current state, the current symbol and all satates
def get_all_states_impl(current_state, current_symbol, all_states):
    #for all the rows and columns
    for i in range(BOARD_ROWS):
        for j in range(BOARD_COLS):
            #if the i,j entry is equal to an empty space
            if current_state.data[i][j] == 0:
            #the new states is the current state of i,j and the current symbol
                new_state = current_state.next_state(i, j, current_symbol)
            #we hash the new state with the unique hash
                new_hash = new_state.hash()
            #we check to see if the hash is in all states
                if new_hash not in all_states:
            #if has is not in all states, the game had ended
                    is_end = new_state.is_end()
                    all_states[new_hash] = (new_state, is_end)
            #if the game is continouing, then recursively get the newest symbol for the state of the game
                    if not is_end:
                        get_all_states_impl(new_state, -current_symbol, all_states)


#Helper function to generate all states of the board configurations

def get_all_states():
    current_symbol = 1
    current_state = State()
    #all states is an empty dict to hold the states of the game
    all_states = dict()
    #fill in kev value pairs of all states of the game based on the hashing function
    all_states[current_state.hash()] = (current_state, current_state.is_end())
    get_all_states_impl(current_state, current_symbol, all_states)
    return all_states


# all possible board configurations
all_states = get_all_states()


class Judger:
    # @player1: the player who will move first, its chessman will be 1
    # @player2: another player with a chessman -1
    def __init__(self, player1, player2):
        self.p1 = player1
        self.p2 = player2
        self.current_player = None
        self.p1_symbol = 1
        self.p2_symbol = -1
        self.p1.set_symbol(self.p1_symbol)
        self.p2.set_symbol(self.p2_symbol)
        self.current_state = State()
#reset players in the judger class
    def reset(self):
        self.p1.reset()
        self.p2.reset()
#Allow players to alternate turns, by yielding player 1 and then player 2 while game is playing
    def alternate(self):
        while True:
            yield self.p1
            yield self.p2

    # @print_state: if True, print each board during the game
    # define helper function for play
    def play(self, print_state=False):
        #intialize alternator to allow players to take turns
        alternator = self.alternate()
        #reset players
        self.reset()
        current_state = State()
        #set for players 1 and 2
        self.p1.set_state(current_state)
        self.p2.set_state(current_state)
        if print_state:
            current_state.print_state()
        #print out of the state of the board if true
        while True:
            player = next(alternator)
            i, j, symbol = player.act()
            next_state_hash = current_state.next_state(i, j, symbol).hash()
            current_state, is_end = all_states[next_state_hash]
            self.p1.set_state(current_state)
            self.p2.set_state(current_state)
            if print_state:
                current_state.print_state()
            if is_end:
                return current_state.winner


# RL ALGORITHM
class Player:
    # @step_size: the step size to update estimations
    # @epsilon: the probability to explore
    def __init__(self, step_size=0.1, epsilon=0.1):
        #Estimations of win rate probability as an empty dictionary
        self.estimations = dict()
        #Stepsize is a small positive fraction, and influences learning rate
        self.step_size = step_size
        # greedy action rate to be used when the random state generated is less than epsilon, the move is not a good one
        self.epsilon = epsilon
        #state of the game to be captured through the iterations
        self.states = []
        #Greedy play is the move that brought the alogrithm that it rated best
        self.greedy = []
        #intialize the symbol to an empty board
        self.symbol = 0
    #reset the states and greedy moves back to an empty list
    def reset(self):
        self.states = []
        self.greedy = []
    #set the state and greedy moves and append to states list and append greedy as true intially
    def set_state(self, state):
        self.states.append(state)
        self.greedy.append(True)

    def set_symbol(self, symbol):
        #symbol is empty 0
        self.symbol = symbol
        #for all hash values in all states, the tuple state and is_end is the states hash value and whether the state of games is win, loss or tie
        for hash_val in all_states:
            state, is_end = all_states[hash_val]
            # if is_end state
            if is_end:
                #if the state is a winner is equal to the symbol the player is playing, then the outcome is 1.0 and the hash value is a winner
                if state.winner == self.symbol:
                    self.estimations[hash_val] = 1.0
                #else if the state of the winner is a 0 the estimation of the hash value is .50 and a tie
                elif state.winner == 0:
                    # we need to distinguish between a tie and a lose
                    self.estimations[hash_val] = 0.5
                #otherwise no reward is given at the hash value index for the move and it is a lose
                else:
                    self.estimations[hash_val] = 0
            else:
                #the game continues
                self.estimations[hash_val] = 0.5

    # update value estimation, a temporal-difference method based on two estimates on the current and previous states
    #V(S_t) <- V(S_t) + step_size * [V(St+1) - V(S_t)]
    def backup(self):
        #Get list of hashes for states in game
        states = [state.hash() for state in self.states]
        #For the states in the most recent list of events
        for i in reversed(range(len(states) - 1)):
            state = states[i]
            #calculate the temporal-difference of the greedy move of the optimal future value from the old value
            # See act function where we zero out greedy estimate if guess is less than epsilon
            td_error = self.greedy[i] * (
                self.estimations[states[i + 1]] - self.estimations[state]
            )
            #The next estimation becomes the temporal difference error times the step size + the value estimation
            self.estimations[state] += self.step_size * td_error

    # choose an action based on the state
    def act(self):
        #State is the most recent move made from the previous state
        state = self.states[-1]
        #Make empty lists to store the next states and positions
        next_states = []
        next_positions = []
        #for all the board positions
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                #if there is an open board position the next position will append
                if state.data[i, j] == 0:
                    next_positions.append([i, j])
                    #the next state will append the state with the symbol and unique hash
                    next_states.append(state.next_state(
                        i, j, self.symbol).hash())
        #if random move is less than epsilon, the greedy estimation is False. This will zero out estimate as a bad choice
        if np.random.rand() < self.epsilon:
            #The next move on the board is the index within the next_positions list of touples
            action = next_positions[np.random.randint(len(next_positions))]
            #append the the player's symbol
            action.append(self.symbol)
            #this is a bad move, so the last greedy index is false to zero out TD error as a penality
            self.greedy[-1] = False
            return action
# values is an empty list to contain the hash value of the esimation and positions for the moves played and append as a touple
        values = []
        #for the hash value zipped to a dictionary of next state and next position  append the estimation and position
        for hash_val, pos in zip(next_states, next_positions):
            values.append((self.estimations[hash_val], pos))
        # to select one of the actions of equal value at random due to Python's sort is stable
        np.random.shuffle(values)
        #sort the values on the largest estimation
        values.sort(key=lambda x: x[0], reverse=True)
        #the next move to play is the highest value, with the move being the next position
        action = values[0][1]
        #append the action symbol for the move and return the action
        action.append(self.symbol)
        return action
#Save the policy to a binary file
    def save_policy(self):
        with open('policy_%s.bin' % ('first' if self.symbol == 1 else 'second'), 'wb') as f:
            pickle.dump(self.estimations, f)
#open the binary file of the policy
    def load_policy(self):
        with open('policy_%s.bin' % ('first' if self.symbol == 1 else 'second'), 'rb') as f:
            self.estimations = pickle.load(f)

# QLEARN ALGORITHM
class PlayerQ:
    # @step_size: the step size to update estimations
    # @epsilon: the probability to explore
    def __init__(self, step_size=0.1, epsilon=0.1, gamma = 0.8):
        #Estimations of win rate probability as an empty dictionary
        self.estimations = dict()
        #Stepsize is a small positive fraction, and influences learning rate
        self.step_size = step_size
        # greedy action rate to be used when the random state generated is less than epsilon, the move is not a good one
        self.epsilon = epsilon
        #Gamma is the discount or decay
        self.gamma = gamma
        #state of the game to be captured through the iterations
        self.states = []
        #Greedy play is the move that brought the alogrithm that it rated best
        self.greedy = []
        #intialize the symbol to an empty board
        self.symbol = 0
    #reset the states and greedy moves back to an empty list
    def reset(self):
        self.states = []
        self.greedy = []
    #set the state and greedy moves and append to states list and append greedy as true intially
    def set_state(self, state):
        self.states.append(state)
        self.greedy.append(True)

    def set_symbol(self, symbol):
        #symbol is empty 0
        self.symbol = symbol
        #for all hash values in all states, the tuple state and is_end is the states hash value and whether the state of games is win, loss or tie
        for hash_val in all_states:
            state, is_end = all_states[hash_val]
            # if is_end state
            if is_end:
                #if the state is a winner is equal to the symbol the player is playing, then the outcome is 1.0 and the hash value is a winner
                if state.winner == self.symbol:
                    self.estimations[hash_val] = 1.0
                #else if the state of the winner is a 0 the estimation of the hash value is .50 and a tie
                elif state.winner == 0:
                    # we need to distinguish between a tie and a lose
                    self.estimations[hash_val] = 0.5
                #otherwise no reward is given at the hash value index for the move and it is a lose
                else:
                    self.estimations[hash_val] = 0
            else:
                #the game continues
                self.estimations[hash_val] = 0.5

    # update value estimation, a temporal-difference method based on two estimates on the current and previous states
    #V(S_t) <- V(S_t) + step_size * [V(St+1) - V(S_t)]
    def backup(self):
        #Get list of hashes for states in game
        states = [state.hash() for state in self.states]
        #For the states in the most recent list of events
        for i in reversed(range(len(states) - 1)):
            state = states[i]
            #calculate the temporal-difference of the greedy move of the optimal future value from the old value
            # See act function where we zero out greedy estimate if guess is less than epsilon
            td_error = self.greedy[i] * ( self.gamma * (
                self.estimations[states[i + 1]] - self.estimations[state]
            ))
            #The next estimation becomes the temporal difference error times the step size + the value estimation
            self.estimations[state] += self.step_size * td_error

    # choose an action based on the state
    def act(self):
        #State is the most recent move made from the previous state
        state = self.states[-1]
        #Make empty lists to store the next states and positions
        next_states = []
        next_positions = []
        #for all the board positions
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                #if there is an open board position the next position will append
                if state.data[i, j] == 0:
                    next_positions.append([i, j])
                    #the next state will append the state with the symbol and unique hash
                    next_states.append(state.next_state(
                        i, j, self.symbol).hash())
        #if random move is less than epsilon, the greedy estimation is False. This will zero out estimate as a bad choice
        if random.uniform(0, 1) < self.epsilon:
            #The next move on the board is the index within the next_positions list of touples
            action = next_positions[np.random.randint(len(next_positions))]
            #append the the player's symbol
            action.append(self.symbol)
            #this is a bad move, so the last greedy index is false to zero out TD error as a penality
            self.greedy[-1] = False
            return action

# values is an empty list to contain the hash value of the esimation and positions for the moves played and append as a touple
        values = []
        #for the hash value zipped to a dictionary of next state and next position  append the estimation and position
        for hash_val, pos in zip(next_states, next_positions):
            values.append((self.estimations[hash_val], pos))
        # to select one of the actions of equal value at random due to Python's sort is stable
        np.random.shuffle(values)
        #sort the values on the largest/max estimation
        values.sort(key=lambda x: x[0], reverse=True)
        #the next move to play is the highest value, with the move being the next position
        action = values[0][1]
        #append the action symbol for the move and return the action
        action.append(self.symbol)
        return action
#Save the policy to a binary file
    def save_policy(self):
        with open('policyQ_%s.bin' % ('first' if self.symbol == 1 else 'second'), 'wb') as g:
            pickle.dump(self.estimations, g)
#open the binary file of the policy
    def load_policy(self):
        with open('policyQ_%s.bin' % ('first' if self.symbol == 1 else 'second'), 'rb') as g:
            self.estimations = pickle.load(g)

###

# human interface
# input a number to put a chessman
# | q | w | e |
# | a | s | d |
# | z | x | c |
class HumanPlayer:
    def __init__(self, **kwargs):
        self.symbol = None
        self.keys = ['q', 'w', 'e', 'a', 's', 'd', 'z', 'x', 'c']
        self.state = None

    def reset(self):
        pass

    def set_state(self, state):
        self.state = state

    def set_symbol(self, symbol):
        self.symbol = symbol

    def act(self):
        self.state.print_state()
        print(self.keys)
        key = input("Input your position:")
        data = self.keys.index(key)
        i = data // BOARD_COLS
        j = data % BOARD_COLS
        return i, j, self.symbol

# print the learning rate of the players
def train(epochs, print_every_n=500):
    player1 = Player(step_size = .1, epsilon=0.01)
    player2 = PlayerQ(step_size = .1, epsilon=0.01, gamma = .90)
    judger = Judger(player1, player2)
    player1_win = 0.0
    player2_win = 0.0
    for i in range(1, epochs + 1):
        winner = judger.play(print_state=False)
        if winner == 1:
            player1_win += 1
        if winner == -1:
            player2_win += 1
        if i % print_every_n == 0:
            print('Epoch %d, player 1 winrate: %.02f, player 2 winrate: %.02f' % (i, player1_win / i, player2_win / i))
        player1.backup()
        player2.backup()
        judger.reset()
    player1.save_policy()
    player2.save_policy()

# print the learning rate of the players
def compete(turns):
    player1 = Player(epsilon=0.10)
    player2 = PlayerQ(epsilon=0.10)
    judger = Judger(player1, player2)
    player1.load_policy()
    player2.load_policy()
    player1_win = 0.0
    player2_win = 0.0
    for _ in range(turns):
        winner = judger.play()
        if winner == 1:
            player1_win += 1
        if winner == -1:
            player2_win += 1
        judger.reset()
    print('%d turns, player 1 win %.02f, player 2 win %.02f' % (turns, player1_win / turns, player2_win / turns))


# The game is a zero sum game. If both players are playing with an optimal strategy, every game will end in a tie.
# So we test whether the AI can guarantee at least a tie if it goes second.
def play():
    while True:
        player1 = HumanPlayer()
        player2 = PlayerQ(epsilon=0.10)
        judger = Judger(player1, player2)
        player2.load_policy()
        winner = judger.play()
        if winner == player2.symbol:
            print("You lose!")
        elif winner == player1.symbol:
            print("You win!")
        else:
            print("It is a tie!")


if __name__ == '__main__':
    #train(int(1e5))
    #compete(int(1e3))
    train(int(1e4))
    compete(int(1e3))
    play()
