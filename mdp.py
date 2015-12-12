import numpy as np
from transitionModel import createTransitionModel



class DrivingMDP():

    def __init__(self, epsilon=0.001, gamma = 0.1):

      self.epsilon = epsilon
      self.gamma = gamma
      self.states = ['Upper East Side', 'Upper West Side', 'East Harlem', 'Harlem', 'Washington Heights', 'Chelsea', "Hell's Kitchen", 'Midtown', 'Midtown East', 'Murray Hill and Gramercy', 'East Village', 'West Village', 'Greenwich Village', 'Financial District', 'Lower East Side', 'Soho', 'Central Park', 'Laguardia Airport', 'JFK Airport']

      self.actions = ['North','East','South', 'West','Commit']

      self.numStates = len(self.states)

      self.transitionModel, self.pickupProb = createTransitionModel()
      #print pickupProb
      self.failureReward = -2. # reward for not finding a customer at cur location
      """
      create a matrix with rows representing neighborhoods, and columns for north east south west,
      with entries encoding which neighborhood you end up in if you are in neighborhood i and drive in direction j
      """
      self.trans = np.array([
              ['East Harlem',0,'Midtown East','Central Park'],            #Upper East
              ['Harlem','Central Park',"Hell's Kitchen",0],               #Upper West
              ['Washington Heights',0, 'Upper East Side', 'Harlem'],          #etc.
              ['Washington Heights','East Harlem','Upper West Side',0],
              [0,'East Harlem',0,'East Harlem'],
              ["Hell's Kitchen",'Murray Hill and Gramercy', 'West Village',0],
              ['Upper West Side','Midtown','Chelsea',0],
              ['Central Park','Midtown East','Murray Hill and Gramercy',"Hell's Kitchen"],
              ['Upper East Side',0,'Murray Hill and Gramercy', 'Midtown'],
              ['Midtown East',0, 'East Village','Chelsea'],
              ['Murray Hill and Gramercy',0,'Lower East Side','Greenwich Village'],
              ['Chelsea','Greenwich Village','Soho',0],
              ['Chelsea','East Village','Soho','West Village'],
              [0,'Lower East Side',0,'Soho'],
              ['East Village',0,'Financial District','Soho'],
              ['West Village','Lower East Side','Financial District',0],
              ['Harlem', 'Upper East Side','Midtown','Upper West Side'],
              [0,0,0,0],              #Laguardia
              [0,0,0,0]                #JFK
          ])
    def validateState(self, state):
      return state in self.states

    def getActions(self, state):
      possible_actions = []
      state_idx = self.states.index(state)
      for action_idx,action in enumerate(self.actions):
        if action == "Commit" or self.trans[state_idx][action_idx] != '0':
          possible_actions.append(action)
      return possible_actions

    def determ_transition(self,newState):
        """function used for deterministic transition, will return a list of tuples [(state1,0),(state2,0),...(newState,1),...] """
        index = self.states.index(newState)
        probs = [0]*(index)+[1]+[0]*(self.numStates-index-1)
        return zip(self.states,probs)

    """uses the trans matrix to look up successor state for deterministic 'drive away' """
    def successorState(self, state, action):
            row = self.states.index(state)
            col = self.actions.index(action)
            return self.trans[row][col]

    def T(self, state, action):
        """Transition model.  From a state and an action, return a list
        of (result-state, probability) pairs."""
        if action == 'Commit':
            keys = self.transitionModel[state].keys() #all the next states
            keys.append(None) # account for failure
            probs = [item[0] * self.pickupProb[state] for item in self.transitionModel[state].values()] #the prob values of the next states
            probs.append(1. - self.pickupProb[state]) #probability you don't pick up anyone
            return zip(keys,probs) #return the tuple
        #else choose to drive elsewhere
        else:
            newState = self.successorState(state,action)
            return self.determ_transition(newState)

    # def commit_expect(state):
    #     expectations = [item[0]*item[1] for item in transitionModel[state].values()]
    #     return sum(expectations)

    """
    defining R(s,a,s')
    """
    def R(self, state, action, state1):
        "Return a numeric reward for this state."
        if action == 'Commit':
            if state1:
                return self.transitionModel[state][state1][1]
            else:
                return self.failureReward
        else:
            newState = self.successorState(state,action)
            if newState not in self.transitionModel[state]:
              return 0
            else:
              return self.transitionModel[state][newState][1]

    #think value iteration is good to go
    def value_iteration(self):
        "Solving an MDP by value iteration."
        U1 = {s: 0. for s in self.states}
        foo = 0
        while True:
            U = U1.copy()
            delta = 0
            for s in self.states:
              max_of_actions = -99999999
              for a in self.getActions(s):
                sum_of_reward_over_states = 0
                for (s1, p) in self.T(s, a):
                  if s1 != None:
                    sum_of_reward_over_states += p * (self.R(s,a,s1) + self.gamma * U[s1])
                  else:
                    sum_of_reward_over_states += p * (self.R(s,a,s1) + self.gamma * U[s])

                if sum_of_reward_over_states > max_of_actions:
                  max_of_actions = sum_of_reward_over_states
                  U1[s] = sum_of_reward_over_states
                delta = abs(U1[s] - U[s])
            if delta < self.epsilon :
                 return U


