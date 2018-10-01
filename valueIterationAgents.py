# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here

        for k in range(self.iterations):
            new_values = util.Counter()

            for state in self.mdp.getStates():
                possible_actions = self.mdp.getPossibleActions(state)

                if(len(possible_actions) == 0):
                    new_values[state] = self.values[state]
                else:
                    action_evaluation_list = []

                    for action in possible_actions:
                        sum = 0.

                        for (next_state,proba) in self.mdp.getTransitionStatesAndProbs(state, action):
                            sum+= proba*(self.mdp.getReward(state, action, next_state) + self.discount * self.values[next_state])

                        action_evaluation_list.append(sum)
                    new_values[state] = max(action_evaluation_list)

            self.values = new_values



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        res = 0.
        for next_state, proba in self.mdp.getTransitionStatesAndProbs(state, action):
            res+= proba*(self.mdp.getReward(state, action, next_state) + self.discount * self.values[next_state])

        return res


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        possible_actions = self.mdp.getPossibleActions(state)

        # for action in possible_actions:
        #     sum = 0.
        #     for (next_state,proba) in self.mdp.getTransitionStatesAndProbs(state, action):
        #         sum+= proba*(self.mdp.getReward(state, action, next_state) + self.discount * self.values[next_state])
        #
        #     if(abs(sum - self.values[state])<0.01):
        #         return action



        for action in possible_actions:
            if abs(self.computeQValueFromValues(state, action) - self.values[state]) < 0.01:
                return(action)

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        counter = 0
        while(counter<self.iterations):

            for state in self.mdp.getStates():
                counter+=1
                if(counter>self.iterations):
                    break
                possible_actions = self.mdp.getPossibleActions(state)

                if(len(possible_actions) == 0):
                    x = 3
                else:
                    action_evaluation_list = []

                    for action in possible_actions:
                        sum = 0.

                        for (next_state,proba) in self.mdp.getTransitionStatesAndProbs(state, action):
                            sum+= proba*(self.mdp.getReward(state, action, next_state) + self.discount * self.values[next_state])

                        action_evaluation_list.append(sum)
                    self.values[state] = max(action_evaluation_list)

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def predecessor_computation(self):
        """
        """
        dic = {}
        for state in self.mdp.getStates():
            dic[state] = set()

        for state in self.mdp.getStates():
            possible_actions  = self.mdp.getPossibleActions(state)

            for action in possible_actions:
                for (next_state, proba) in self.mdp.getTransitionStatesAndProbs(state, action):
                    if(proba>0):
                        dic[next_state].add(state)
        return dic

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        predecessor_dic = self.predecessor_computation()
        queue = util.PriorityQueue()

        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                possible_actions = self.mdp.getPossibleActions(state)
                diff = abs(self.values[state] - max([self.computeQValueFromValues(state, action) for action in possible_actions]))
                queue.push(state, -diff)

        for i in range(self.iterations):
            if(queue.isEmpty()):
                break
            else:
                state = queue.pop()
                possible_actions = self.mdp.getPossibleActions(state)
                self.values[state] = max([self.computeQValueFromValues(state, action) for action in possible_actions])

                for predecessor in predecessor_dic[state]:
                    possible_actions = self.mdp.getPossibleActions(predecessor)
                    diff = abs(self.values[predecessor] - max([self.computeQValueFromValues(predecessor, action) for action in possible_actions]))
                    if(diff > self.theta):
                        queue.update(predecessor, -diff)
