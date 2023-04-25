# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** MY CODE HERE ***"

        # Don't ever stop
        if action == 'Stop':
            return float('-inf')  # Minus infinity

        for state in newGhostStates:
            #  If we can eat the ghost - we will. Otherwise we'll run away
            if state.getPosition() == newPos:
                if state.scaredTimer == 0:
                    return float('-inf')
                else:
                    return float('inf')

        if newFood.count() == 0:  # Victory
            return 0

        # Eating a food
        if newFood.count() != currentGameState.getFood().count():
            return float('inf')

        min_distance = min([manhattanDistance(newPos, x) for x in newFood.asList()])

        return (-1) * min_distance


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** MY CODE HERE ***"

        # Minimax value
        minimaxValue, action = self.calculate_minimax_value(gameState, 0, depth=0)
        return action

    def calculate_minimax_value(self, gameState, agent, depth):
        # If we have no where to go, or the depth reach his max, or lost, or we won.
        if (not (gameState.getLegalActions(agent))) \
                or (depth >= self.depth) \
                or (gameState.isLose()) \
                or (gameState.isWin()):
            return self.evaluationFunction(gameState), None

        new_states = \
            [(gameState.generateSuccessor(agent, action), action) for action in gameState.getLegalActions(agent)]
        next_agent = (agent + 1) % gameState.getNumAgents()
        if next_agent == 0:
            depth += 1

        mini_maxes = [(self.calculate_minimax_value(state, next_agent, depth)[0], action) for state, action in new_states]
        if agent == 0:  # Pacman plays
            return max(mini_maxes)
        else:  # Ghost plays
            return min(mini_maxes)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** MY CODE HERE ***"

        # Determine the correct action by minimax algorithm
        action = self.alpha_beta_pruning(state=gameState,
                                         depth=self.depth,
                                         agent=0,
                                         alpha=float('-inf'),
                                         beta=float('inf'))[1]

        return action

    def alpha_beta_pruning(self, state, depth, agent, alpha, beta):
        # If we won, lost, no action left, or the assessment is returned by the minimax algorithm's maximum depth
        if (state.isWin()) or (state.isLose()) or (depth == 0) or (len(state.getLegalActions(agent)) == 0):
            return self.evaluationFunction(state), None

        # Pacman turn
        if agent == 0:
            return self.max_value(state, depth, agent, alpha, beta)
        # Ghost turn
        else:
            return self.min_value(state, depth, agent, alpha, beta)

    def max_value(self, state, depth, agent, alpha, beta):

        legal_actions = state.getLegalActions(agent)  # Retrieve legal actions for Pacman

        max_action = None
        max_value = float('-inf')  # Init max value to minus infinity

        for action in legal_actions:  # Get value of the successor state for each action
            successor_state = state.generateSuccessor(agent, action)

            # Get value of the successor state for each action of ghost
            value = self.alpha_beta_pruning(successor_state, depth, agent + 1, alpha, beta)[0]

            # Update the maximum value and action if the value surpasses the present maximum
            if value > max_value:
                max_value = value
                max_action = action

            # We can prune the branch if the value exceeds beta.
            if max_value > beta:
                return max_value, max_action

            # Update the value of alpha
            alpha = max(alpha, max_value)

        return max_value, max_action

    def min_value(self, state, depth, agent, alpha, beta):

        legal_actions = state.getLegalActions(agent)  # Get legal actions for ghost agent

        min_val = float('inf')  # Initialize the minimum value to the highest possible value
        min_action = None

        for action in legal_actions:  # For each legal action, get the value of the successor state
            successor_state = state.generateSuccessor(agent, action)

            ghost_number = state.getNumAgents() - 1  # Ghosts number

            if agent == ghost_number:  # In case we have reached the last ghost, we return to Pacman.
                # Value of successor
                value = self.alpha_beta_pruning(successor_state, depth - 1, 0, alpha, beta)[0]
            else:  # Otherwise, we'll go to the next ghost
                # Value of the successor
                value = self.alpha_beta_pruning(successor_state, depth, agent + 1, alpha, beta)[0]

            # Update the minimum value and action if the value is lower than the current minimum.
            if value < min_val:
                min_val = value
                min_action = action

            # We can prune the branch if the value less than alpha.
            if min_val < alpha:
                return min_val, min_action

            # Update beta
            beta = min(beta, min_val)

        return min_val, min_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      My expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** MY CODE HERE ***"
        minimaxValue, action = self.calculate_minimax_value(gameState, 0, depth=0)
        return action

    def calculate_minimax_value(self, gameState, agent, depth):  # Same as line 167
        if (not (gameState.getLegalActions(agent))) \
                or (depth >= self.depth) \
                or (gameState.isLose()) \
                or (gameState.isWin()):
            return self.evaluationFunction(gameState), None

        new_states = \
            [(gameState.generateSuccessor(agent, action), action) for action in
                     gameState.getLegalActions(agent)]
        next_agent = (agent + 1) % gameState.getNumAgents()
        if next_agent == 0:
            depth += 1

        mini_maxes = [(self.calculate_minimax_value(state, next_agent, depth)[0], action) for state, action in new_states]
        # Pacman turn
        if agent == 0:
            return max(mini_maxes)
        # Ghost turn
        else:
            mini_maxes_values = [x[0] for x in mini_maxes]
            avg = sum(mini_maxes_values) / float(len(mini_maxes_values))

            # since it's an average, we're losing the data of which direction we came from. We pass None
            return avg, None


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    Description:

    Defines an evaluation function for Pacman.
    It calculates a score based on the remaining food pellets, distances to ghosts, and Pacman's score.
    The evaluation returns a high score if Pacman has won, and a low score if Pacman has lost.

    The function calculates the distances from Pacman to different ghosts, and remaining food pellets.
    It updates the evaluation score based on the number of remaining food pellets and the distances to ghosts.

    The exact amount of score described in below



    """
    "*** MY CODE HERE ***"

    # Win situation
    if currentGameState.isWin():
        return 10000000.

    # Lose situation
    if currentGameState.isLose():
        return -10000000.

    ghost_states = currentGameState.getGhostStates()
    current_position = currentGameState.getPacmanPosition()
    food_list = currentGameState.getFood().asList()

    normal_ghosts = []
    scared_ghosts = []

    for ghost in ghost_states:
        if ghost.scaredTimer:
            scared_ghosts.append(ghost.getPosition())
        else:
            normal_ghosts.append(ghost.getPosition())

    # Base evaluation = 10 * current game state
    evaluation = 10 * currentGameState.getScore()

    # Evaluation = Current evaluation (-15 * number of foods that left))
    evaluation += - 15 * len(food_list)

    # Evaluation = Current evaluation -15
    # (If there is food left on the game)
    food_distances = [manhattanDistance(current_position, food) for food in food_list]
    sort_foods_distances = sorted(food_distances)

    almost_closest_foods_sum = sum(sort_foods_distances[-4:])
    most_closest_foods_sum = sum(sort_foods_distances[-2:])

    if len(sort_foods_distances) > 0:
        evaluation += - 15.

    # Lower the evaluation for close food
    # Evaluation = Current evaluation + (15 / <Sum the distance from Pacman to the top 2 foods that are the closest>)
    # + (10 / <Sum the distance from Pacman to the top 4 foods that are the closest>)
    evaluation += (10. / almost_closest_foods_sum) + (15. / most_closest_foods_sum)

    # Update evaluation by the distances of normal ghosts
    # Evaluation = Current evaluation (-15) * (1 / <Distance to the closest normal ghosts>)
    normal_ghost_distances = [manhattanDistance(current_position, ghost) for ghost in normal_ghosts]

    if len(normal_ghost_distances) > 0:
        evaluation += - 15. / min(normal_ghost_distances)

    # Update evaluation by the distances to the scared ghosts that closest
    # Evaluation = Current evaluation (+15) * (1 / <Distance to the scared ghosts that closest>)
    scared_ghost_distances = [manhattanDistance(current_position, ghost) for ghost in scared_ghosts]

    if len(scared_ghost_distances) > 0:
        evaluation += + 15. / min(scared_ghost_distances)

    return evaluation


# Abbreviation
better = betterEvaluationFunction
