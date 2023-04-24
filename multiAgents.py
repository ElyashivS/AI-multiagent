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
        minimaxValue, action = self.calculateMinimaxValue(gameState, 0, depth=0)
        return action

    def calculateMinimaxValue(self, gameState, agent, depth):
        # If we have no where to go, or the depth reach his max, or lost, or we won.
        if (not (gameState.getLegalActions(agent))) \
                or (depth >= self.depth) \
                or (gameState.isLose()) \
                or (gameState.isWin()):
            return self.evaluationFunction(gameState), None

        newStates = \
            [(gameState.generateSuccessor(agent, action), action) for action in gameState.getLegalActions(agent)]
        nextAgent = (agent + 1) % gameState.getNumAgents()
        if nextAgent == 0:
            depth += 1

        miniMaxes = [(self.calculateMinimaxValue(state, nextAgent, depth)[0], action) for state, action in newStates]
        if agent == 0:  # Pacman plays
            return max(miniMaxes)
        else:  # Ghost plays
            return min(miniMaxes)

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
        action = self.alphaBetaPruning(state=gameState,
                                       depth=self.depth,
                                       agent=0,
                                       alpha=float('-inf'),
                                       beta=float('inf'))[1]

        return action

    def alphaBetaPruning(self, state, depth, agent, alpha, beta):
        # If we won, lost, no action left, or the assessment is returned by the minimax algorithm's maximum depth
        if (state.isWin()) or (state.isLose()) or (depth == 0) or (len(state.getLegalActions(agent)) == 0):
            return self.evaluationFunction(state), None

        # Pacman turn
        if agent == 0:
            return self.maxValue(state, depth, agent, alpha, beta)
        # Ghost turn
        else:
            return self.minValue(state, depth, agent, alpha, beta)

    def maxValue(self, state, depth, agent, alpha, beta):

        legalActions = state.getLegalActions(agent)  # Retrieve legal actions for Pacman

        maxValue = float('-inf')  # Init max value to minus infinity
        maxAction = None

        for action in legalActions:  # Get value of the successor state for each action
            successorState = state.generateSuccessor(agent, action)

            # Get value of the successor state for each action of ghost
            value = self.alphaBetaPruning(successorState, depth, agent + 1, alpha, beta)[0]

            # Update the maximum value and action if the value surpasses the present maximum
            if value > maxValue:
                maxValue = value
                maxAction = action

            # We can prune the branch if the value exceeds beta.
            if maxValue > beta:
                return maxValue, maxAction

            # Update the value of alpha
            alpha = max(alpha, maxValue)

        return maxValue, maxAction

    def minValue(self, state, depth, agent, alpha, beta):

        legalActions = state.getLegalActions(agent)  # Get legal actions for ghost agent

        minVal = float('inf')  # Initialize the minimum value to the highest possible value
        minAction = None

        for action in legalActions:  # For each legal action, get the value of the successor state
            successorState = state.generateSuccessor(agent, action)

            ghostNumber = state.getNumAgents() - 1  # Ghosts number

            if agent == ghostNumber:  # In case we have reached the last ghost, we return to Pacman.
                # Value of successor
                value = self.alphaBetaPruning(successorState, depth - 1, 0, alpha, beta)[0]
            else:  # Otherwise, we'll go to the next ghost
                # Value of the successor
                value = self.alphaBetaPruning(successorState, depth, agent + 1, alpha, beta)[0]

            # Update the minimum value and action if the value is lower than the current minimum.
            if value < minVal:
                minVal = value
                minAction = action

            # We can prune the branch if the value less than alpha.
            if minVal < alpha:
                return minVal, minAction

            # Update beta
            beta = min(beta, minVal)

        return minVal, minAction


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
        minimaxValue, action = self.calculateMinimaxValue(gameState, 0, depth=0)
        return action

    def calculateMinimaxValue(self, gameState, agent, depth):  # Same as line 167
        if (not (gameState.getLegalActions(agent))) \
                or (depth >= self.depth) \
                or (gameState.isLose()) \
                or (gameState.isWin()):
            return self.evaluationFunction(gameState), None

        newStates = \
            [(gameState.generateSuccessor(agent, action), action) for action in
                     gameState.getLegalActions(agent)]
        nextAgent = (agent + 1) % gameState.getNumAgents()
        if nextAgent == 0:
            depth += 1

        minimaxes = [(self.calculateMinimaxValue(state, nextAgent, depth)[0], action) for state, action in newStates]
        # Pacman turn
        if agent == 0:
            return max(minimaxes)
        # Ghost turn
        else:
            minimaxes_vals = [x[0] for x in minimaxes]
            avg = sum(minimaxes_vals) / float(len(minimaxes_vals))

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


    foodList = currentGameState.getFood().asList()
    currentPosition = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()

    activeGhosts = []
    scaredGhosts = []

    for ghost in ghostStates:
        if ghost.scaredTimer:
            scaredGhosts.append(ghost.getPosition())
        else:
            activeGhosts.append(ghost.getPosition())

    # Base evaluation = 10 * current game state
    evaluation = 10 * currentGameState.getScore()

    # Evaluation = Current evaluation (-15 * number of foods that left))
    evaluation += - 15 * len(foodList)

    # Evaluation = Current evaluation -15
    # (If there is food left on the game)
    foodDistances = [manhattanDistance(currentPosition, food) for food in foodList]
    sortedFoodsDistances = sorted(foodDistances)

    almostClosestFoodsSum = sum(sortedFoodsDistances[-4:])
    mostClosestFoodsSum = sum(sortedFoodsDistances[-2:])

    if len(sortedFoodsDistances) > 0:
        evaluation += - 15.

    # Lower the evaluation for close food
    # Evaluation = Current evaluation + (15 / <Sum the distance from Pacman to the top 2 foods that are the closest>)
    # + (10 / <Sum the distance from Pacman to the top 4 foods that are the closest>)
    evaluation += (10. / almostClosestFoodsSum) + (15. / mostClosestFoodsSum)

    # Update evaluation by the distances of active ghosts
    # Evaluation = Current evaluation (-15) * (1 / <Distance to the closest active ghosts>)
    activeGhostDistances = [manhattanDistance(currentPosition, ghost) for ghost in activeGhosts]

    if len(activeGhostDistances) > 0:
        evaluation += - 15. / min(activeGhostDistances)

    # Update evaluation by the distances to the scared ghosts that closest
    # Evaluation = Current evaluation (+15) * (1 / <Distance to the scared ghosts that closest>)
    scaredGhostDistances = [manhattanDistance(currentPosition, ghost) for ghost in scaredGhosts]

    if len(scaredGhostDistances) > 0:
        evaluation += + 15. / min(scaredGhostDistances)

    return evaluation


# Abbreviation
better = betterEvaluationFunction
