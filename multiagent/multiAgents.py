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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        print 'newPos', newPos
        newFood = successorGameState.getFood()
        print 'newFood', newFood
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        print 'newScaredTimes', newScaredTimes

        
        return successorGameState.getScore()

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
      Your minimax agent (question 1)
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
        "*** YOUR CODE HERE ***"
        bestChoice = self.value(gameState, 0, 0)
        return bestChoice[1]
        #util.raiseNotDefined()

    def value(self, gameState, dep, agentIndex):
        if gameState.isLose() or gameState.isWin():
            return (self.evaluationFunction(gameState), Directions.STOP)
        if agentIndex == 0:
            return self.maxValue(gameState, dep, agentIndex)
        else:
            return self.minValue(gameState, dep, agentIndex)

    def maxValue(self, gameState, dep, agentIndex):
        if dep == self.depth:
            return (self.evaluationFunction(gameState), Directions.STOP)
        legalActions = gameState.getLegalActions(agentIndex)
        inf = -100000000
        maxValue = (inf, Directions.STOP)
        for action in legalActions:
            state = gameState.generateSuccessor(agentIndex, action)
            valueAndAction = self.value(state, dep, agentIndex + 1)
            if valueAndAction[0] > maxValue[0] or maxValue[0] == inf:
                maxValue = (valueAndAction[0], action)
        return maxValue

    def minValue(self, gameState, dep, agentIndex):
        legalActions = gameState.getLegalActions(agentIndex)
        inf = 100000000
        minValue = (inf, Directions.STOP)
        for action in legalActions:
            state = gameState.generateSuccessor(agentIndex, action)
            if agentIndex == gameState.getNumAgents() - 1 :
                valueAndAction = self.value(state, dep + 1, 0)
            else:
                valueAndAction = self.value(state, dep, agentIndex + 1)

            if valueAndAction[0] < minValue[0] or minValue[0] == inf:
                minValue = (valueAndAction[0], action)
        return minValue



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        inf = 100000000
        bestChoice = self.value(gameState, 0, 0, -inf, inf)
        return bestChoice[1]
        #util.raiseNotDefined()

    def value(self, gameState, dep, agentIndex, alpha, beta):
        if gameState.isLose() or gameState.isWin():
            return (self.evaluationFunction(gameState), Directions.STOP)
        if agentIndex == 0:
            return self.maxValue(gameState, dep, agentIndex, alpha, beta)
        else:
            return self.minValue(gameState, dep, agentIndex, alpha, beta)

    def maxValue(self, gameState, dep, agentIndex, alpha, beta):
        if dep == self.depth:
            return (self.evaluationFunction(gameState), Directions.STOP)
        legalActions = gameState.getLegalActions(agentIndex)
        inf = -100000000
        maxValue = (inf, Directions.STOP)
        for action in legalActions:
            state = gameState.generateSuccessor(agentIndex, action)
            valueAndAction = self.value(state, dep, agentIndex + 1, alpha, beta)
            if valueAndAction[0] > maxValue[0] or maxValue[0] == inf:
                maxValue = (valueAndAction[0], action)

            if maxValue[0] > beta:
                return maxValue
            alpha = max(alpha, maxValue[0])
        return maxValue

    def minValue(self, gameState, dep, agentIndex, alpha, beta):
        legalActions = gameState.getLegalActions(agentIndex)
        inf = 100000000
        minValue = (inf, Directions.STOP)
        for action in legalActions:
            state = gameState.generateSuccessor(agentIndex, action)
            if agentIndex == gameState.getNumAgents() - 1 :
                valueAndAction = self.value(state, dep + 1, 0, alpha, beta)
            else:
                valueAndAction = self.value(state, dep, agentIndex + 1, alpha, beta)

            if valueAndAction[0] < minValue[0] or minValue[0] == inf:
                minValue = (valueAndAction[0], action)

            if minValue[0] < alpha:
                return minValue
            beta = min(beta, minValue[0])

        return minValue

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        bestChoice = self.value(gameState, 0, 0)
        return bestChoice[1]
        #util.raiseNotDefined()
    def value(self, gameState, dep, agentIndex):
        if gameState.isLose() or gameState.isWin():
            return (self.evaluationFunction(gameState), Directions.STOP)
        if agentIndex == 0:
            return self.maxValue(gameState, dep, agentIndex)
        else:
            return self.expectValue(gameState, dep, agentIndex)

    def maxValue(self, gameState, dep, agentIndex):
        if dep == self.depth:
            return (self.evaluationFunction(gameState), Directions.STOP)
        legalActions = gameState.getLegalActions(agentIndex)
        inf = -100000000
        maxValue = (inf, Directions.STOP)
        for action in legalActions:
            state = gameState.generateSuccessor(agentIndex, action)
            valueAndAction = self.value(state, dep, agentIndex + 1)
            if valueAndAction[0] > maxValue[0] or maxValue[0] == inf:
                maxValue = (valueAndAction[0], action)
        return maxValue

    def expectValue(self, gameState, dep, agentIndex):
        legalActions = gameState.getLegalActions(agentIndex)
        sum = 0.0
        for action in legalActions:
            state = gameState.generateSuccessor(agentIndex, action)
            if agentIndex == gameState.getNumAgents() - 1 :
                valueAndAction = self.value(state, dep + 1, 0)
            else:
                valueAndAction = self.value(state, dep, agentIndex + 1)
            sum =  sum + valueAndAction[0]
        expectValue = (sum / len(legalActions), Directions.STOP)
        return expectValue

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 4).

      DESCRIPTION: <write something here so we know what you did>
      Evaluation the score base on:
      1.  The minium distance between pacman and ghost
      2.  The scaredTimes of ghost
      3.  Current display Score
    """
    "*** YOUR CODE HERE ***"
    pacmanPos = currentGameState.getPacmanPosition()
    score = currentGameState.getScore()
    for action in currentGameState.getLegalActions():
        state = currentGameState.generatePacmanSuccessor(action)
        pos = state.getPacmanPosition()
        if currentGameState.hasFood(pos[0], pos[1]):
            score = score + 5
    ghostStates = currentGameState.getGhostStates()
    ghostPostions = [ghostState.getPosition() for ghostState in ghostStates]
    distances = [manhattanDistance(gPos, pacmanPos) for gPos in ghostPostions]
    minDistances = min(distances)
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    allScaredTimes = max(scaredTimes)

    return score + minDistances * 1 + allScaredTimes * 2
    #util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

