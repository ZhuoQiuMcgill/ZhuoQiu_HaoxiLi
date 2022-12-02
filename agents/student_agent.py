# Student agent: Add your own agent here
import copy
import random

from agents.agent import Agent
from store import register_agent
import sys


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.autoplay = True

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        valid_moves = self.get_move_area(chess_board, my_pos, adv_pos, max_step)
        if len(valid_moves) == 0:
            next_pos = my_pos
        else:
            i = random.randint(0, len(valid_moves) - 1)
            next_pos = valid_moves[i]
        # print(valid_moves, "\n", next_pos)
        next_dir = 0
        for key in self.dir_map:
            if not chess_board[next_pos[0]][next_pos[1]][self.dir_map[key]]:
                next_dir = self.dir_map[key]
                break
        return next_pos, next_dir


class Board:
    def __init__(self, chess_board, my_pos, adv_pos, max_step):
        pass

    def __copy__(self):
        pass

    def check_endgame(self):
        pass

    def move_to(self, move):
        """

        return deep copy of the board after move
        """
        pass


class MCSTree:
    def __init__(self, board):
        pass

    class Node:
        def __init__(self, father, move, max_sim):
            self.father = father
            self.wins = 0
            self.simulations = 0
            self.max_sim = max_sim
            self.chess_board = father.move_to(move)
            self.simulation_board = copy.deepcopy(self.chess_board)

        def reset_board(self):
            self.simulation_board = copy.deepcopy(self.chess_board)

        def run_simulate(self):
            player1_win = 0
            player2_win = 0
            for _ in range(self.max_sim):
                pass






