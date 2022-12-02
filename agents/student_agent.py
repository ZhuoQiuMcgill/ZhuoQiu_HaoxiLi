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

    def deep_copy(self):
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
            self.chess_board = self.father.chess_board.deep_copy()
            self.chess_board.move_to(move)
            self.simulation_board = self.father.chess_board.deep_copy()

        def reset_board(self):
            self.simulation_board = copy.deepcopy(self.chess_board)

        def run_simulate(self):
            player1_win = 0
            player2_win = 0
            for _ in range(self.max_sim):
                pass

        def random_player_step(self):
            """
            this method simulate a random player's move
            """
            chess_board_r = self.simulation_board.chess_board
            max_step_r = self.simulation_board.max_step
            my_pos_r = self.simulation_board.adv_pos
            adv_pos_r = self.simulation_board.my_pos
            dir_map = {
                "u": 0,
                "r": 1,
                "d": 2,
                "l": 3,
            }

            def valid_move(x, y, x_max, y_max):
                return 0 <= x < x_max and 0 <= y < y_max

            def get_move_area(chess_board, my_pos, adv_pos, max_step):
                """
                This method is to find all the available moves in current position by BFS
                return: list[(x,y)]
                """
                max_x, max_y = len(chess_board), len(chess_board[0])
                result = []
                moves = [my_pos]
                directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

                for i in range(max_step):
                    next_move = []
                    for pos in moves:
                        r, c = pos

                        # check four direction
                        for key in dir_map:
                            direction = dir_map[key]

                            # block by wall
                            if chess_board[r][c][direction]:
                                continue

                            block = 0
                            new_x, new_y = (r + directions[direction][0], c + directions[direction][1])
                            for wall in chess_board[new_x][new_y]:
                                if wall:
                                    block += 1

                            # more than 2 walls in new position
                            if block > 2:
                                continue

                            if valid_move(new_x, new_y, max_x, max_y) and \
                                    (new_x, new_y) not in result and (new_x, new_y) != adv_pos:
                                next_move.append((new_x, new_y))
                                result.append((new_x, new_y))

                    moves = next_move[:]
                return result

            valid_moves = get_move_area(chess_board_r, my_pos_r, adv_pos_r, max_step_r)
            if len(valid_moves) == 0:
                next_pos = my_pos_r
            else:
                i = random.randint(0, len(valid_moves) - 1)
                next_pos = valid_moves[i]
            # print(valid_moves, "\n", next_pos)
            next_dir = random.randint(0, 3)
            while chess_board_r[next_pos[0]][next_pos[1]][next_dir]:
                next_dir = random.randint(0, 3)
            return next_pos, next_dir







