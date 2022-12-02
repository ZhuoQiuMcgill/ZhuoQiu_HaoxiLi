# Student agent: Add your own agent here
import copy
import math
import random
import numpy as np

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
        self.chess_board = chess_board
        self.my_pos = my_pos
        self.adv_pos = adv_pos
        self.max_step = max_step

    def deep_copy(self):
        dup = copy.deepcopy(self)
        return dup

    def check_endgame(self):
        board_size = int(math.sqrt(self.chess_board.size / 4))
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        father = dict()
        for r in range(board_size):
            for c in range(board_size):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(board_size):
            for c in range(board_size):
                for dir, move in enumerate(
                        moves[1:3]
                ):  # Only check down and right
                    if self.chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(board_size):
            for c in range(board_size):
                find((r, c))
        p0_r = find(tuple(self.my_pos))
        p1_r = find(tuple(self.adv_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            return False, p0_score, p1_score
        # player_win = None
        # win_blocks = -1
        # if p0_score > p1_score:
        #     player_win = 0
        #     win_blocks = p0_score
        # elif p0_score < p1_score:
        #     player_win = 1
        #     win_blocks = p1_score
        # else:
        #     player_win = -1  # Tie
        # if player_win >= 0:
        #     logging.info(
        #         f"Game ends! Player {self.player_names[player_win]} wins having control over {win_blocks} blocks!"
        #     )
        # else:
        #     logging.info("Game ends! It is a Tie!")
        return True, p0_score, p1_score

    def move_to(self, move, player):
        xdir = move[0][0]
        ydir = move[0][1]
        wall = move[1]
        if player == 0:
            self.my_pos[0] = self.my_pos[0] + xdir
            self.my_pos[1] = self.my_pos[1] + ydir
            self.chess_board[xdir, ydir, wall] = True
        else:
            self.adv_pos[0] = self.adv_pos[0] + xdir
            self.adv_pos[1] = self.adv_pos[1] + ydir
            self.chess_board[xdir, ydir, wall] = True







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







