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
        self.current_turn = 0
        self.board = None
        self.MCS_Tree = None
        self.last_step = None
        self.init_expansions = 100

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
        if self.current_turn == 0:
            self.board = Board(chess_board, my_pos, adv_pos, max_step)
            self.MCS_Tree = MCSTree(self.board)
            for _ in range(self.init_expansions):
                self.MCS_Tree.expend()
            next_step = self.MCS_Tree.best_step()
            new_root = self.MCS_Tree.root.children[next_step]
            self.last_step = next_step
            self.MCS_Tree.update_root(new_root)
            return next_step

        cur_board = Board(chess_board, my_pos, adv_pos, max_step)
        adv_step = self.board.get_step(cur_board, self.last_step)
        next_root = self.MCS_Tree.root.children[adv_step]
        self.MCS_Tree.update_root(next_root)

        next_step = self.MCS_Tree.best_step()
        next_root = self.MCS_Tree.root.children[next_step]
        self.MCS_Tree.update_root(next_root)

        self.last_step = next_step
        return next_step


class Board:
    def __init__(self, chess_board, my_pos, adv_pos, max_step):
        self.chess_board = chess_board
        self.my_pos = my_pos
        self.adv_pos = adv_pos
        self.max_step = max_step

    def deep_copy(self):
        dup = copy.deepcopy(self)
        return dup

    def get_step(self, board, last_move):
        self.move_to(self, last_move, 0)
        for i in range(4):
            if self.chess_board[board.adv_pos[0], board.adv_pos[1], i] != board.chess_board[board.adv_pos[0], board.adv_pos[1], i]:
                wall = i
        xdir = board.adv_pos[0] - self.adv_pos[0]
        ydir = board.adv_pos[1] - self.adv_pos[1]
        return xdir, ydir, wall

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

        if move is None:
            return

        xdir = move[0][0]
        ydir = move[0][1]
        wall = move[1]
        if player == 0:
            self.my_pos[0] = self.my_pos[0] + xdir
            self.my_pos[1] = self.my_pos[1] + ydir
            self.chess_board[self.my_pos[0], self.my_pos[1], wall] = True
        else:
            self.adv_pos[0] = self.adv_pos[0] + xdir
            self.adv_pos[1] = self.adv_pos[1] + ydir
            self.chess_board[self.adv_pos[0], self.adv_pos[1], wall] = True


class Node:
    def __init__(self, father, player, board, max_sim, move):
        self.father = father
        self.player = player
        self.wins = 0
        self.simulations = max_sim
        self.max_sim = max_sim
        self.chess_board = board
        self.move = move

        self.simulation_board = self.chess_board.deep_copy()
        self.children = dict()

    def reset_board(self):
        self.simulation_board = self.chess_board.deep_copy()

    def run_simulation(self):
        for _ in range(self.max_sim):
            self.reset_board()
            turn = self.player + 1
            is_end, p0_score, p1_score = self.simulation_board.check_endgame()
            while not is_end:
                player = turn % 2
                self.simulation_board.move_to(self.random_player_step(player), player)
                turn += 1
                is_end, p0_score, p1_score = self.simulation_board.check_endgame()

            if p0_score > p1_score and self.player == 0:
                self.wins += 1
            elif p0_score < p1_score and self.player == 1:
                self.wins += 1
        self.update_simulations()

    def update_simulations(self):
        father = self.father
        while father is not None:
            father.wins += self.wins
            father.simulations += self.simulations
            father = father.father

    def q_star(self):
        if self.father is None:
            return 0
        return self.wins / self.simulations * math.sqrt(2 * math.log10(self.father.simulations))

    def expend(self):
        chess_board = self.chess_board.chess_board
        my_pos = self.chess_board.my_pos
        adv_pos = self.chess_board.adv_pos
        max_step = self.chess_board.max_step

        # find all moves
        if self.player == 0:
            move_area = self.get_move_area(chess_board, my_pos, adv_pos, max_step, True)
        else:
            move_area = self.get_move_area(chess_board, adv_pos, my_pos, max_step, False)
        all_moves = []
        for move in move_area:
            i = 0
            for wall in chess_board[move[0]][move[1]]:
                if not wall:
                    all_moves.append((move, i))
                i += 1

        # create new node
        for move in all_moves:
            new_chess_board = self.chess_board.deep_copy()
            new_chess_board.move_to(move, self.player)
            new_child = Node(self, (self.player + 1) % 2, new_chess_board, self.max_sim, move)
            self.children[move] = new_child
            new_child.run_simulation()

    def random_player_step(self, player_num):
        """
        this method simulate a random player's move
        """
        chess_board_r = self.simulation_board.chess_board
        max_step_r = self.simulation_board.max_step
        if player_num == 0:
            my_pos_r = self.simulation_board.my_pos
            adv_pos_r = self.simulation_board.adv_pos
        else:
            my_pos_r = self.simulation_board.adv_pos
            adv_pos_r = self.simulation_board.my_pos

        valid_moves = self.get_move_area(chess_board_r, my_pos_r, adv_pos_r, max_step_r, True)
        if len(valid_moves) == 0:
            next_pos = my_pos_r
        else:
            i = random.randint(0, len(valid_moves) - 1)
            next_pos = valid_moves[i]

        next_dir = random.randint(0, 3)
        while chess_board_r[next_pos[0]][next_pos[1]][next_dir]:
            next_dir = random.randint(0, 3)
        return next_pos, next_dir

    def get_move_area(self, chess_board, my_pos, adv_pos, max_step, improved):
        """
        improved (bool): if improved, it will not return the position with more than 2 barriers
        This method is to find all the available moves in current position by BFS
        return: list[(x,y)]
        """
        max_x, max_y = len(chess_board), len(chess_board[0])
        result = []
        moves = [my_pos]
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

        def valid_move(x, y, x_max, y_max):
            return 0 <= x < x_max and 0 <= y < y_max

        for _ in range(max_step):
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
                    if block > 2 and improved:
                        continue

                    if valid_move(new_x, new_y, max_x, max_y) and \
                            (new_x, new_y) not in result and (new_x, new_y) != adv_pos:
                        next_move.append((new_x, new_y))
                        result.append((new_x, new_y))

            moves = next_move[:]
        return result


class MCSTree:

    def __init__(self, board):
        self.max_sim = 100
        self.root = Node(None, 0, board, self.max_sim, None)

    def expend(self):
        node_ptr = self.root
        while len(node_ptr.children) != 0:
            max_q = 0
            next_child = None
            for child in node_ptr.children:
                q = child.q_star()
                if q > max_q:
                    max_q = q
                    next_child = child
            node_ptr = next_child
        node_ptr.expend()

    def update_root(self, node):
        self.root = node
        self.root.father = None

    def best_step(self):
        max_q = 0
        best_child = None
        for child in self.root.children:
            q = child.q_star()
            if q > max_q:
                max_q = q
                best_child = child
        return best_child.step
