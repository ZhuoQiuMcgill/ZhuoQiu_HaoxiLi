import copy
import math


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
            if self.chess_board[board.adv_pos[0], board.adv_pos[1], i] != board.chess_board[
                board.adv_pos[0], board.adv_pos[1], i]:
                wall = i
        xdir = board.adv_pos[0] - self.adv_pos[0]
        ydir = board.adv_pos[1] - self.adv_pos[1]
        return xdir, ydir, wall

    def check_endgame(self):
        board_size = 5
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
                # right
                if (r, c) == (3, 3):
                    print(self.chess_board[r][c])
                if not self.chess_board[r][c][1]:
                    pos_a = find((r, c))
                    pos_b = find((r, c + 1))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

                # down
                if not self.chess_board[r][c][2]:
                    pos_a = find((r, c))
                    pos_b = find((r + 1, c))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(board_size):
            for c in range(board_size):
                find((r, c))
        p0_r = find(tuple(self.my_pos))
        p1_r = find(tuple(self.adv_pos))
        #print(self.my_pos, self.adv_pos)
        print(father)
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


chess_board = [[[True, False, False, True],
                [True, False, False, False],
                [True, False, True, False],
                [True, False, False, False],
                [True, True, False, False]],

               [[True, True, False, True],
                [True, False, False, False],
                [True, False, True, False],
                [False, False, True, False],
                [False, True, False, False]],

               [[False, False, False, True],
                [True, False, True, False],
                [True, True, True, False],
                [True, False, False, False],
                [True, True, False, False]],

               [[False, True, False, True],
                [True, False, False, True],
                [True, False, True, True],
                [True, True, True, True],
                [True, True, True, False]],

               [[True, False, True, True],
                [False, False, True, True],
                [True, False, True, True],
                [True, True, True, True],
                [False, True, True, True]]]

board = Board(chess_board, (3, 3), (4, 3), 3)
print(board.check_endgame())
