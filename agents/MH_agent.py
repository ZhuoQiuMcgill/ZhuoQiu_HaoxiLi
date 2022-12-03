
import numpy as np
import constants
from time   import time
from store  import register_agent
from copy   import deepcopy
from agents import *


@register_agent("MH_agent")
class MH_agent(Agent):

    # Max step
    simulation_number = 1

    # Testing only
    selection_time  = 0
    expansion_time  = 0
    simulation_time = 0
    heuristic_time  = 0

    selection_count  = 0
    expansion_count  = 0
    simulation_count = 0
    heuristic_count  = 0

    class Tree:
        def __init__(
                self,
                chess_board : np.ndarray,
                cur_pos     : np.ndarray,
                adv_pos     : np.ndarray,
                max_step    : int,
                ):
            self.size     = 0
            self.max_step = max_step
            self.root     = self.Node(chess_board, cur_pos, adv_pos, -1, None, self)

        class Node:
            def __init__(
                    self,
                    chess_board : np.ndarray,
                    cur_pos     : np.ndarray,
                    adv_pos     : np.ndarray,
                    direction   : int,
                    parent,
                    tree
                    ):
                self.chess_board  = chess_board
                self.cur_pos      = cur_pos
                self.adv_pos      = adv_pos
                self.direction    = direction
                self.parent       = parent
                self.children     = []
                self.level        = 0
                self.tree         = tree
                self.s            = 0
                self.w            = 0
                self.c_square     = 2.2

            def uct_estimation(self) -> float:
                if self.s == 0:
                    return np.Infinity
                return self.w / self.s + np.sqrt(self.c_square * np.log(self.parent.s) / self.s)

            def find_best_child(self):
                max_uc = self.children[0].uct_estimation()
                max_child = self.children[0]
                for child in self.children:
                    child_uc = child.uct_estimation()
                    if child_uc > max_uc:
                        max_uc = child_uc
                        max_child = child
                return max_child

            def add_children(self):
                chess_board  = self.chess_board
                current_turn = self.level % 2 == 0

                if current_turn:
                    cur_pos = self.cur_pos
                    adv_pos = self.adv_pos
                else:
                    cur_pos = self.adv_pos
                    adv_pos = self.cur_pos

                for x in range(0, self.tree.max_step + 1):
                    for y in range(0, self.tree.max_step - x):
                        for direction in range(0, 4):
                            end_pos_0 = np.asarray((cur_pos[0] - x, cur_pos[1] - y))
                            end_pos_1 = np.asarray((cur_pos[0] + x, cur_pos[1] + y))
                            end_pos_2 = np.asarray((cur_pos[0] - x, cur_pos[1] + y))
                            end_pos_3 = np.asarray((cur_pos[0] + x, cur_pos[1] - y))

                            def append_children(pos, face, turn):
                                if MH_agent.check_valid_step(chess_board, cur_pos, pos, adv_pos, self.tree.max_step, face):
                                    new_board = MH_agent.set_barrier(deepcopy(chess_board), pos[0], pos[1], face)
                                    if turn:
                                        child = MH_agent.Tree.Node(new_board, pos,     adv_pos, face, self, self.tree)
                                    else:
                                        child = MH_agent.Tree.Node(new_board, adv_pos, pos,     face, self, self.tree)
                                    self.children.append(child)
                                    child.level = self.level + 1
                                    child.tree.size += 1

                            if x == 0 and y == 0:
                                append_children(end_pos_0, direction, current_turn)
                            elif x * y == 0:
                                append_children(end_pos_0, direction, current_turn)
                                append_children(end_pos_1, direction, current_turn)
                            else:
                                append_children(end_pos_0, direction, current_turn)
                                append_children(end_pos_1, direction, current_turn)
                                append_children(end_pos_2, direction, current_turn)
                                append_children(end_pos_3, direction, current_turn)

    def __init__(self):
        super(MH_agent, self).__init__()
        self.name     = "MH_agent"
        self.autoplay = True
        self.dir_map  = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
            }

    def step(self, chess_board, my_pos, adv_pos, max_step):
        self.selection_time  = 0
        self.expansion_time  = 0
        self.simulation_time = 0
        self.heuristic_time  = 0

        self.selection_count  = 0
        self.expansion_count  = 0
        self.simulation_count = 0
        self.heuristic_count  = 0

        start_time = time()
        best_action_node = self.monte_carlo(chess_board, np.asarray(my_pos), np.asarray(adv_pos), max_step)

        if constants.DEBUG_MODE:
            tot_time = time() - start_time
            print(f"total_time per step = {tot_time}")
            print(f"selection  time ({self.selection_count}) = {self.selection_time} ({self.selection_time / tot_time * 100} %)")
            print(f"expansion  time ({self.expansion_count}) = {self.expansion_time} ({self.expansion_time / tot_time * 100} %)")
            print(f"simulation time ({self.simulation_count}) = {self.simulation_time} ({self.simulation_time / tot_time * 100} %)")
            print(f"heuristic  time ({self.heuristic_count}) = {self.heuristic_time} ({self.heuristic_time / tot_time * 100} %)")
        """
        for node in best_action_node.tree.root.children:
            print(f" {node.uct_estimation()} ", end="")
        """
        return (best_action_node.cur_pos[0], best_action_node.cur_pos[1]), best_action_node.direction

    def monte_carlo(self,
                    chess_board : np.ndarray,
                    cur_pos     : np.ndarray,
                    adv_pos     : np.ndarray,
                    max_step    : int,
                    ) -> Tree.Node:
        start_time  = time()
        time_budget = 1.75
        tree = self.monte_carlo_init(chess_board, cur_pos, adv_pos, max_step)
        root = tree.root

        while time() - start_time < time_budget:
            leaf  = self.selection(root)
            self.expansion(leaf)
            if leaf.children:
                child = self.pick_best_node(leaf.cur_pos, leaf.children, np.shape(chess_board)[0])
                #child = np.random.choice(leaf.children)
            else:
                child = leaf
            self.simulation(child, self.simulation_number, max_step)
        return root.find_best_child()

    def monte_carlo_init(self,
                         chess_board : np.ndarray,
                         cur_pos     : np.ndarray,
                         adv_pos     : np.ndarray,
                         max_step    : int,
                         ) -> Tree:
        # Create tree
        tree = self.Tree(chess_board, cur_pos, adv_pos, max_step)

        # Generate first level
        tree.root.add_children()
        for child in tree.root.children:
            self.simulation(child, self.simulation_number, max_step)
        return tree

    def selection(self, node : Tree.Node):
        self.selection_count += 1
        start_time = time()
        if not node.children or node.s == 0:  # leaf node
            return node

        node_selected = self.selection(node.find_best_child())
        if constants.DEBUG_MODE:
            self.selection_time += (time() - start_time)
        return node_selected

    def pick_best_node(
            self,
            cur_pos     : np.ndarray,
            children    : list[Tree.Node],
            board_size  : int
            ):
        self.heuristic_count += 1
        start_time = time()
        if board_size % 2 == 1:
            x_mid = board_size // 2 + 1
            y_mid = board_size // 2 + 1
        else:
            x_mid = board_size / 2
            y_mid = board_size / 2

        mid_pos = np.asarray((x_mid, y_mid))
        best_child = children[0]
        best_value = 0

        for child in children:
            # Get x2, y2 from node.
            child_pos = child.cur_pos
            value = np.linalg.norm(cur_pos - child_pos) - np.linalg.norm(child_pos - mid_pos)
            if value > best_value:
                best_value = value
                best_child = child

        if constants.DEBUG_MODE:
            self.heuristic_time += (time() - start_time)
        return best_child

    def expansion(self, node : Tree.Node):
        self.expansion_count += 1
        start_time = time()
        if node.s == 0:
            self.expansion_count -= 1
            return
        node.add_children()
        if constants.DEBUG_MODE:
            self.expansion_time += (time() - start_time)

    def simulation(self,
                   node     : Tree.Node,
                   n        : int,
                   max_step : int
                   ):
        self.simulation_count += 1
        start_time = time()
        current_turn = node.level % 2 == 0
        total_win    = 0
        for i in range(n):
            cur_board = deepcopy(node.chess_board)
            pos_0     = deepcopy(node.cur_pos)
            pos_1     = deepcopy(node.adv_pos)
            is_end = self.check_endgame(cur_board, pos_0, pos_1)[0]
            is_win = self.check_endgame(cur_board, pos_0, pos_1)[1]
            while not is_end:
                if current_turn:
                    pos, direction = self.random_step(cur_board, tuple(pos_0), tuple(pos_1), max_step)
                    pos0 = pos
                else:
                    pos, direction = self.random_step(cur_board, tuple(pos_1), tuple(pos_0), max_step)
                    pos1 = pos
                current_turn = not current_turn
                self.set_barrier(cur_board, pos[0], pos[1], direction)
                is_end, is_win = self.check_endgame(cur_board, pos_0, pos_1)
            if is_win:
                total_win += 1
        self.backpropagation(node, total_win, n)
        if constants.DEBUG_MODE:
            self.simulation_time += (time() - start_time)

    def backpropagation(self,
                        node : Tree.Node,
                        w    : int,
                        s    : int,
                        ):
        node.w += w
        node.s += s
        if node.parent is None:
            return
        self.backpropagation(node.parent, w, s)

    @staticmethod
    def random_step(
            chess_board : np.ndarray,
            cur_pos     : tuple,
            adv_pos     : tuple,
            max_step    : int,
            ):
        steps = np.random.randint(0, max_step + 1)
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        my_pos = deepcopy(cur_pos)
        for _ in range(steps):
            choices = []
            choices.extend(range(0, 4))
            last_choice = -1
            for _ in range(4):
                if last_choice in choices:
                    choices.remove(last_choice)
                my_choice = np.random.choice(choices)
                last_choice = my_choice
                x_move, y_move = moves[my_choice]
                x, y = my_pos
                my_pos = (x + x_move, y + y_move)

                if (chess_board[x, y, my_choice]) or (my_pos == adv_pos):
                    my_pos = x, y
                    continue
                else:
                    break
            if my_pos == cur_pos:
                break

        directions = [0, 1, 2, 3]
        last_dir = -1
        for _ in range(4):
            if last_dir in directions:
                directions.remove(last_dir)
            x, y = my_pos
            direction = np.random.choice(directions)
            last_dir = direction
            if not chess_board[x, y, direction]:
                break
        return my_pos, last_dir

    @staticmethod
    def check_endgame(
            chess_board : np.ndarray,
            cur_pos     : np.ndarray,
            adv_pos     : np.ndarray,
            ) -> (bool, bool):
        """
        Check if the game ends and who wins.

        :returns:
            - is_endgame - Whether the game ends. <br>
            - is_winning - Whether the current agent wins.
        """

        board_size = np.shape(chess_board)[0]
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        # Union-Find
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
                for direction, move in enumerate(moves[1:3]):
                    # Only check down and right
                    if chess_board[r, c, direction + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(board_size):
            for c in range(board_size):
                find((r, c))
        p0_r = find(tuple(cur_pos))
        p1_r = find(tuple(adv_pos))
        p0_final_score = list(father.values()).count(p0_r)
        p1_final_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            return False, False
        return True, p0_final_score >= p1_final_score

    @staticmethod
    def set_barrier(chess_board, x, y, direction):
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        opposites = {0: 2, 1: 3, 2: 0, 3: 1}
        # Set the barrier to True
        chess_board[x, y, direction] = True
        # Set the opposite barrier to True
        move = moves[direction]
        chess_board[x + move[0], y + move[1], opposites[direction]] = True
        return chess_board

    @staticmethod
    def check_valid_step(
            chess_board : np.ndarray,
            start_pos   : np.ndarray,
            end_pos     : np.ndarray,
            adv_pos     : np.ndarray,
            max_step    : int,
            barrier_dir : int
            ):

        board_size = np.shape(chess_board)[0]
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

        def check_boundary(pos):
            x, y = pos
            return 0 <= x < board_size and 0 <= y < board_size

        if not check_boundary(end_pos):
            return False
        # Endpoint already has barrier or is boarder
        r, c = end_pos
        if chess_board[r, c, barrier_dir]:
            return False
        if np.array_equal(start_pos, end_pos):
            return True

        # BFS
        state_queue = [(start_pos, 0)]
        visited = {tuple(start_pos)}
        is_reached = False
        while state_queue and not is_reached:
            cur_pos, cur_step = state_queue.pop(0)
            r, c = cur_pos
            if cur_step == max_step:
                break
            for direction, move in enumerate(moves):
                if chess_board[r, c, direction]:
                    continue
                next_pos = cur_pos + move
                if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                    continue
                if np.array_equal(next_pos, end_pos):
                    is_reached = True
                    break
                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))
        return is_reached
