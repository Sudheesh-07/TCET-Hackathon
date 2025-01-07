import numpy as np
from enum import Enum
from typing import List, Tuple
from queue import PriorityQueue


class Direction(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

    def get_color(self):
        return {
            Direction.NORTH: "#FF6B6B",
            Direction.SOUTH: "#4ECDC4",
            Direction.EAST: "#95E1D3",
            Direction.WEST: "#A8E6CF"
        }[self]


class MineMovement(Enum):
    DIAGONAL = 0
    HORIZONTAL = 1
    VERTICAL = 2


class Kingdom:
    def __init__(self, size: int = 20):
        self.size = size
        self.castle_pos = (size // 2, size // 2)
        self.mines = set()
        self.casualties = {d: 0 for d in Direction}
        self.grid = np.zeros((size, size), dtype=int)

    def initialize_mines(self, num_mines: int = 50):
        self.mines.clear()
        while len(self.mines) < num_mines:
            x, y = np.random.randint(0, self.size), np.random.randint(0, self.size)
            if (x, y) != self.castle_pos:
                self.mines.add((x, y))
        self._update_grid()

    def _update_grid(self):
        self.grid.fill(0)
        for x, y in [(x, y) for x in range(self.size) for y in range(self.size)]:
            if (x, y) not in self.mines:
                self.grid[x, y] = sum(1 for dx, dy in [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]]
                                      if (x + dx, y + dy) in self.mines and (dx, dy) != (0, 0))

    def move_mines(self, movement: MineMovement):
        moves = {
            MineMovement.DIAGONAL: [(1, 1), (1, -1), (-1, 1), (-1, -1)],
            MineMovement.HORIZONTAL: [(0, 1), (0, -1)],
            MineMovement.VERTICAL: [(1, 0), (-1, 0)]
        }[movement]

        new_mines = set()
        for x, y in self.mines:
            valid_moves = [(x + dx, y + dy) for dx, dy in moves
                           if 0 <= x + dx < self.size and 0 <= y + dy < self.size
                           and (x + dx, y + dy) != self.castle_pos]
            new_mines.add(valid_moves[np.random.randint(len(valid_moves))] if valid_moves else (x, y))

        self.mines = new_mines
        self._update_grid()

    def get_kingdom_position(self, direction: Direction) -> Tuple[int, int]:
        return {
            Direction.NORTH: (0, self.size // 2),
            Direction.SOUTH: (self.size - 1, self.size // 2),
            Direction.EAST: (self.size // 2, self.size - 1),
            Direction.WEST: (self.size // 2, 0)
        }[direction]

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int],
                  blocked_positions: set = None) -> List[Tuple[int, int]]:
        blocked_positions = blocked_positions or set()
        pq = PriorityQueue()
        pq.put((0, start, [start]))
        visited = {start}

        while not pq.empty():
            cost, current, path = pq.get()
            if current == end:
                return path

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                next_pos = (current[0] + dx, current[1] + dy)
                if (0 <= next_pos[0] < self.size and
                        0 <= next_pos[1] < self.size and
                        next_pos not in visited and
                        next_pos not in blocked_positions):
                    visited.add(next_pos)
                    new_cost = cost + self.grid[next_pos]
                    pq.put((new_cost, next_pos, path + [next_pos]))
        return []
