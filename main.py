import time
import math
import random
import heapq
import sys
import pygame

# ══════════════════════════════════════════════════════════════════════
#  SECTION 1 — GRID MODEL
# ══════════════════════════════════════════════════════════════════════

EMPTY = 0
WALL  = 1
START = 2
GOAL  = 3


class Cell:
    def __init__(self, row: int, col: int):
        self.row = row
        self.col = col
        self.cell_type = EMPTY
        self.g = float('inf')
        self.h = 0.0
        self.f = float('inf')
        self.parent = None
        self.in_open   = False
        self.in_closed = False
        self.on_path   = False

    @property
    def pos(self):
        return (self.row, self.col)

    def is_walkable(self):
        return self.cell_type != WALL

    def reset_search(self):
        self.g = float('inf')
        self.h = 0.0
        self.f = float('inf')
        self.parent    = None
        self.in_open   = False
        self.in_closed = False
        self.on_path   = False

    def __lt__(self, other):
        return self.f < other.f


class Grid:
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.cells: list[list[Cell]] = [
            [Cell(r, c) for c in range(cols)] for r in range(rows)
        ]
        self.start_cell: Cell | None = None
        self.goal_cell:  Cell | None = None
        self._set_default_start_goal()

    def get(self, row, col) -> Cell | None:
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.cells[row][col]
        return None

    def neighbors(self, cell: Cell) -> list[Cell]:
        result = []
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nb = self.get(cell.row + dr, cell.col + dc)
            if nb and nb.is_walkable():
                result.append(nb)
        return result

    def _set_default_start_goal(self):
        self._place_start(self.cells[0][0])
        self._place_goal(self.cells[self.rows - 1][self.cols - 1])

    def _place_start(self, cell: Cell):
        if self.start_cell:
            self.start_cell.cell_type = EMPTY
        self.start_cell = cell
        cell.cell_type  = START

    def _place_goal(self, cell: Cell):
        if self.goal_cell:
            self.goal_cell.cell_type = EMPTY
        self.goal_cell = cell
        cell.cell_type = GOAL

    def toggle_wall(self, row, col):
        cell = self.get(row, col)
        if cell and cell.cell_type not in (START, GOAL):
            cell.cell_type = EMPTY if cell.cell_type == WALL else WALL

    def place_wall(self, row, col):
        cell = self.get(row, col)
        if cell and cell.cell_type not in (START, GOAL):
            cell.cell_type = WALL

    def remove_wall(self, row, col):
        cell = self.get(row, col)
        if cell and cell.cell_type == WALL:
            cell.cell_type = EMPTY

    def generate_random_maze(self, density=0.30):
        for row in self.cells:
            for cell in row:
                if cell.cell_type in (START, GOAL):
                    continue
                cell.cell_type = WALL if random.random() < density else EMPTY

    def reset_search_state(self):
        for row in self.cells:
            for cell in row:
                cell.reset_search()

    def full_reset(self):
        for row in self.cells:
            for cell in row:
                if cell.cell_type not in (START, GOAL):
                    cell.cell_type = EMPTY
                cell.reset_search()

    def spawn_random_obstacle(self) -> Cell | None:
        candidates = [
            cell for row in self.cells for cell in row
            if cell.cell_type == EMPTY and not cell.on_path
        ]
        if candidates:
            cell = random.choice(candidates)
            cell.cell_type = WALL
            return cell
        return None
    
    