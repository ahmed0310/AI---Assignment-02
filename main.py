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
    

# ══════════════════════════════════════════════════════════════════════
#  SECTION 2 — HEURISTICS & SEARCH ALGORITHMS
# ══════════════════════════════════════════════════════════════════════

def heuristic_manhattan(a: Cell, b: Cell) -> float:
    return abs(a.row - b.row) + abs(a.col - b.col)

def heuristic_euclidean(a: Cell, b: Cell) -> float:
    return math.sqrt((a.row - b.row)**2 + (a.col - b.col)**2)

HEURISTICS = {
    "Manhattan": heuristic_manhattan,
    "Euclidean": heuristic_euclidean,
}


class SearchResult:
    def __init__(self):
        self.path: list[Cell] = []
        self.nodes_visited = 0
        self.path_cost = 0.0
        self.found = False


def _reconstruct_path(goal: Cell) -> list[Cell]:
    path, node = [], goal
    while node:
        path.append(node)
        node = node.parent
    path.reverse()
    for cell in path:
        cell.on_path = True
    return path


def greedy_bfs(grid: Grid, start: Cell, goal: Cell, heuristic_name="Manhattan") -> SearchResult:
    """Greedy Best-First Search — expands node with smallest h(n). NOT optimal."""
    h = HEURISTICS[heuristic_name]
    result = SearchResult()

    start.h = h(start, goal)
    start.f = start.h
    start.g = 0

    counter = 0
    heap = []
    heapq.heappush(heap, (start.h, counter, start))
    start.in_open = True
    visited: set = set()

    while heap:
        _, _, current = heapq.heappop(heap)
        if current.pos in visited:
            continue
        visited.add(current.pos)
        current.in_open   = False
        current.in_closed = True
        result.nodes_visited += 1

        if current is goal:
            result.found     = True
            result.path      = _reconstruct_path(goal)
            result.path_cost = goal.g
            return result

        for nb in grid.neighbors(current):
            if nb.pos in visited:
                continue
            new_g = current.g + 1
            nb.h  = h(nb, goal)
            nb.f  = nb.h
            if not nb.in_open or new_g < nb.g:
                nb.g = new_g
                nb.parent = current
                nb.in_open = True
                counter += 1
                heapq.heappush(heap, (nb.h, counter, nb))

    return result



def astar(grid: Grid, start: Cell, goal: Cell, heuristic_name="Manhattan") -> SearchResult:
    """A* Search — f(n)=g(n)+h(n). Guarantees optimal path with admissible heuristic."""
    h = HEURISTICS[heuristic_name]
    result = SearchResult()

    start.g = 0
    start.h = h(start, goal)
    start.f = start.g + start.h

    counter = 0
    heap = []
    heapq.heappush(heap, (start.f, counter, start))
    start.in_open = True
    expanded: dict = {}

    while heap:
        _, _, current = heapq.heappop(heap)
        if current.pos in expanded and expanded[current.pos] <= current.g:
            continue
        expanded[current.pos] = current.g
        current.in_open   = False
        current.in_closed = True
        result.nodes_visited += 1

        if current is goal:
            result.found     = True
            result.path      = _reconstruct_path(goal)
            result.path_cost = goal.g
            return result

        for nb in grid.neighbors(current):
            new_g = current.g + 1
            if new_g < nb.g:
                nb.g      = new_g
                nb.h      = h(nb, goal)
                nb.f      = nb.g + nb.h
                nb.parent = current
                nb.in_open   = True
                nb.in_closed = False
                counter += 1
                heapq.heappush(heap, (nb.f, counter, nb))

    return result


def run_search(grid: Grid, algo: str, heuristic: str) -> SearchResult:
    if algo == "A* Search":
        return astar(grid, grid.start_cell, grid.goal_cell, heuristic)
    return greedy_bfs(grid, grid.start_cell, grid.goal_cell, heuristic)


def replan(grid: Grid, current: Cell, goal: Cell, algo: str, heuristic: str) -> SearchResult:
    grid.reset_search_state()
    if algo == "A* Search":
        return astar(grid, current, goal, heuristic)
    return greedy_bfs(grid, current, goal, heuristic)


# ══════════════════════════════════════════════════════════════════════
#  SECTION 3 — AGENT
# ══════════════════════════════════════════════════════════════════════

class Agent:
    def __init__(self):
        self.cell: Cell | None = None
        self.path: list[Cell]  = []
        self.path_index  = 0
        self.moving      = False
        self.reached_goal= False
        self.replanning  = False

    def set_path(self, path: list[Cell]):
        self.path        = path
        self.path_index  = 0
        self.moving      = bool(path)
        self.reached_goal= False
        if path:
            self.cell = path[0]

    def step(self) -> bool:
        if not self.moving or not self.path:
            return False
        nxt = self.path_index + 1
        if nxt >= len(self.path):
            self.moving = False
            self.reached_goal = True
            return False
        next_cell = self.path[nxt]
        if next_cell.cell_type == WALL:
            self.moving     = False
            self.replanning = True
            return False
        self.path_index = nxt
        self.cell       = next_cell
        return True

    def needs_replan(self):
        return self.replanning

    def accept_replan(self, new_path: list[Cell]):
        self.path       = new_path
        self.path_index = 0
        self.replanning = False
        self.moving     = bool(new_path)
        if new_path:
            self.cell = new_path[0]

    def is_path_blocked(self) -> bool:
        for cell in self.path[self.path_index + 1:]:
            if cell.cell_type == WALL:
                return True
        return False

    def reset(self):
        self.path        = []
        self.path_index  = 0
        self.moving      = False
        self.reached_goal= False
        self.replanning  = False
        self.cell        = None


# ══════════════════════════════════════════════════════════════════════
#  SECTION 4 — PYGAME GUI
# ══════════════════════════════════════════════════════════════════════

# ── Palette ─────────────────────────────────────────────────────────
BG_COLOR       = (15,  17,  26)
CELL_EMPTY     = (28,  32,  45)
CELL_WALL      = (12,  14,  20)
CELL_START     = (50, 205,  50)
CELL_GOAL      = (220,  60,  60)
CELL_OPEN      = (255, 200,  40)
CELL_CLOSED    = (60,  120, 200)
CELL_PATH      = (50,  230, 130)
CELL_AGENT     = (255, 100, 200)
PANEL_BG       = (20,  24,  36)
PANEL_BORDER   = (50,  60,  85)
TEXT_PRIMARY   = (220, 230, 255)
TEXT_SECONDARY = (120, 140, 180)
TEXT_ACCENT    = (80,  200, 160)
BTN_NORMAL     = (40,  50,  75)
BTN_HOVER      = (60,  75, 110)
BTN_ACTIVE     = (70,  160, 120)
BTN_BORDER     = (70,  85, 120)

PANEL_W = 260


class Button:
    def __init__(self, rect, label, toggle=False):
        self.rect    = pygame.Rect(rect)
        self.label   = label
        self.toggle  = toggle
        self.active  = False
        self.hovered = False

    def draw(self, surface, font):
        color = BTN_ACTIVE if (self.toggle and self.active) else \
                BTN_HOVER  if self.hovered else BTN_NORMAL
        pygame.draw.rect(surface, color,      self.rect, border_radius=6)
        pygame.draw.rect(surface, BTN_BORDER, self.rect, 1, border_radius=6)
        txt = font.render(self.label, True, TEXT_PRIMARY)
        surface.blit(txt, txt.get_rect(center=self.rect.center))

    def handle_event(self, event) -> bool:
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                if self.toggle:
                    self.active = not self.active
                return True
        return False


class Dropdown:
    def __init__(self, rect, options, label=""):
        self.rect    = pygame.Rect(rect)
        self.options = options
        self.label   = label
        self.index   = 0
        self.open    = False

    @property
    def value(self):
        return self.options[self.index]

    def draw(self, surface, font, small_font):
        if self.label:
            lbl = small_font.render(self.label, True, TEXT_SECONDARY)
            surface.blit(lbl, (self.rect.x, self.rect.y - 18))
        pygame.draw.rect(surface, BTN_NORMAL, self.rect, border_radius=6)
        pygame.draw.rect(surface, BTN_BORDER, self.rect, 1, border_radius=6)
        txt = font.render(self.value, True, TEXT_PRIMARY)
        surface.blit(txt, txt.get_rect(midleft=(self.rect.x + 8, self.rect.centery)))
        arrow = small_font.render("▼", True, TEXT_SECONDARY)
        surface.blit(arrow, arrow.get_rect(midright=(self.rect.right - 8, self.rect.centery)))
        if self.open:
            for i, opt in enumerate(self.options):
                r = pygame.Rect(self.rect.x,
                                self.rect.bottom + i * self.rect.height,
                                self.rect.width, self.rect.height)
                col = BTN_ACTIVE if i == self.index else BTN_HOVER
                pygame.draw.rect(surface, col,       r, border_radius=4)
                pygame.draw.rect(surface, BTN_BORDER,r, 1, border_radius=4)
                t = font.render(opt, True, TEXT_PRIMARY)
                surface.blit(t, t.get_rect(midleft=(r.x + 8, r.centery)))

    def handle_event(self, event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.open = not self.open
                return False
            if self.open:
                for i in range(len(self.options)):
                    r = pygame.Rect(self.rect.x,
                                    self.rect.bottom + i * self.rect.height,
                                    self.rect.width, self.rect.height)
                    if r.collidepoint(event.pos):
                        self.open  = False
                        self.index = i
                        return True
                self.open = False
        return False

