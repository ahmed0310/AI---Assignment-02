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



# ══════════════════════════════════════════════════════════════════════
#  SECTION 5 — MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════

WIN_W, WIN_H  = 1100, 700
FPS           = 30
AGENT_STEP_MS = 80
DYN_SPAWN_MS  = 600
DYN_PROB      = 0.40
DEFAULT_ROWS  = 20
DEFAULT_COLS  = 30


class App:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Dynamic Pathfinding Agent — AI2002 Q6")
        self.screen = pygame.display.set_mode((WIN_W, WIN_H), pygame.RESIZABLE)
        self.clock  = pygame.time.Clock()

        self.grid  = Grid(DEFAULT_ROWS, DEFAULT_COLS)
        self.agent = Agent()

        self.font_lg = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_md = pygame.font.SysFont("Consolas", 13)
        self.font_sm = pygame.font.SysFont("Consolas", 11)

        self._build_controls()

        self.metrics = {"Nodes Visited": 0, "Path Cost": 0, "Time (ms)": 0}
        self.status  = "Ready — configure grid and press Run"
        self.drawing_wall = False
        self.erasing_wall = False
        self._last_step_t = 0
        self._last_dyn_t  = 0

    # ── Cell sizing helpers ──────────────────────────────────────────

    def _grid_area(self):
        return self.screen.get_width() - PANEL_W, self.screen.get_height()

    def _cell_size(self):
        gw, gh = self._grid_area()
        return max(4, gw // self.grid.cols), max(4, gh // self.grid.rows)

    def _cell_rect(self, cell: Cell) -> pygame.Rect:
        cw, ch = self._cell_size()
        return pygame.Rect(cell.col * cw, cell.row * ch, cw - 1, ch - 1)

    def _pixel_to_cell(self, px, py):
        gw, _ = self._grid_area()
        if px >= gw:
            return None
        cw, ch = self._cell_size()
        col, row = px // cw, py // ch
        if 0 <= row < self.grid.rows and 0 <= col < self.grid.cols:
            return row, col
        return None

    # ── Controls ────────────────────────────────────────────────────

    def _build_controls(self):
        px = self.screen.get_width() - PANEL_W + 12
        y  = 68
        self.dd_algo  = Dropdown((px, y, 234, 30), ["A* Search", "Greedy BFS"], "Algorithm"); y += 120
        self.dd_heur  = Dropdown((px, y, 234, 30), ["Manhattan", "Euclidean"],  "Heuristic"); y += 100
        self.btn_run  = Button((px, y, 234, 34), "▶  Run Search");    y += 44
        self.btn_reset= Button((px, y, 234, 34), "↺  Reset Grid");    y += 44
        self.btn_rand = Button((px, y, 234, 34), "⚡  Random Map");    y += 44
        self.btn_dyn  = Button((px, y, 234, 34), "⚠  Dynamic Mode", toggle=True)

    # ── Main loop ───────────────────────────────────────────────────

    def run(self):
        while True:
            self.clock.tick(FPS)
            self._handle_events()
            self._update()
            self._draw()

    # ── Events ──────────────────────────────────────────────────────

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            if event.type == pygame.VIDEORESIZE:
                self.screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
                self._build_controls()

            self.dd_algo.handle_event(event)
            self.dd_heur.handle_event(event)

            if self.btn_run.handle_event(event):   self._run_search()
            if self.btn_reset.handle_event(event): self._reset()
            if self.btn_rand.handle_event(event):
                self.grid.generate_random_maze(0.30)
                self._clear_result()
            self.btn_dyn.handle_event(event)

            # Wall drawing
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                rc = self._pixel_to_cell(*event.pos)
                if rc:
                    cell = self.grid.get(*rc)
                    if cell:
                        self.erasing_wall = cell.cell_type == WALL
                        self.drawing_wall = not self.erasing_wall
                        self._paint(event.pos)

            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.drawing_wall = self.erasing_wall = False

            if event.type == pygame.MOUSEMOTION:
                if self.drawing_wall or self.erasing_wall:
                    self._paint(event.pos)

    def _paint(self, pos):
        rc = self._pixel_to_cell(*pos)
        if not rc:
            return
        if self.drawing_wall:
            self.grid.place_wall(*rc)
        elif self.erasing_wall:
            self.grid.remove_wall(*rc)

    # ── Update ──────────────────────────────────────────────────────

    def _update(self):
        now = pygame.time.get_ticks()

        if self.agent.moving and now - self._last_step_t > AGENT_STEP_MS:
            self._last_step_t = now
            moved = self.agent.step()
            if not moved and self.agent.needs_replan():
                self._do_replan()

        if self.btn_dyn.active and self.agent.moving:
            if now - self._last_dyn_t > DYN_SPAWN_MS:
                self._last_dyn_t = now
                if random.random() < DYN_PROB:
                    new_wall = self.grid.spawn_random_obstacle()
                    if new_wall and self.agent.is_path_blocked():
                        self.status = f"⚠  Obstacle at ({new_wall.row},{new_wall.col}) — Replanning…"
                        self._do_replan()

        if self.agent.reached_goal:
            self.status       = "✅  Goal Reached!"
            self.agent.moving = False

    def _do_replan(self):
        if not self.agent.cell or not self.grid.goal_cell:
            return
        t0  = time.perf_counter()
        res = replan(self.grid, self.agent.cell, self.grid.goal_cell,
                     self.dd_algo.value, self.dd_heur.value)
        ms  = round((time.perf_counter() - t0) * 1000, 2)
        if res.found:
            self.agent.accept_replan(res.path)
            self.metrics["Nodes Visited"] += res.nodes_visited
            self.metrics["Path Cost"]      = round(res.path_cost, 2)
            self.metrics["Time (ms)"]     += ms
            self.status = f"🔄  Replanned — cost {res.path_cost}"
        else:
            self.agent.moving = False
            self.status = "❌  No path found after replanning"

    # ── Search ──────────────────────────────────────────────────────

    def _run_search(self):
        if not self.grid.start_cell or not self.grid.goal_cell:
            self.status = "Set start and goal first!"; return
        self._clear_result()
        self.grid.reset_search_state()

        t0  = time.perf_counter()
        res = run_search(self.grid, self.dd_algo.value, self.dd_heur.value)
        ms  = round((time.perf_counter() - t0) * 1000, 2)

        self.metrics = {"Nodes Visited": res.nodes_visited,
                        "Path Cost":     round(res.path_cost, 2),
                        "Time (ms)":     ms}
        if res.found:
            self.status = f"Path found!  Cost={res.path_cost}  Nodes={res.nodes_visited}"
            self.agent.reset()
            self.agent.set_path(res.path)
            self._last_step_t = pygame.time.get_ticks()
        else:
            self.status = "No path exists."

    def _reset(self):
        self.grid.full_reset()
        self.agent.reset()
        self._clear_result()
        self.status = "Grid reset."

    def _clear_result(self):
        self.metrics = {"Nodes Visited": 0, "Path Cost": 0, "Time (ms)": 0}
        self.agent.reset()

    # ── Draw ────────────────────────────────────────────────────────

    def _cell_color(self, cell: Cell):
        if cell.cell_type == WALL:  return CELL_WALL
        if cell.cell_type == START: return CELL_START
        if cell.cell_type == GOAL:  return CELL_GOAL
        if cell.on_path:            return CELL_PATH
        if cell.in_closed:          return CELL_CLOSED
        if cell.in_open:            return CELL_OPEN
        return CELL_EMPTY

    def _draw(self):
        W, H   = self.screen.get_size()
        panel_x= W - PANEL_W

        self.screen.fill(BG_COLOR)

        # ── Grid ──
        for row in self.grid.cells:
            for cell in row:
                pygame.draw.rect(self.screen, self._cell_color(cell), self._cell_rect(cell))

        # ── Agent dot ──
        if self.agent.cell:
            r    = self._cell_rect(self.agent.cell)
            rad  = max(3, min(r.width, r.height) // 3)
            pygame.draw.circle(self.screen, CELL_AGENT, r.center, rad)

        # ── Panel background ──
        pygame.draw.rect(self.screen, PANEL_BG, (panel_x, 0, PANEL_W, H))
        pygame.draw.line(self.screen, PANEL_BORDER, (panel_x, 0), (panel_x, H), 2)

        # Title
        self.screen.blit(self.font_lg.render("PATHFINDER", True, TEXT_ACCENT),        (panel_x+12, 14))
        self.screen.blit(self.font_sm.render("AI2002  ·  Q6 Visualizer", True, TEXT_SECONDARY), (panel_x+12, 36))
        pygame.draw.line(self.screen, PANEL_BORDER, (panel_x+8, 54), (W-8, 54), 1)

        # Dropdowns & buttons
        self.dd_algo.draw(self.screen, self.font_md, self.font_sm)
        self.dd_heur.draw(self.screen, self.font_md, self.font_sm)
        for btn in (self.btn_run, self.btn_reset, self.btn_rand, self.btn_dyn):
            btn.draw(self.screen, self.font_md)

        # Metrics
        pygame.draw.line(self.screen, PANEL_BORDER, (panel_x+8, H-170), (W-8, H-170), 1)
        my = H - 158
        self.screen.blit(self.font_lg.render("METRICS", True, TEXT_ACCENT), (panel_x+12, my)); my += 26
        for label, val in self.metrics.items():
            self.screen.blit(self.font_sm.render(label+":", True, TEXT_SECONDARY), (panel_x+12, my))
            self.screen.blit(self.font_md.render(str(val),  True, TEXT_PRIMARY),   (panel_x+12, my+14))
            my += 36

        # Legend
        pygame.draw.line(self.screen, PANEL_BORDER, (panel_x+8, H-36), (W-8, H-36), 1)
        legend = [(CELL_OPEN,"Frontier"),(CELL_CLOSED,"Explored"),(CELL_PATH,"Path"),(CELL_AGENT,"Agent")]
        lx = panel_x + 10
        for color, lbl in legend:
            pygame.draw.rect(self.screen, color, (lx, H-26, 10, 10), border_radius=2)
            self.screen.blit(self.font_sm.render(lbl, True, TEXT_SECONDARY), (lx+14, H-28))
            lx += 60

        # Status bar
        self.screen.blit(self.font_sm.render(self.status, True, TEXT_SECONDARY), (8, H-18))

        pygame.display.flip()



App().run()