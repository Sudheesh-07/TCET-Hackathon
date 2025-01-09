import tkinter as tk
from tkinter import ttk, messagebox
import time
from typing import Dict, List, Set, Tuple
import threading
from enum import Enum
import numpy as np
from heapq import heappush, heappop

class Direction(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

    def get_color(self):
        return {
            Direction.NORTH: "#87CEEB",
            Direction.SOUTH: "#98FB98",
            Direction.EAST: "#DDA0DD",
            Direction.WEST: "#F0E68C"
        }[self]

class Kingdom:
    def __init__(self, size: int = 30):
        self.size = size
        self.castle_pos = (size // 2, size // 2)
        self.mines = set()
        self.casualties = {d: float('inf') for d in Direction}
        self.grid = np.zeros((size, size), dtype=int)
        self.exit_paths = {}
        self.entry_paths = {}
        self.update_callback = None
        self.status_callback = None
        self.simulation_start_time = None

    def set_callbacks(self, update_callback, status_callback):
        self.update_callback = update_callback
        self.status_callback = status_callback

    def initialize_mines(self, num_mines: int):
        self.mines.clear()
        while len(self.mines) < num_mines:
            x, y = np.random.randint(0, self.size), np.random.randint(0, self.size)
            if (x, y) != self.castle_pos:
                self.mines.add((x, y))
        self._update_grid()

    def _update_grid(self):
        self.grid.fill(0)
        for x in range(self.size):
            for y in range(self.size):
                if (x, y) not in self.mines:
                    count = sum(1 for dx in [-1, 0, 1] for dy in [-1, 0, 1]
                              if 0 <= x + dx < self.size and 0 <= y + dy < self.size
                              and (x + dx, y + dy) in self.mines)
                    self.grid[x, y] = count

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        visited = set()
        pq = [(0, start, [start])]

        while pq:
            danger, current, path = heappop(pq)

            if current == end:
                return path

            if current in visited:
                continue

            visited.add(current)

            if self.update_callback:
                self.update_callback(current, None, None)
                time.sleep(0.3)
                self.root.update_idletasks()

            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = current[0] + dx, current[1] + dy
                next_pos = (nx, ny)

                if (0 <= nx < self.size and 0 <= ny < self.size and
                    next_pos not in visited and
                    next_pos not in self.mines):
                    new_danger = danger + self.grid[nx, ny]
                    heappush(pq, (new_danger, next_pos, path + [next_pos]))

        return []

    def spy_exit_mission(self, direction: Direction) -> bool:
        if self.status_callback:
            self.status_callback(direction, "Spy finding exit path...")

        king_pos = self.get_position(direction)
        path = self.find_path(self.castle_pos, king_pos)

        if not path:
            if self.status_callback:
                self.status_callback(direction, "No exit path found!")
            return False

        for pos in path:
            time.sleep(0.3)
            if pos in self.mines:
                if self.status_callback:
                    self.status_callback(direction, "Spy hit mine during exit!")
                return False
            if self.update_callback:
                self.update_callback(pos, None, direction)
                self.root.update_idletasks()

        self.exit_paths[direction] = path
        return True

    def spy_entry_mission(self, direction: Direction) -> bool:
        if self.status_callback:
            self.status_callback(direction, "Spy finding entry path...")

        king_pos = self.get_position(direction)
        path = self.find_path(king_pos, self.castle_pos)

        if not path:
            if self.status_callback:
                self.status_callback(direction, "No entry path found!")
            return False

        for pos in path:
            time.sleep(0.3)
            if pos in self.mines:
                if self.status_callback:
                    self.status_callback(direction, "Spy hit mine during entry!")
                return False
            if self.update_callback:
                self.update_callback(pos, None, direction)

        self.entry_paths[direction] = path
        return True

    def king_mission(self, direction: Direction) -> bool:
        if direction not in self.entry_paths:
            return False

        path = self.entry_paths[direction]
        step_time = 0.3
        total_time = 0

        if self.status_callback:
            self.status_callback(direction, "King following entry path...")

        for pos in path:
            total_time += step_time
            time.sleep(step_time)
            if pos in self.mines:
                self.casualties[direction] = total_time
                if self.status_callback:
                    self.status_callback(direction, "King's army hit a mine!")
                if self.update_callback:
                    self.update_callback(None, pos, direction)
                return False

            if self.update_callback:
                self.update_callback(None, pos, direction)
                self.root.update_idletasks()

        self.casualties[direction] = total_time
        return True

    def get_position(self, direction: Direction) -> Tuple[int, int]:
        return {
            Direction.NORTH: (0, self.size // 2),
            Direction.SOUTH: (self.size - 1, self.size // 2),
            Direction.EAST: (self.size // 2, self.size - 1),
            Direction.WEST: (self.size // 2, 0)
        }[direction]

    def reset(self):
        self.mines.clear()
        self.casualties = {d: float('inf') for d in Direction}
        self.exit_paths.clear()
        self.entry_paths.clear()
        self.grid.fill(0)
        self.simulation_start_time = None

class KingdomGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Kingdom Pathfinding Simulation")
        self.kingdom = Kingdom()
        self.simulation_running = False
        self.setup_gui()

    def setup_gui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")

        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="5")
        control_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        ttk.Label(control_frame, text="Number of Mines:").grid(row=0, column=0, padx=5)
        self.mine_count = tk.StringVar(value="100")
        ttk.Entry(control_frame, textvariable=self.mine_count, width=10).grid(row=0, column=1, padx=5)

        ttk.Button(control_frame, text="Start Simulation", command=self.start_simulation).grid(row=0, column=2, padx=5)
        ttk.Button(control_frame, text="Reset", command=self.reset_simulation).grid(row=0, column=3, padx=5)

        self.grid_frame = ttk.Frame(main_frame, padding="5")
        self.grid_frame.grid(row=1, column=0, sticky="nsew")

        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="5")
        status_frame.grid(row=1, column=1, sticky="nsew", padx=(10, 0))

        self.status_vars = {}
        for direction in Direction:
            self.status_vars[direction] = tk.StringVar(value=f"{direction.name}: Waiting...")
            ttk.Label(status_frame, textvariable=self.status_vars[direction]).pack(anchor="w", pady=2)

        self.phase_var = tk.StringVar(value="Ready to start")
        ttk.Label(status_frame, textvariable=self.phase_var, wraplength=200).pack(anchor="w", pady=10)

        self.create_grid()

    def create_grid(self):
        self.cells = {}
        for i in range(self.kingdom.size):
            for j in range(self.kingdom.size):
                cell = ttk.Label(self.grid_frame, width=3, text="·",
                               background="white", relief="solid")
                cell.grid(row=i, column=j, padx=1, pady=1)
                self.cells[(i, j)] = cell

    def update_grid(self, spy_pos=None, king_pos=None, direction=None):
        for pos, cell in self.cells.items():
            i, j = pos
            if pos == spy_pos:
                cell.configure(text="🕵️", background=direction.get_color() if direction else "gray")
            elif pos == king_pos:
                cell.configure(text="👑", background=direction.get_color() if direction else "gray")
            elif pos == self.kingdom.castle_pos:
                cell.configure(text="🏰", background="yellow")
            elif pos in self.kingdom.mines:
                cell.configure(text="💣", background="red")
            else:
                cell.configure(text=str(self.kingdom.grid[i, j]) if self.kingdom.grid[i, j] > 0 else "·",
                            background="white")

    def simulation_thread(self):
        try:
            num_mines = int(self.mine_count.get())
            self.kingdom.initialize_mines(num_mines)
            self.update_grid()
            self.kingdom.simulation_start_time = time.time()

            self.phase_var.set("Phase 1: Spies finding paths to kings")
            for direction in Direction:
                if not self.simulation_running:
                    return
                success = self.kingdom.spy_exit_mission(direction)
                self.status_vars[direction].set(
                    f"{direction.name}: {'Found exit path' if success else 'Failed to exit'}"
                )

            self.phase_var.set("Mines repositioning...")
            self.kingdom.initialize_mines(num_mines)
            self.update_grid()
            time.sleep(2)

            self.phase_var.set("Phase 2: Spies finding entry paths")
            successful_kings = []

            for direction in Direction:
                if not self.simulation_running:
                    return

                if self.kingdom.spy_entry_mission(direction):
                    if self.kingdom.king_mission(direction):
                        successful_kings.append(direction)
                        self.status_vars[direction].set(
                            f"{direction.name}: Reached castle in {self.kingdom.casualties[direction]:.1f}s"
                        )
                    else:
                        self.status_vars[direction].set(f"{direction.name}: King failed")
                else:
                    self.status_vars[direction].set(f"{direction.name}: No safe entry path")

            if successful_kings:
                winner = min(successful_kings, key=lambda d: self.kingdom.casualties[d])
                result = f"Winner: {winner.name}\n\nResults:\n"
                for direction in Direction:
                    time_taken = self.kingdom.casualties[direction]
                    result += f"\n{direction.name}: "
                    result += f"{time_taken:.1f}s" if time_taken != float('inf') else "Failed"
                result += "\n\nExit and Entry Paths:\n"
                for direction in Direction:
                    exit_path = " -> ".join([f"({x},{y})" for x, y in self.kingdom.exit_paths.get(direction, [])])
                    entry_path = " -> ".join([f"({x},{y})" for x, y in self.kingdom.entry_paths.get(direction, [])])
                    result += f"{direction.name} Exit: {exit_path}\n{direction.name} Entry: {entry_path}\n"
                messagebox.showinfo("Simulation Complete", result)
            else:
                messagebox.showinfo("Simulation Complete", "No kings reached the castle!")

        except ValueError as e:
            messagebox.showerror("Error", "Invalid mine count")
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self.simulation_running = False
            self.phase_var.set("Simulation complete")

    def start_simulation(self):
        if not self.simulation_running:
            self.simulation_running = True
            threading.Thread(target=self.simulation_thread, daemon=True).start()

    def reset_simulation(self):
        self.simulation_running = False
        self.kingdom.reset()
        for direction in Direction:
            self.status_vars[direction].set(f"{direction.name}: Waiting...")
        self.phase_var.set("Ready to start")
        self.update_grid()

def main():
    root = tk.Tk()
    app = KingdomGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
