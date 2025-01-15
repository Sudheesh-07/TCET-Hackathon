from typing import Optional, Callable
# from typing import Dict, List, Set, Tuple
from enum import Enum
import numpy as np
from heapq import heappush, heappop

import tkinter as tk
from tkinter import ttk, messagebox
import time
from typing import Dict, List, Set, Tuple
import threading
from colorama import  Fore
class Direction(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

    def get_color(self) -> str:
        return {
            Direction.NORTH: "#87CEEB",  # Light blue
            Direction.SOUTH: "#98FB98",  # Light green
            Direction.EAST: "#DDA0DD",  # Light purple
            Direction.WEST: "RED"  # Light yellow
        }[self]

class Kingdom:
    while True:
        grid = int(input(f"enter number of grid (minimum 7X7) : "))
        if grid >= 7:
            break
        else:
            print(Fore.RED,"grid size should be atleast 7. Please try again.",Fore.RESET)

    print(Fore.GREEN,"WAIT...LOADING GUI",Fore.RESET)
    def __init__(self, size: int = grid):
        self.size = max(7, size)
        self.castle_pos = (size // 2, size // 2)
        self.mines: Set[Tuple[int, int]] = set()
        self.time_taken: Dict[Direction, float] = {d: float('inf') for d in Direction}
        self.grid = np.zeros((size, size), dtype=int)
        self.exit_paths: Dict[Direction, List[Tuple[int, int]]] = {}
        self.entry_paths: Dict[Direction, List[Tuple[int, int]]] = {}
        self.edge_entry_points: Dict[Direction, Tuple[int, int]] = {}
        self.king_positions: Dict[Direction, Tuple[int, int]] = {}
        self.update_callback: Optional[Callable] = None
        self.status_callback: Optional[Callable] = None
        self.simulation_start_time: Optional[float] = None
        self.casualties = {d: 0 for d in Direction}

    def set_callbacks(self, update_callback: Callable, status_callback: Callable) -> None:
        self.update_callback = update_callback
        self.status_callback = status_callback

    def get_random_edge_position(self, direction: Direction) -> Tuple[int, int]:
        """Get random position on the given edge."""
        if direction == Direction.NORTH:
            return (0, np.random.randint(0, self.size))
        elif direction == Direction.SOUTH:
            return (self.size - 1, np.random.randint(0, self.size))
        elif direction == Direction.EAST:
            return (np.random.randint(0, self.size), self.size - 1)
        else:  # WEST
            return (np.random.randint(0, self.size), 0)

    def get_king_position(self, direction: Direction) -> Tuple[int, int]:
        """Get random king position outside the grid for the given direction."""
        x, y = self.get_random_edge_position(direction)
        if direction == Direction.NORTH:
            return (-1, y)
        elif direction == Direction.SOUTH:
            return (self.size, y)
        elif direction == Direction.EAST:
            return (x, self.size)
        else:  # WEST
            return (x, -1)

    def initialize_mines(self, num_mines: int) -> None:
        """Initialize mines with random distribution across the grid."""
        self.mines.clear()
        self.edge_entry_points.clear()
        self.king_positions.clear()

        # Set random king positions
        for direction in Direction:
            self.king_positions[direction] = self.get_king_position(direction)
            self.edge_entry_points[direction] = self.get_random_edge_position(direction)

        section_size = self.size // 3
        sections = []
        for i in range(3):
            for j in range(3):
                section = []
                for x in range(i * section_size, min((i + 1) * section_size, self.size)):
                    for y in range(j * section_size, min((j + 1) * section_size, self.size)):
                        if (x, y) != self.castle_pos:
                            section.append((x, y))
                sections.append(section)

        # Distribute mines across sections
        mines_per_section = num_mines // 9
        extra_mines = num_mines % 9

        for section in sections:
            if section:
                section_mines = mines_per_section + (1 if extra_mines > 0 else 0)
                extra_mines = max(0, extra_mines - 1)

                if section_mines > 0:
                    selected_positions = np.random.choice(
                        len(section),
                        size=min(section_mines, len(section)),
                        replace=False
                    )
                    self.mines.update(section[i] for i in selected_positions)

        self._update_grid()

    def _update_grid(self) -> None:
        """Update grid with mine proximity information."""
        self.grid.fill(0)
        for x in range(self.size):
            for y in range(self.size):
                if (x, y) not in self.mines:
                    count = sum(1 for dx in [-1, 0, 1] for dy in [-1, 0, 1]
                                if 0 <= x + dx < self.size and 0 <= y + dy < self.size
                                and (x + dx, y + dy) in self.mines)
                    self.grid[x, y] = count

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int], is_entry: bool = False) -> List[Tuple[int, int]]:
        """
        Find path using A* with simpler, more reliable pathfinding for entry paths.
        """
        if start == end:
            return [start]

        visited = set()
        pq = [(0, 0, start, [start])]  # (priority, cost, current, path)

        def heuristic(pos: Tuple[int, int]) -> float:
            return abs(pos[0] - end[0]) + abs(pos[1] - end[1])

        while pq:
            _, cost, current, path = heappop(pq)

            if current == end:
                return path

            if current in visited:
                continue

            visited.add(current)

            # Simple movement in cardinal directions
            moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            if np.random.random() < 0.1:  # 10% chance to include diagonals
                moves.extend([(1, 1), (1, -1), (-1, 1), (-1, -1)])
            np.random.shuffle(moves)

            for dx, dy in moves:
                nx, ny = current[0] + dx, current[1] + dy
                next_pos = (nx, ny)

                if (0 <= nx < self.size and 0 <= ny < self.size and
                        next_pos not in visited and
                        next_pos not in self.mines):
                    move_cost = 1.4 if dx != 0 and dy != 0 else 1
                    new_cost = cost + move_cost
                    priority = new_cost + heuristic(next_pos)
                    heappush(pq, (priority, new_cost, next_pos, path + [next_pos]))

        return []

    def spy_exit_mission(self, direction: Direction) -> Tuple[bool, float]:
        """Execute spy's exit mission to reach the king."""
        if self.status_callback:
            self.status_callback(direction, "Spy starting exit mission...")

        start_time = time.time()
        edge_pos = self.get_random_edge_position(direction)

        path = self.find_path(self.castle_pos, edge_pos)
        if path:
            for pos in path:
                time.sleep(0.5)
                if pos in self.mines:
                    if self.status_callback:
                        self.status_callback(direction, "Spy hit mine during exit!")
                    return False, time.time() - start_time
                if self.update_callback:
                    self.update_callback(pos, None, direction)

            self.exit_paths[direction] = path
            return True, time.time() - start_time

        if self.status_callback:
            self.status_callback(direction, "No exit path found!")
        return False, 0

    def spy_entry_mission(self, direction: Direction) -> bool:
        """Execute spy's entry mission with improved reliability."""
        if self.status_callback:
            self.status_callback(direction, "Spy starting entry mission...")

        while True:
            max_attempts = int(input(f"Enter current king army count (minimum 1 and maximum 10) : "))
            if 1 <= max_attempts <= 10:
                break
            else:
                print("Army size must be between 1 and 10 , Please try again ")

        for attempt in range(max_attempts):
            start_pos = self.get_random_edge_position(direction)
            path = self.find_path(start_pos, self.castle_pos, is_entry=True)

            if path:
                path_safe = True
                for pos in path:
                    time.sleep(0.5)  # Reduced delay for smoother simulation
                    if pos in self.mines:
                        if self.status_callback:
                            self.status_callback(direction, "Spy found unsafe path, trying again...")
                        path_safe = False
                        break

                    if self.update_callback:
                        self.update_callback(pos, None, direction)

                if path_safe:
                    self.entry_paths[direction] = path
                    if self.status_callback:
                        self.status_callback(direction, "Spy successfully found entry path!")
                    return True

            if attempt < max_attempts - 1:  # Don't show message on last attempt
                if self.status_callback:
                    self.status_callback(direction, f"Attempt {attempt + 1} failed, trying again...")
                time.sleep(0.5)

        if self.status_callback:
            self.status_callback(direction, "All entry attempts failed!")
        return False

    def king_mission(self, direction: Direction) -> bool:
        """Execute king's mission with random casualties."""
        if direction not in self.entry_paths:
            return False

        path = self.entry_paths[direction]
        step_time = 0.5
        total_time = 0

        if self.status_callback:
            self.status_callback(direction, "King following entry path...")

        # Initialize casualties if not already set
        if direction not in self.casualties:
            self.casualties[direction] = 0

        for pos in path:
            total_time += step_time
            time.sleep(step_time)

            # Random casualty,20% chance of casualties at each step
            if np.random.random() < 0.2:
                # Random number of casualties (1-3)
                new_casualties = np.random.randint(1, 4)
                self.casualties[direction] += new_casualties

                if self.status_callback:
                    self.status_callback(direction,
                                         f"Encountered resistance! {new_casualties} casualties. Total: {self.casualties[direction]}")

                if self.update_callback:
                    self.update_callback(None, pos, direction)
                time.sleep(0.8)  # Brief pause for casualty notification
            else:
                if self.update_callback:
                    self.update_callback(None, pos, direction)

        self.time_taken[direction] = total_time
        return True

    def reset(self) -> None:
        """Reset the kingdom state."""
        self.mines.clear()
        self.time_taken = {d: float('inf') for d in Direction}
        self.casualties = {d: 0 for d in Direction}
        self.exit_paths.clear()
        self.entry_paths.clear()
        self.edge_entry_points.clear()
        self.grid.fill(0)
        self.simulation_start_time = None


# def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
#     """Calculate Manhattan distance between two points."""
#     return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


class KingdomGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Kingdom Pathfinding Simulation")
        self.kingdom = Kingdom()
        self.simulation_running = False
        self.setup_gui()
        self.kingdom.set_callbacks(self.update_grid, self.update_status)

    def setup_gui(self):
        self.root.configure(bg='white')
        self.root.resizable(False, False)

        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")

        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="5")
        control_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        ttk.Label(control_frame, text="Number of Mines:").grid(row=0, column=0, padx=5)
        self.mine_count = tk.StringVar(value="40")
        mine_entry = ttk.Entry(control_frame, textvariable=self.mine_count, width=10)
        mine_entry.grid(row=0, column=1, padx=5)

        ttk.Button(control_frame, text="Start Simulation", command=self.start_simulation).grid(row=0, column=2, padx=5)
        ttk.Button(control_frame, text="Reset", command=self.reset_simulation).grid(row=0, column=3, padx=5)

        self.extended_grid_frame = ttk.Frame(main_frame, padding="5")
        self.extended_grid_frame.grid(row=1, column=0, sticky="nsew")

        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="5")
        status_frame.grid(row=1, column=1, sticky="nsew", padx=(10, 0))

        self.status_vars = {}
        self.direction_labels = {}
        for direction in Direction:
            frame = ttk.Frame(status_frame)
            frame.pack(fill="x", pady=2)
            self.status_vars[direction] = tk.StringVar(value=f"{direction.name}: Waiting...")
            label = ttk.Label(frame, textvariable=self.status_vars[direction],
                              background=direction.get_color())
            label.pack(fill="x", padx=2)
            self.direction_labels[direction] = label

        self.phase_var = tk.StringVar(value="Ready to start")
        ttk.Label(status_frame, textvariable=self.phase_var, wraplength=200).pack(anchor="w", pady=10)

        self.create_extended_grid()

    def create_extended_grid(self):
        self.cells = {}
        grid_size = self.kingdom.size

        for i in range(-1, grid_size + 1):
            for j in range(-1, grid_size + 1):
                is_main_grid = 0 <= i < grid_size and 0 <= j < grid_size

                cell = ttk.Label(self.extended_grid_frame, width=3,
                                 text="·" if is_main_grid else " ",
                                 background="white" if is_main_grid else "lightgray",
                                 relief="solid" if is_main_grid else "flat")
                cell.grid(row=i + 1, column=j + 1, padx=1, pady=1)
                self.cells[(i, j)] = cell

        castle_cell = self.cells[self.kingdom.castle_pos]
        castle_cell.configure(text="🏰", background="yellow")

    def update_grid(self, spy_pos=None, king_pos=None, direction=None):

        try:
            # Reset all cells first
            for pos, cell in self.cells.items():
                i, j = pos
                is_main_grid = 0 <= i < self.kingdom.size and 0 <= j < self.kingdom.size

                # Get the direction for this position if it's a king's position
                pos_direction = next((d for d, p in self.kingdom.king_positions.items() if p == pos), None)

                # Determine cell appearance based on position type
                if pos == spy_pos and direction:
                    cell.configure(text="🕵️", background=direction.get_color())
                elif pos == king_pos and direction:
                    cell.configure(text="👑", background=direction.get_color())
                elif pos == self.kingdom.castle_pos:
                    cell.configure(text="🏰", background="yellow")
                elif is_main_grid and pos in self.kingdom.mines:
                    cell.configure(text="💣", background="red")
                elif pos_direction:  # If this is a king's position
                    cell.configure(text="👑", background=pos_direction.get_color())
                elif is_main_grid:
                    cell.configure(text=str(self.kingdom.grid[i, j]) if self.kingdom.grid[i, j] > 0 else "·",
                                   background="white")
                else:
                    cell.configure(text=" ", background="lightgray")

            self.root.update_idletasks()
        except Exception as e:
            print(f"Error updating grid: {e}")

    def update_status(self, direction, status):
        """Update status message for a direction."""
        try:
            self.status_vars[direction].set(f"{direction.name}: {status}")
            self.root.update_idletasks()
        except Exception as e:
            print(f"Error updating status: {e}")

    def simulation_thread(self):
        """Run the simulation in a separate thread with proper exit-before-entry logic."""
        try:
            # Validate mine count
            num_mines = int(self.mine_count.get())
            max_mines = (self.kingdom.size * self.kingdom.size) // 3 #3 for 3 x 3 grid
            min_mine = max_mines // 2
            if num_mines == 0 or num_mines > max_mines or num_mines < min_mine:
                raise ValueError(f"Invalid number of mines. Please use between {min_mine} and {max_mines}")

            # Initialize mines and update display
            self.kingdom.initialize_mines(num_mines)
            self.update_grid()
            self.kingdom.simulation_start_time = time.time()

            # Phase 1: Spy Exit Missions
            self.phase_var.set("Phase 1: Spies finding paths to kings")
            exit_times = {}
            successful_exits = set()  # Track which directions had successful exits

            for direction in Direction:
                if not self.simulation_running:
                    return

                king_pos = self.kingdom.king_positions[direction]
                self.update_grid(king_pos=king_pos, direction=direction)

                success, exit_time = self.kingdom.spy_exit_mission(direction)
                exit_times[direction] = exit_time

                if success:
                    successful_exits.add(direction)
                    status = f"{direction.name}: Found exit path in {exit_time:.1f}s"
                else:
                    status = f"{direction.name}: Failed to exit - king's army cannot proceed"

                self.status_vars[direction].set(status)
                time.sleep(0.7)

            if not successful_exits:
                self.phase_var.set("Simulation complete - no spies found exit paths")
                messagebox.showinfo("🎮 Simulation Complete",
                                    "No spies found exit paths!\nThe castle remains safe... 🏰")
                return

            self.phase_var.set("Repositioning mines for entry phase...")
            self.kingdom.initialize_mines(num_mines)
            self.update_grid()
            time.sleep(2)

            self.phase_var.set("Phase 2: Spies finding entry paths")
            king_results = {}  # Store results for each king
            successful_kings = []

            # Only attempt entry missions for directions that had successful exits
            for direction in successful_exits:
                if not self.simulation_running:
                    return

                king_pos = self.kingdom.king_positions[direction]
                self.update_grid(king_pos=king_pos, direction=direction)

                if self.kingdom.spy_entry_mission(direction):
                    time.sleep(1)
                    if self.kingdom.king_mission(direction):
                        successful_kings.append(direction)
                        king_results[direction] = {
                            'time': self.kingdom.time_taken[direction],
                            'casualties': self.kingdom.casualties[direction]
                        }
                        self.status_vars[direction].set(
                            f"{direction.name}: Reached castle in {self.kingdom.time_taken[direction]:.1f}s "
                            f"with {self.kingdom.casualties[direction]} casualties"
                        )
                    else:
                        self.status_vars[direction].set(
                            f"{direction.name}: Mission incomplete after {self.kingdom.time_taken[direction]:.1f}s"
                        )
                else:
                    self.status_vars[direction].set(f"{direction.name}: No safe entry path found")

            self.spy_path()

            if successful_kings:
                # Determine winner based on weighted score of time and casualties
                def get_score(direction):
                    result = king_results[direction]
                    # Weight: 50% time, 50% casualties
                    time_score = result['time'] / max(r['time'] for r in king_results.values())
                    casualty_score = result['casualties'] / max(r['casualties'] for r in king_results.values())
                    return (time_score * 0.5) + (casualty_score * 0.5)

                winner = min(successful_kings, key=get_score)

                result = f"🏆 Winner: {winner.name} 🏆\n"
                result += f"Time: {king_results[winner]['time']:.1f}s\n"
                result += f"Casualties: {king_results[winner]['casualties']}\n\n"
                result += "Detailed Results:\n"

                for direction in Direction:
                    result += f"\n\n{direction.name}:\n"
                    if direction in successful_exits:
                        result += f"Spy Exit Time: {exit_times[direction]:.1f}s\n"
                        if direction in king_results:
                            result += f"Time: {king_results[direction]['time']:.1f}s\n"
                            result += f"Casualties: {king_results[direction]['casualties']}"
                            result += " ✅\n" if direction in successful_kings else " ❌\n"
                        else:
                            result += "Failed to reach castle ❌"
                    else:
                        result += "Spy failed to find exit path ❌"

                messagebox.showinfo("🎮 Simulation Complete", result)
            else:
                messagebox.showinfo("🎮 Simulation Complete",
                                    "No kings reached the castle!\nThe castle remains unconquered... 👑")

        except ValueError as e:
            messagebox.showerror("Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
        finally:
            self.simulation_running = False
            self.phase_var.set("Simulation complete")
            print((self.kingdom.casualties))
            print(type(self.kingdom.casualties))

    def spy_path(self):
        def format_path(path: List[Tuple[int, int]]) -> str:
            if not path:
                return "No path found"
            return " → ".join([f"({x},{y})" for x, y in path])

        path = []
        path.append("\n=== Spy Mission Report ===\n")

        for direction in Direction:
            path.append(f"\n{direction.name} SPY MISSIONS:")
            path.append("-" * 50)

            path.append("EXIT PATH (Castle → King):")
            if direction in self.kingdom.exit_paths:
                exit_path = self.kingdom.exit_paths[direction]
                path.append(format_path(exit_path))
                path.append(f"Total steps: {len(exit_path)}")
            else:
                path.append("No exit path found")

            path.append("ENTRY PATH (King → Castle):")

            if direction in self.kingdom.entry_paths:
                entry_path = self.kingdom.entry_paths[direction]
                path.append(format_path(entry_path))
                path.append(f"Total steps: {len(entry_path)}")
            else:
                path.append("No entry path found")

            path.append("-" * 50)

        full_path = "\n".join(path)
        messagebox.showinfo("Spy Path Report", full_path)

    def start_simulation(self):
        """Start the simulation if not already running."""
        if not self.simulation_running:
            self.simulation_running = True
            threading.Thread(target=self.simulation_thread, daemon=True).start()

    def reset_simulation(self):
        """Reset the simulation state."""
        self.simulation_running = False
        self.kingdom.reset()
        for direction in Direction:
            self.status_vars[direction].set(f"{direction.name}: Waiting...")
        self.phase_var.set("Ready to start")
        self.update_grid()


def main():
    """Main entry point of the application."""
    try:
        root = tk.Tk()
        app = KingdomGUI(root)
        root.mainloop()
    except Exception as e:
        print(f"Fatal error: {e}")
        messagebox.showerror("Fatal Error",
                             f"The application encountered a fatal error: {str(e)}")


if __name__ == "__main__":
    main()