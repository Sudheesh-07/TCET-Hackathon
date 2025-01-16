from typing import  Optional, Callable
# from typing import Dict, List, Set, Tuple
from enum import Enum
import numpy as np
from heapq import heappush, heappop
# import time
import tkinter as tk
from tkinter import ttk, messagebox
import time
from typing import Dict, List, Set, Tuple
import threading


class Direction(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

    def get_color(self) -> str:
        return {
            Direction.NORTH: "CYAN",  # Light blue
            Direction.SOUTH: "GREEN",  # Light green
            Direction.EAST: "BLUE",  # Light purple
            Direction.WEST: "PINK"  # Light yellow
        }[self]

class SpyMission(threading.Thread):
    def __init__(self, kingdom, direction, mission_type="exit", gui_update_lock=None):
        super().__init__()
        self.kingdom = kingdom
        self.direction = direction
        self.mission_type = mission_type
        self.success = False
        self.time_taken = float('inf')
        self.path = []
        self.gui_update_lock = gui_update_lock

    def run(self):
        if self.mission_type == "exit":
            with self.gui_update_lock:
                self.success, self.time_taken = self.kingdom.spy_exit_mission(self.direction)
            if self.success:
                self.path = self.kingdom.exit_paths.get(self.direction, [])
        else:  # entry
            with self.gui_update_lock:
                self.success = self.kingdom.spy_entry_mission(self.direction)
            if self.success:
                self.path = self.kingdom.entry_paths.get(self.direction, [])
                
class Kingdom:
    grid = int(input("enter number of grid: "))
    def __init__(self, size: int = grid):
        self.size = max(7, size)
        self.castle_pos = (size // 2, size // 2)
        self.mines: Set[Tuple[int, int]] = set()
        self.casualties: Dict[Direction, float] = {d: float('inf') for d in Direction}
        self.grid = np.zeros((size, size), dtype=int)
        self.exit_paths: Dict[Direction, List[Tuple[int, int]]] = {}
        self.entry_paths: Dict[Direction, List[Tuple[int, int]]] = {}
        self.edge_entry_points: Dict[Direction, Tuple[int, int]] = {}
        self.king_positions: Dict[Direction, Tuple[int, int]] = {}
        self.update_callback: Optional[Callable] = None
        self.status_callback: Optional[Callable] = None
        self.simulation_start_time: Optional[float] = None

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
                    print(self.mines)

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
        Find path using A* with significantly increased difficulty for entry paths.
        """
        if start == end:
            return [start]

        visited = set()
        pq = [(0, 0, start, [start])]  # (priority, cost, current, path)

        def heuristic(pos: Tuple[int, int]) -> float:
            manhattan = abs(pos[0] - end[0]) + abs(pos[1] - end[1])
            if is_entry:
                # Dramatically increased randomness for entry paths
                random_factor = np.random.uniform(0, 4.0)  # Increased from 2.0

                # Higher chance of path distortion
                if np.random.random() < 0.5:  # Increased from 0.3
                    random_factor *= 3  # Increased from 2

                # Add periodic difficulty spikes
                distance_to_castle = manhattan_distance(pos, self.castle_pos)
                if distance_to_castle < self.size // 2:  # Closer to castle
                    random_factor *= 2  # Makes it harder to approach the castle
            else:
                random_factor = np.random.uniform(0, 0.8)

            return manhattan + random_factor

        path_attempts = 0
        max_attempts = 3 if is_entry else 1   #no of kings army

        while pq and path_attempts < max_attempts:
            _, cost, current, path = heappop(pq)

            if current == end:
                path_attempts += 1
                # Higher rejection chance for entry paths
                if is_entry:
                    # 60% chance to reject a path and try again
                    if np.random.random() < 0.6:  # Increased from 0.2
                        continue
                    # 40% chance to require a minimum path length
                    if np.random.random() < 0.4 and len(path) < self.size * 1.2:
                        continue
                return path

            if current in visited:
                continue

            visited.add(current)

            moves = [(0, 1), (1, 0), (0, -1), (-1, 0),
                     (1, 1), (1, -1), (-1, 1), (-1, -1)]

            if is_entry:
                # Add more random movement patterns
                if np.random.random() < 0.3:
                    moves = moves[:4]  # Sometimes only allow orthogonal moves
                else:
                    np.random.shuffle(moves)

            for dx, dy in moves:
                nx, ny = current[0] + dx, current[1] + dy
                next_pos = (nx, ny)

                if (0 <= nx < self.size and 0 <= ny < self.size and
                        next_pos not in visited and
                        next_pos not in self.mines):

                    move_cost = 1.4 if dx != 0 and dy != 0 else 1

                    if is_entry:
                        # Significantly increased base cost for entry paths
                        move_cost *= (1 + np.random.uniform(0, 1.5))  # Increased from 0.5

                        # Add extra cost near the castle
                        distance_to_castle = manhattan_distance(next_pos, self.castle_pos)
                        if distance_to_castle < self.size // 2:
                            move_cost *= 2

                        # Random difficulty spikes
                        if np.random.random() < 0.2:
                            move_cost *= 2

                        # Terrain difficulty based on grid values
                        if self.grid[nx, ny] > 0:
                            move_cost *= (1 + self.grid[nx, ny] * 0.5)

                    new_cost = cost + move_cost + self.grid[nx, ny]
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
                time.sleep(0.2)
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
        """Execute spy's entry mission with increased difficulty."""
        if self.status_callback:
            self.status_callback(direction, "Spy starting entry mission...")

        # Multiple entry attempts with increasing difficulty
        max_attempts = int(input(f"Enter current kings army count:"))
        for attempt in range(max_attempts):
            if attempt > 0:
                if self.status_callback:
                    self.status_callback(direction, f"Spy making another attempt ({attempt + 1}/{max_attempts})...")
                time.sleep(1)  # Pause between attempts

            start_pos = self.get_random_edge_position(direction)
            path = self.find_path(start_pos, self.castle_pos, is_entry=True)

            if path:
                # Verify path safety with higher scrutiny
                path_safe = True
                for pos in path:
                    time.sleep(0.2)
                    if pos in self.mines:
                        if self.status_callback:
                            self.status_callback(direction, "Spy hit mine during entry!")
                        path_safe = False
                        break

                    # Random chance of path becoming unsafe
                    if np.random.random() < 0.1:  # 10% chance at each step
                        if self.status_callback:
                            self.status_callback(direction, "Spy lost the safe path!")
                        path_safe = False
                        break

                    if self.update_callback:
                        self.update_callback(pos, None, direction)

                if path_safe:
                    self.entry_paths[direction] = path
                    return True

        if self.status_callback:
            self.status_callback(direction, "All entry attempts failed!")
        return False

    def king_mission(self, direction: Direction) -> bool:
        """Execute king's mission to reach castle."""
        if direction not in self.entry_paths:
            return False

        path = self.entry_paths[direction]
        step_time = 0.2
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

        self.casualties[direction] = total_time
        return True

    def reset(self) -> None:
        """Reset the kingdom state."""
        self.mines.clear()
        self.casualties = {d: float('inf') for d in Direction}
        self.exit_paths.clear()
        self.entry_paths.clear()
        self.edge_entry_points.clear()
        self.grid.fill(0)
        self.simulation_start_time = None


def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    """Calculate Manhattan distance between two points."""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


class KingdomGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Kingdom Pathfinding Simulation")
        self.kingdom = Kingdom()
        self.simulation_running = False
        self.spy_states = {}
        self.setup_gui()
        self.kingdom.set_callbacks(self.update_grid, self.update_status)
        
        # Track spy movements
        self.active_spies = set()
        self.spy_paths = {}
        self.spy_positions = {}
        self.path_indices = {}
        self.completed_exits = set()
        self.completed_entries = set()

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
        """Initialize and start concurrent spy missions"""
        try:
            num_mines = int(self.mine_count.get())
            max_mines = (self.kingdom.size * self.kingdom.size) // 3
            if num_mines < 0 or num_mines > max_mines:
                raise ValueError(f"Invalid number of mines. Please use between 0 and {max_mines}")

            # Initialize simulation state
            self.kingdom.initialize_mines(num_mines)
            self.update_grid()
            self.kingdom.simulation_start_time = time.time()
            
            # Reset tracking variables
            self.active_spies.clear()
            self.spy_paths.clear()
            self.spy_positions.clear()
            self.path_indices.clear()
            self.completed_exits.clear()
            self.completed_entries.clear()

            # Start Phase 1: Exit missions
            self.phase_var.set("Phase 1: Spies racing to find paths to kings")
            self.start_exit_missions()

        except ValueError as e:
            messagebox.showerror("Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def start_exit_missions(self):
        """Start all exit missions simultaneously"""
        for direction in Direction:
            # Calculate exit path
            edge_pos = self.kingdom.get_random_edge_position(direction)
            path = self.kingdom.find_path(self.kingdom.castle_pos, edge_pos)
            
            if path:
                self.spy_paths[direction] = path
                self.spy_positions[direction] = self.kingdom.castle_pos
                self.path_indices[direction] = 0
                self.active_spies.add(direction)
                
                # Start movement for this spy
                self.root.after(100, lambda d=direction: self.move_spy(d, "exit"))
            else:
                self.status_vars[direction].set(f"{direction.name}: No exit path found")

    def move_spy(self, direction: Direction, phase: str):
        """Move a spy one step along its path"""
        if not self.simulation_running or direction not in self.active_spies:
            return

        current_index = self.path_indices[direction]
        path = self.spy_paths[direction]

        if current_index < len(path):
            # Move to next position
            next_pos = path[current_index]
            self.spy_positions[direction] = next_pos
            self.path_indices[direction] = current_index + 1

            # Update display
            if phase == "exit":
                self.update_grid(next_pos, None, direction)
            else:
                self.update_grid(None, next_pos, direction)

            # Schedule next move
            self.root.after(200, lambda: self.move_spy(direction, phase))
        else:
            # Path complete
            if phase == "exit":
                self.completed_exits.add(direction)
                self.active_spies.remove(direction)
                self.status_vars[direction].set(
                    f"{direction.name}: Found exit path in {time.time() - self.kingdom.simulation_start_time:.1f}s"
                )
                
                # Check if all exit missions are complete
                if not self.active_spies:
                    self.root.after(1000, self.start_entry_phase)
            else:  # entry phase
                self.completed_entries.add(direction)
                self.active_spies.remove(direction)
                self.start_king_mission(direction)

    def start_entry_phase(self):
        """Start the entry phase for successful exits"""
        if not self.completed_exits:
            self.phase_var.set("Simulation complete - no spies found exit paths")
            messagebox.showinfo("🎮 Simulation Complete",
                              "No spies found exit paths!\nThe castle remains safe... 🏰")
            return

        # Reposition mines for entry phase
        self.phase_var.set("Repositioning mines for entry phase...")
        self.kingdom.initialize_mines(int(self.mine_count.get()))
        self.update_grid()
        
        # Clear tracking for new phase
        self.active_spies.clear()
        self.spy_paths.clear()
        self.spy_positions.clear()
        self.path_indices.clear()

        self.root.after(2000, self.start_entry_missions)

    def start_entry_missions(self):
        """Start all entry missions simultaneously"""
        self.phase_var.set("Phase 2: Spies racing to find entry paths")
        
        for direction in self.completed_exits:
            # Calculate entry path
            start_pos = self.kingdom.get_random_edge_position(direction)
            path = self.kingdom.find_path(start_pos, self.kingdom.castle_pos, is_entry=True)
            
            if path:
                self.spy_paths[direction] = path
                self.spy_positions[direction] = start_pos
                self.path_indices[direction] = 0
                self.active_spies.add(direction)
                
                # Start movement for this spy
                self.root.after(100, lambda d=direction: self.move_spy(d, "entry"))
            else:
                self.status_vars[direction].set(f"{direction.name}: No entry path found")

    def start_king_mission(self, direction: Direction):
        """Start the king's mission after spy completes entry path"""
        path = self.spy_paths[direction]
        self.kingdom.entry_paths[direction] = path
        
        if self.kingdom.king_mission(direction):
            self.status_vars[direction].set(
                f"{direction.name}: Reached castle in {self.kingdom.casualties[direction]:.1f}s"
            )
        else:
            self.status_vars[direction].set(
                f"{direction.name}: King fell in battle after {self.kingdom.casualties[direction]:.1f}s"
            )

        # Check if all missions are complete
        if not self.active_spies and len(self.completed_entries) == len(self.completed_exits):
            self.show_final_results()

    def show_final_results(self):
        """Display the final results of the simulation"""
        successful_kings = [d for d in Direction if d in self.completed_entries 
                          and self.kingdom.casualties[d] != float('inf')]
        
        if successful_kings:
            winner = min(successful_kings, key=lambda d: self.kingdom.casualties[d])
            result = f"🏆 Winner: {winner.name} 🏆\n\nDetailed Results:\n"

            for direction in Direction:
                result += f"\n{direction.name}:\n"
                if direction in self.completed_exits:
                    result += f"Exit Path Found ✅\n"
                    time_taken = self.kingdom.casualties[direction]
                    result += f"King's Journey: "
                    if time_taken != float('inf'):
                        result += f"{time_taken:.1f}s"
                        if direction in successful_kings:
                            result += " ✅"
                        else:
                            result += " ❌"
                    else:
                        result += "Failed to reach castle ❌"
                else:
                    result += "Spy failed to find exit path ❌"

            messagebox.showinfo("🎮 Simulation Complete", result)
        else:
            messagebox.showinfo("🎮 Simulation Complete",
                              "No kings reached the castle!\nThe castle remains unconquered... 👑")
    def start_simulation(self):
        """Start the simulation if not already running."""
        if not self.simulation_running:
            self.simulation_running = True
            self.simulation_thread()

    def reset_simulation(self):
        """Reset the simulation state."""
        self.simulation_running = False
        self.kingdom.reset()
        self.active_spies.clear()
        self.spy_paths.clear()
        self.spy_positions.clear()
        self.path_indices.clear()
        self.completed_exits.clear()
        self.completed_entries.clear()
        
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