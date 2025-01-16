# kingdom_core.py

from typing import Dict, List, Set, Tuple
from enum import Enum
import numpy as np
from heapq import heappush, heappop

import threading

import concurrent.futures
from colorama import Fore, init

# Initialize colorama
init()


class Direction(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

    def get_color(self) -> str:
        return {
            Direction.NORTH: Fore.CYAN,
            Direction.SOUTH: Fore.GREEN,
            Direction.EAST: Fore.MAGENTA,
            Direction.WEST: Fore.RED
        }[self]


class Kingdom:
    while True:
        grid = int(input(f"Enter number of grid (minimum 7X7): "))
        if grid >= 7:
            break
        else:
            print(Fore.RED, "Grid size should be at least 7. Please try again.", Fore.RESET)

    print(Fore.GREEN, "Initializing Kingdom...", Fore.RESET)

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
        self.casualties = {d: 0 for d in Direction}
        self.print_lock = threading.Lock()
        self.path_lock = threading.Lock()
        self.grid_state = {}
        self.mission_events = {d: threading.Event() for d in Direction}

    def print_grid(self, positions=None):
        """Thread-safe grid printing with multiple positions"""
        with self.print_lock:
            print("\n" + "=" * (self.size * 4 + 1))
            for i in range(-1, self.size + 1):
                for j in range(-1, self.size + 1):
                    is_main_grid = 0 <= i < self.size and 0 <= j < self.size
                    pos = (i, j)

                    if positions and pos in positions:
                        direction = positions[pos]
                        char = 'S' if isinstance(direction, tuple) else 'K'
                        color = direction[0].get_color() if isinstance(direction, tuple) else direction.get_color()
                        print(f"{color}{char}{Fore.RESET}", end=" | ")
                    elif pos == self.castle_pos:
                        print(f"{Fore.YELLOW}🏰{Fore.RESET}", end=" | ")
                    elif is_main_grid and pos in self.mines:
                        print(f"{Fore.RED}💣{Fore.RESET}", end=" | ")
                    elif is_main_grid:
                        val = self.grid[i, j]
                        print(str(val) if val > 0 else "·", end=" | ")
                    else:
                        print(" ", end=" | ")
                print("\n" + "=" * (self.size * 4 + 1))

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

    def initialize_mines(self, num_mines: int) -> None:
        """Initialize mines with random distribution across the grid."""
        self.mines.clear()
        self.edge_entry_points.clear()
        self.king_positions.clear()
        self.grid_state.clear()

        # Set random king positions
        for direction in Direction:
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
        """Find path using A* algorithm."""
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

    def spy_exit_mission_thread(self, direction: Direction, results: Dict):
        """Threaded version of spy exit mission"""
        print(f"\n{direction.get_color()}Spy starting exit mission for {direction.name}...{Fore.RESET}")

        start_time = time.time()
        edge_pos = self.get_random_edge_position(direction)

        path = self.find_path(self.castle_pos, edge_pos)
        if path:
            for pos in path:
                time.sleep(0.2)
                if pos in self.mines:
                    print(f"{direction.get_color()}Spy {direction.name} hit mine during exit!{Fore.RESET}")
                    results[direction] = (False, time.time() - start_time)
                    return

                with self.path_lock:
                    self.grid_state[pos] = (direction, 'spy')
                    self.print_grid(self.grid_state)

            with self.path_lock:
                self.exit_paths[direction] = path
            results[direction] = (True, time.time() - start_time)
            return

        print(f"{direction.get_color()}No exit path found for {direction.name}!{Fore.RESET}")
        results[direction] = (False, 0)

    def spy_entry_mission_thread(self, direction: Direction, results: Dict):
        """Threaded version of spy entry mission"""
        print(f"\n{direction.get_color()}Spy starting entry mission for {direction.name}...{Fore.RESET}")

        while True:
            max_attempts = int(input(f"Enter current {direction.name} king army count (1-10): "))
            if 1 <= max_attempts <= 10:
                break
            print(f"{Fore.RED}Army size must be between 1 and 10{Fore.RESET}")

        for attempt in range(max_attempts):
            start_pos = self.get_random_edge_position(direction)
            path = self.find_path(start_pos, self.castle_pos, is_entry=True)

            if path:
                path_safe = True
                for pos in path:
                    time.sleep(0.2)
                    if pos in self.mines:
                        print(f"{direction.get_color()}Spy {direction.name} found unsafe path{Fore.RESET}")
                        path_safe = False
                        break

                    with self.path_lock:
                        self.grid_state[pos] = (direction, 'spy')
                        self.print_grid(self.grid_state)

                if path_safe:
                    with self.path_lock:
                        self.entry_paths[direction] = path
                    print(f"{direction.get_color()}Spy {direction.name} found entry path!{Fore.RESET}")
                    results[direction] = True
                    return

            if attempt < max_attempts - 1:
                print(f"{direction.get_color()}Attempt {attempt + 1} failed for {direction.name}{Fore.RESET}")
                time.sleep(0.5)

        print(f"{direction.get_color()}All entry attempts failed for {direction.name}!{Fore.RESET}")
        results[direction] = False

    def king_mission_thread(self, direction: Direction, results: Dict):
        """Threaded version of king mission"""
        if direction not in self.entry_paths:
            results[direction] = False
            return

        path = self.entry_paths[direction]
        step_time = 0.2
        total_time = 0

        print(f"\n{direction.get_color()}King {direction.name} following entry path...{Fore.RESET}")

        for pos in path:
            total_time += step_time
            time.sleep(step_time)

            if np.random.random() < 0.2:
                new_casualties = np.random.randint(1, 4)
                with self.path_lock:
                    self.casualties[direction] += new_casualties
                print(f"{direction.get_color()}King {direction.name}: {new_casualties} casualties. "
                      f"Total: {self.casualties[direction]}{Fore.RESET}")

            with self.path_lock:
                self.grid_state[pos] = direction
                self.print_grid(self.grid_state)

        self.time_taken[direction] = total_time
        results[direction] = True


# kingdom_main.py


import concurrent.futures
import time


def print_spy_path_report(kingdom: Kingdom):
    """Print detailed spy path report."""

    def format_path(path: List[Tuple[int, int]]) -> str:
        if not path:
            return "No path found"
        return " → ".join([f"({x},{y})" for x, y in path])

    print("\n=== Spy Mission Report ===\n")

    for direction in Direction:
        print(f"\n{direction.get_color()}{direction.name} SPY MISSIONS:{Fore.RESET}")
        print("-" * 50)

        print("EXIT PATH (Castle → King):")
        if direction in kingdom.exit_paths:
            exit_path = kingdom.exit_paths[direction]
            print(format_path(exit_path))
            print(f"Total steps: {len(exit_path)}")
        else:
            print("No exit path found")

        print("\nENTRY PATH (King → Castle):")
        if direction in kingdom.entry_paths:
            entry_path = kingdom.entry_paths[direction]
            print(format_path(entry_path))
            print(f"Total steps: {len(entry_path)}")
        else:
            print("No entry path found")

        print("-" * 50)


def main():
    """Main function with multi-threading implementation"""
    try:
        kingdom = Kingdom()

        while True:
            try:
                print("\nInitializing Kingdom Simulation...")
                max_mines = (kingdom.size * kingdom.size) // 3
                min_mines = max_mines // 2

                while True:
                    try:
                        num_mines = int(input(f"Enter number of mines ({min_mines}-{max_mines}): "))
                        if min_mines <= num_mines <= max_mines:
                            break
                        print(f"{Fore.RED}Please enter a number between {min_mines} and {max_mines}{Fore.RESET}")
                    except ValueError:
                        print(f"{Fore.RED}Please enter a valid number{Fore.RESET}")

                kingdom.initialize_mines(num_mines)
                kingdom.print_grid()

                # Phase 1: Parallel Spy Exit Missions
                print(f"\n{Fore.CYAN}Phase 1: Spies finding paths to kings{Fore.RESET}")
                exit_results = {}
                threads = []

                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {executor.submit(kingdom.spy_exit_mission_thread, direction, exit_results): direction for
                               direction in Direction}
                    concurrent.futures.wait(futures)

                successful_exits = {direction for direction, (success, _) in exit_results.items() if success}

                if not successful_exits:
                    print(f"\n{Fore.GREEN}Simulation complete - no spies found exit paths!{Fore.RESET}")
                    print(f"{Fore.YELLOW}The castle remains safe... 🏰{Fore.RESET}")
                    break

                print(f"\n{Fore.CYAN}Repositioning mines for entry phase...{Fore.RESET}")
                kingdom.initialize_mines(num_mines)
                kingdom.print_grid()
                time.sleep(2)

                # Phase 2: Parallel Entry Missions
                print(f"\n{Fore.CYAN}Phase 2: Parallel spy entry missions{Fore.RESET}")
                entry_results = {}

                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {executor.submit(kingdom.spy_entry_mission_thread, direction, entry_results): direction
                               for direction in successful_exits}
                    concurrent.futures.wait(futures)

                successful_entries = {direction for direction, success in entry_results.items() if success}

                if not successful_entries:
                    print(f"\n{Fore.GREEN}No spies found safe entry paths!{Fore.RESET}")
                    print(f"{Fore.YELLOW}The castle remains secure... 🏰{Fore.RESET}")
                    break

                # Phase 3: Parallel King Missions
                print(f"\n{Fore.CYAN}Phase 3: Kings advancing simultaneously{Fore.RESET}")
                king_results = {}

                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {executor.submit(kingdom.king_mission_thread, direction, king_results): direction
                               for direction in successful_entries}
                    concurrent.futures.wait(futures)

                successful_kings = {direction for direction, success in king_results.items() if success}

                # Print final report
                print_spy_path_report(kingdom)

                # Determine and announce winner
                if successful_kings:
                    def get_score(direction):
                        time_score = kingdom.time_taken[direction] / max(kingdom.time_taken.values())
                        casualty_score = kingdom.casualties[direction] / max(kingdom.casualties.values())
                        return (time_score * 0.2) + (casualty_score * 0.8)

                    winner = min(successful_kings, key=get_score)
                    print(f"\n{Fore.YELLOW}🏆 Winner: {winner.name} 🏆{Fore.RESET}")
                    print(f"Time: {kingdom.time_taken[winner]:.1f}s")
                    print(f"Casualties: {kingdom.casualties[winner]}")

                    # Print final statistics for all successful kings
                    print("\nFinal Statistics for All Successful Kings:")
                    for direction in successful_kings:
                        print(f"\n{direction.get_color()}{direction.name}:{Fore.RESET}")
                        print(f"  Time: {kingdom.time_taken[direction]:.1f}s")
                        print(f"  Casualties: {kingdom.casualties[direction]}")
                        # print(f"  Score: {get_score(direction):.3f}")
                else:
                    print(f"\n{Fore.GREEN}No kings reached the castle!{Fore.RESET}")
                    print(f"{Fore.YELLOW}The castle remains unconquered... 👑{Fore.RESET}")

                # Ask if user wants to play again
                while True:
                    play_again = input("\nWould you like to run another simulation? (yes/no): ").lower()
                    if play_again in ['yes', 'no']:
                        break
                    print(f"{Fore.RED}Please enter 'yes' or 'no'{Fore.RESET}")

                if play_again == 'no':
                    print(f"\n{Fore.GREEN}Thank you for playing the Kingdom Simulation!{Fore.RESET}")
                    break
                else:
                    # Reset kingdom state for new game
                    kingdom.mines.clear()
                    kingdom.exit_paths.clear()
                    kingdom.entry_paths.clear()
                    kingdom.time_taken = {d: float('inf') for d in Direction}
                    kingdom.casualties = {d: 0 for d in Direction}
                    kingdom.grid_state.clear()
                    continue

            except ValueError as e:
                print(f"{Fore.RED}Error: {str(e)}{Fore.RESET}")
                continue
            except Exception as e:
                print(f"{Fore.RED}An unexpected error occurred: {str(e)}{Fore.RESET}")
                break

    except Exception as e:
        print(f"{Fore.RED}Fatal error: {str(e)}{Fore.RESET}")


if __name__ == "__main__":
    main()