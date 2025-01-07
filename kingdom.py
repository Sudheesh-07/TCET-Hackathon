import tkinter as tk
from tkinter import ttk, messagebox
import time
from kingdom_layout import Kingdom, Direction, MineMovement


class KingdomGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Kingdom Pathfinding Simulation")
        self.kingdom = Kingdom()
        self.current_spies = {d: None for d in Direction}
        self.blocked_positions = set()
        self.setup_gui()

    def setup_gui(self):
        control_frame = ttk.Frame(self.root, padding="5")
        control_frame.grid(row=0, column=0, sticky="nsew")

        ttk.Label(control_frame, text="Mines:").grid(row=0, column=0, padx=5)
        self.mine_count = tk.StringVar(value="50")
        ttk.Entry(control_frame, textvariable=self.mine_count, width=10).grid(row=0, column=1, padx=5)

        ttk.Label(control_frame, text="Movement:").grid(row=0, column=2, padx=5)
        self.movement = tk.StringVar(value="diagonal")
        ttk.OptionMenu(control_frame, self.movement, "diagonal",
                       "diagonal", "horizontal", "vertical").grid(row=0, column=3, padx=5)

        ttk.Button(control_frame, text="Start", command=self.run_simulation).grid(row=0, column=4, padx=5)
        ttk.Button(control_frame, text="Reset", command=self.reset_simulation).grid(row=0, column=5, padx=5)

        self.grid_frame = ttk.Frame(self.root, padding="5")
        self.grid_frame.grid(row=1, column=0, sticky="nsew")

        self.status_var = tk.StringVar()
        ttk.Label(self.root, textvariable=self.status_var, wraplength=400).grid(row=2, column=0)

        self.create_grid()

    def create_grid(self):
        self.buttons = [[ttk.Label(self.grid_frame, width=3, text="·", background="white", relief="solid")
                         for _ in range(self.kingdom.size)] for _ in range(self.kingdom.size)]
        for i, row in enumerate(self.buttons):
            for j, btn in enumerate(row):
                btn.grid(row=i, column=j, padx=1, pady=1)

        x, y = self.kingdom.castle_pos
        self.buttons[x][y].configure(text="🏰", background="yellow")

    def update_grid(self, spy_pos=None, path=None, king_pos=None, direction=None):
        for i, row in enumerate(self.buttons):
            for j, btn in enumerate(row):
                pos = (i, j)
                if spy_pos and pos == spy_pos:
                    btn.configure(text="🕵️", background=direction.get_color())
                elif king_pos and pos == king_pos:
                    btn.configure(text="👑", background=direction.get_color())
                elif path and pos in path:
                    btn.configure(text="◼️", background=direction.get_color())
                elif pos == self.kingdom.castle_pos:
                    btn.configure(text="🏰", background="yellow")
                elif pos in self.kingdom.mines:
                    btn.configure(text="💣", background="red")
                else:
                    btn.configure(text=str(self.kingdom.grid[i, j]) if self.kingdom.grid[i, j] > 0 else "·",
                                  background="white")
        self.root.update()

    def move_spy(self, start_pos, path, direction):
        if not path:
            return False

        for i, pos in enumerate(path):
            self.status_var.set(f"Spy {direction.name} moving...")
            self.update_grid(spy_pos=pos, path=path[:i], direction=direction)
            time.sleep(0.3)

            if pos in self.kingdom.mines:
                self.blocked_positions.add(pos)
                messagebox.showinfo("Hit", f"{direction.name} spy hit a mine! Finding new path from castle...")
                new_path = self.kingdom.find_path(self.kingdom.castle_pos, start_pos, self.blocked_positions)
                if new_path:
                    return self.move_spy(start_pos, new_path, direction)
                return False
        return True

    def move_king(self, start_pos, path, direction):
        if not path:
            return False

        for i, pos in enumerate(path):
            self.status_var.set(f"King {direction.name} moving...")
            self.update_grid(king_pos=pos, path=path[:i], direction=direction)
            time.sleep(0.3)

            if pos in self.kingdom.mines:
                self.blocked_positions.add(pos)
                messagebox.showinfo("Hit", f"{direction.name} king's army hit a mine!")
                self.kingdom.casualties[direction] += 1
                continue
        return True

    def run_simulation(self):
        try:
            self.kingdom.initialize_mines(int(self.mine_count.get()))
            self.blocked_positions.clear()
            self.update_grid()

            movement = {
                'diagonal': MineMovement.DIAGONAL,
                'horizontal': MineMovement.HORIZONTAL,
                'vertical': MineMovement.VERTICAL
            }[self.movement.get()]

            # Spy phase
            for direction in Direction:
                kingdom_pos = self.kingdom.get_kingdom_position(direction)
                for _ in range(3):
                    path = self.kingdom.find_path(self.kingdom.castle_pos, kingdom_pos, self.blocked_positions)
                    if self.move_spy(kingdom_pos, path, direction):
                        break
                else:
                    messagebox.showinfo("Failed", f"{direction.name} spy failed all attempts!")

            # Move mines and start king phase
            self.kingdom.move_mines(movement)
            self.blocked_positions.clear()
            self.update_grid()

            for direction in Direction:
                kingdom_pos = self.kingdom.get_kingdom_position(direction)
                path = self.kingdom.find_path(kingdom_pos, self.kingdom.castle_pos, self.blocked_positions)
                self.move_king(kingdom_pos, path, direction)

            winner = min(self.kingdom.casualties.items(), key=lambda x: x[1])[0]
            result = f"Winner: {winner.name}\n\nCasualties:\n"
            result += "\n".join(f"{d.name}: {c}" for d, c in self.kingdom.casualties.items())
            messagebox.showinfo("Results", result)

        except ValueError:
            messagebox.showerror("Error", "Invalid mine count")

    def reset_simulation(self):
        self.kingdom = Kingdom()
        self.blocked_positions.clear()
        self.update_grid()
        self.status_var.set("")
