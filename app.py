import tkinter as tk

from kingdom import KingdomGUI


def main():
    root = tk.Tk()
    app = KingdomGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()