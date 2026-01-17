"""
Real-time Audio Classifier with GUI
Entry point of the application
"""

from gui.app import AudioClassifierGUI
import tkinter as tk


def main():
    root = tk.Tk()
    app = AudioClassifierGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()