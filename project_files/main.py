# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 13:16:55 2024

@author: Dr Oriol Sans Planell
"""

import sys
from PyQt5.QtWidgets import QApplication
from main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()