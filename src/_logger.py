"""
TITLE OF PAPAER
-------------------------------------------
Authors:        Shaimaa El-Baklish
Organization:   ETH Zürich, Switzerland, IVT - Institute for Transportation Planning and Systems
Development:    2025
Submitted to:   JOURNAL
-------------------------------------------
"""

# #############################################################################
# IMPORTS
# #############################################################################
import os
import sys
import logging


# #############################################################################
# Class: Logger
# #############################################################################
class Logger:
    def __init__(self, date, intersection, code, timeslot, session, log_dir="../logs", 
                 console_level=logging.INFO, file_level=logging.DEBUG):
        """
        Args:
            date:            e.g. '2025-06-16'
            intersection:    e.g. 'D3'
            code:       e.g. 'E'
            timeslot:        e.g. 'AM1'
            session:         e.g. 'coordinate_transform'       
            log_dir:         directory to write log files
            console_level:   minimum level printed to console
            file_level:      minimum level written to file
        """
        os.makedirs(log_dir, exist_ok=True)

        self.log_path = os.path.join(log_dir, f"{date}_{intersection}_{code}_{timeslot}_{session}.log")
        self._logger = logging.getLogger(f"{date}_{intersection}_{code}_{timeslot}_{session}")
        self._logger.setLevel(logging.DEBUG)

        if self._logger.handlers:
            self._logger.handlers.clear()

        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        fh = logging.FileHandler(self.log_path, mode='w')
        fh.setLevel(file_level)
        fh.setFormatter(formatter)

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(console_level)
        ch.setFormatter(formatter)

        self._logger.addHandler(fh)
        self._logger.addHandler(ch)

        self.info(f"Log file: {self.log_path}")

    def debug(self, msg):   self._logger.debug(msg)
    def info(self, msg):    self._logger.info(msg)
    def warning(self, msg): self._logger.warning(msg)
    def error(self, msg):   self._logger.error(msg, exc_info=True)

    def section(self, title):
        """Print a visual section divider."""
        self._logger.info(f"{'─' * 10} {title} {'─' * 10}")