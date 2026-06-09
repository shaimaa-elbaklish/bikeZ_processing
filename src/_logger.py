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
import unicodedata

from tqdm import tqdm

# #############################################################################
# Class: Logger
# #############################################################################
# ---------------------------
# Helper: sanitize to ASCII
# ---------------------------
def to_ascii_safe(text: str) -> str:
    """
    Convert Unicode text to ASCII-safe version for console output.
    """
    replacements = {
        "²": "^2",
        "³": "^3",
        "ω": "omega",
        "θ": "theta",
        "Δ": "Delta",
        "─": "-",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    # fallback: strip remaining unicode
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


# ---------------------------
# TQDM-safe console handler
# ---------------------------
class TqdmConsoleHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            msg = to_ascii_safe(msg)  # sanitize
            tqdm.write(msg)
        except Exception:
            self.handleError(record)
            
# ---------------------------
# Main Logger
# ---------------------------
class Logger:
    def __init__(self, date, intersection, code, timeslot, session, log_dir="../logs", 
                 console_level=logging.WARNING, file_level=logging.DEBUG):
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

        # ---------------------------
        # File handler (UTF-8, full fidelity)
        # ---------------------------
        fh = logging.FileHandler(self.log_path, mode='w', encoding='utf-8')
        fh.setLevel(file_level)
        fh.setFormatter(formatter)

        # ---------------------------
        # Console handler (ASCII safe + tqdm-safe)
        # ---------------------------
        if console_level is not None:
            ch = TqdmConsoleHandler()
            ch.setLevel(console_level)
            ch.setFormatter(formatter)
            self._logger.addHandler(ch)

        self._logger.addHandler(fh)

        self.info(f"Log file: {self.log_path}")

    def debug(self, msg):   self._logger.debug(msg)
    def info(self, msg):    self._logger.info(msg)
    def warning(self, msg): self._logger.warning(msg)
    def error(self, msg):   self._logger.error(msg, exc_info=True)

    def section(self, title):
        """Print a visual section divider."""
        self._logger.info(f"{'─' * 10} {title} {'─' * 10}")