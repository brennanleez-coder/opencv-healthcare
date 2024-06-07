import logging
import os
import tempfile
from datetime import datetime
import shutil

class SingletonMeta(type):
    """
    A thread-safe implementation of Singleton for the Logger class.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class Logger(metaclass=SingletonMeta):
    def __init__(self):
        base_dir = os.path.dirname(__file__)
        logs_path = os.path.join(base_dir, '../..', 'logs')
        self.log_file_path = os.path.join(logs_path, 'app.log')

        # Ensure log directory exists
        os.makedirs(logs_path, exist_ok=True)
        self._logger = logging.getLogger("VideoProcessingAPI")
        self._logger.setLevel(logging.INFO)

        # Check if logger already has handlers to avoid duplicate entries
        if not self._logger.handlers:
            # Create file handler
            file_handler = logging.FileHandler(self.log_file_path)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)

            # Add file handler to logger
            self._logger.addHandler(file_handler)

    def get_logger(self):
        """
        Returns the logger instance.
        """
        return self._logger

    def shutdown(self):
            """
            Handles cleanup and archiving of the log file upon application shutdown.
            """
            logs_dir = os.path.dirname(self.log_file_path)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            archived_log_filename = f"app_{timestamp}.log"
            archived_log_path = os.path.join(logs_dir, archived_log_filename)

            # Rename the current log file to archive it with a timestamp
            shutil.move(self.log_file_path, archived_log_path)
            print(f"Log file moved to {archived_log_path}")