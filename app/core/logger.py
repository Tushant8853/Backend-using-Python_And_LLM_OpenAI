# app/core/logger.py
import logging
import sys

class Logger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._configure_logger()
        return cls._instance

    def _configure_logger(self):
        self.logger = logging.getLogger("app_logger")
        self.logger.setLevel(logging.DEBUG)  # Set to desired logging level

        # Remove all handlers if any exist
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
                '%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger

# Create a global logger instance
logger = Logger().get_logger()

# import logging
# import sys
# from logging.handlers import RotatingFileHandler
# from pathlib import Path
# from .config import settings

# # Create logs directory if it doesn't exist
# log_dir = Path("logs")
# log_dir.mkdir(exist_ok=True)

# # Configure logger
# class Logger:
#     _instance = None
    
#     def __new__(cls):
#         if cls._instance is None:
#             cls._instance = super(Logger, cls).__new__(cls)
#             cls._instance._configure_logger()
#         return cls._instance
    
#     def _configure_logger(self):
#         self.logger = logging.getLogger("app_logger")
#         self.logger.setLevel(getattr(logging, settings.LOG_LEVEL))
        
#         # Remove all handlers if any exist
#         if self.logger.handlers:
#             self.logger.handlers.clear()
        
#         # Console handler
#         console_handler = logging.StreamHandler(sys.stdout)
#         console_formatter = logging.Formatter(
#             '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
#         )
#         console_handler.setFormatter(console_formatter)
#         self.logger.addHandler(console_handler)
        
#         # File handler (if LOG_FILE is specified)
#         if settings.LOG_FILE:
#             file_path = log_dir / settings.LOG_FILE
#             file_handler = RotatingFileHandler(
#                 file_path, maxBytes=10485760, backupCount=5
#             )
#             file_formatter = logging.Formatter(
#                 '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#             )
#             file_handler.setFormatter(file_formatter)
#             self.logger.addHandler(file_handler)

#     def get_logger(self):
#         return self.logger

# # Create a global logger instance
# logger = Logger().get_logger()
