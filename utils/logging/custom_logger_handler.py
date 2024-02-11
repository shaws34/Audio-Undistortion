import logging
import traceback
from os import path
from config import LOG_FILE_FOLDER_PATH, LOG_LEVEL

# Create a custom logger class that inherits from the built-in Logger class and overrides the error method to include a stack trace by default
class CustomLoggerHandler(logging.FileHandler):
    def __init__(self, filename="logs.log"):
        self.log_file_path = path.normpath(f"{LOG_FILE_FOLDER_PATH}/{filename}")
        super().__init__(self.log_file_path)

        # self.setFormatter(logging.Formatter('%(asctime)s [%(name)s] [%(pathname)s (Thread:%(thread)d %(threadName)s)] - [%(levelname)s]\t%(message)s'))
        self.setFormatter(logging.Formatter('%(asctime)s [%(name)s] [%(levelname)s]\t%(message)s'))
        self.setLevel(LOG_LEVEL)
        self.last_message = None
        

    def emit(self, record):
        """
        Log 'msg % args' with severity 'ERROR', appending a stack trace.
        """
        
        if record.levelno >= logging.ERROR:
            record.stack_info = '\tStack Trace:\n' + '\t\t'.join(traceback.format_stack())
        self.last_message = self.format(record) + self.terminator

        super().emit(record)

