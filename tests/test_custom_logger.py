# test_custom_logger.py
import unittest
import logging
from utils.logging.custom_logger_handler import CustomLoggerHandler
from io import StringIO


class TestCustomLoggerHandler(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("TestLogger")
        self.logger.setLevel(logging.DEBUG)

        self.handler = CustomLoggerHandler(filename="unittests.log")
        self.logger.addHandler(self.handler)

    def tearDown(self):
        self.logger.removeHandler(self.handler)
        self.handler.close()

    def test_exceptions_logs_include_stack_trace(self):
        log_message = "This is an exception message"
        try:
            raise ValueError("Test exception")
        except ValueError:
            self.logger.exception(log_message)
        log_contents = self.handler.last_message
        self.assertIn("Stack Trace", log_contents)
        self.assertIn("[ERROR]", log_contents)
        self.assertIn("[TestLogger]", log_contents)
        self.assertIn("ValueError: Test exception", log_contents)
        self.assertIn(log_message, log_contents)

    def test_error_logs_include_stack_trace(self):
        log_message = "This is an error message"
        try:
            raise ValueError("Test exception")
        except ValueError:
            None
        self.logger.error(log_message)
        log_contents = self.handler.last_message
        self.assertIn("Stack Trace", log_contents)
        self.assertIn("[ERROR]", log_contents)
        self.assertIn("[TestLogger]", log_contents)
        self.assertNotIn("ValueError: Test exception", log_contents)
        self.assertIn(log_message, log_contents)

    def test_info_logs_do_not_include_stack_trace(self):
        log_message = "This is an info message"
        self.logger.info(log_message)
        log_contents = self.handler.last_message
        self.assertNotIn("Stack Trace", log_contents)
        self.assertNotIn("[ERROR]", log_contents)
        self.assertIn("[TestLogger]", log_contents)
        self.assertIn(log_message, log_contents)

if __name__ == '__main__':
    unittest.main()
