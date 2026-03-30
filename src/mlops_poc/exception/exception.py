"""Custom exception that captures file + line number automatically."""
import sys


class MLOpsException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(str(error_message))
        self.error_message = self._format(str(error_message), error_detail)

    @staticmethod
    def _format(message: str, error_detail: sys) -> str:
        _, _, tb = error_detail.exc_info()
        if tb is not None:
            file_name = tb.tb_frame.f_code.co_filename
            line_no = tb.tb_lineno
            return f"Error in [{file_name}] at line [{line_no}]: {message}"
        return message

    def __str__(self) -> str:
        return self.error_message
