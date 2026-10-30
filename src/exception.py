"""Project exception that includes the source file and line number."""
import sys

class CustomException(Exception):
    def __init__(self, message: str, error: Exception | None = None):
        _, _, tb = sys.exc_info()
        if error is not None and error.__traceback__ is not None:
            tb = error.__traceback__
        if tb is not None:
            while tb.tb_next is not None:
                tb = tb.tb_next
            location = f"file={tb.tb_frame.f_code.co_filename}, line={tb.tb_lineno}"
        else:
            location = "file=unknown, line=unknown"
        super().__init__(f"{message} ({location})")
