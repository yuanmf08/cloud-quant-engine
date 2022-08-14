from src.utils.logger import logger
from fastapi import HTTPException
from functools import wraps


def handle_exceptions(f):
    @wraps(f)
    def wrapper(*args, **kw):
        try:
            return f(*args, **kw)
        except Exception as e:
            logger.exception(e)
            raise HTTPException(status_code=500, detail=str(e))
    wrapper.__name__ = f.__name__
    wrapper.__doc__ = f.__doc__
    return wrapper
