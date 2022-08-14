import os
import pickle

import pandas as pd
import redis

from src.utils.logger import logger


class RedisCache:
    def __init__(self, ttl=24 * 3600):
        self._ttl = ttl
        ssl = os.environ.get("REDIS_SSL") == 'true'

        self._redis_db = redis.StrictRedis(
            host=os.environ.get("REDIS_HOST"),
            port=os.environ.get("REDIS_PORT"),
            ssl=ssl,
            password=os.environ.get("REDIS_TOKEN"),
        )

        try:
            if self._redis_db.ping():
                logger.info('Redis Connected')
                self._connection = True
        except Exception:
            self._connection = False
            logger.warning('Redis server failed')

    @staticmethod
    def _key_prefix(key):
        return str(os.environ.get('ENV_NAME')) + key

    def set(self, key, value, ttl=None):
        key = self._key_prefix(key)

        if not self._connection:
            return

        if ttl == -1:
            self._redis_db.set(key, value)
        elif ttl is None:
            self._redis_db.set(key, value, ex=self._ttl)
        else:
            self._redis_db.set(key, value, ex=ttl)

    def set_data_frame(self, key, df: pd.DataFrame, ttl=None):
        key = self._key_prefix(key)

        self.set(
            key=key,
            value=pickle.dumps(df),
            ttl=ttl,
        )

    def get(self, key):
        key = self._key_prefix(key)

        if not self._connection:
            return
        return self._redis_db.get(key)

    def get_data_frame(self, key):
        key = self._key_prefix(key)

        if not self._connection:
            return
        data = self.get(key)
        if data:
            return pickle.loads(data)


redis_db = RedisCache()
