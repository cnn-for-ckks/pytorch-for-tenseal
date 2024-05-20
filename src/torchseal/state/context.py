from typing import Any, Dict, Optional
from threading import Lock

import tenseal as ts


# Source: https://refactoring.guru/design-patterns/singleton/python/example#example-1
# More about metaclasses: https://realpython.com/python-metaclasses/
class CKKSStateMeta(type):
    _instances: Dict["CKKSStateMeta", "CKKSState"] = {}
    _lock: Lock = Lock()

    def __call__(self, *args: Any, **kwargs: Any) -> "CKKSState":
        # Now, imagine that the program has just been launched. Since there's no
        # Singleton instance yet, multiple threads can simultaneously pass the
        # previous conditional and reach this point almost at the same time. The
        # first of them will acquire lock and will proceed further, while the
        # rest will wait here.

        with self._lock:
            # The first thread to acquire the lock, reaches this conditional,
            # goes inside and creates the Singleton instance. Once it leaves the
            # lock block, a thread that might have been waiting for the lock
            # release may then enter this section. But since the Singleton field
            # is already initialized, the thread won't create a new object.

            if self not in self._instances:
                instance = super().__call__(*args, **kwargs)
                self._instances[self] = instance

        return self._instances[self]


# Source: https://refactoring.guru/design-patterns/singleton/python/example#example-1
class CKKSState(metaclass=CKKSStateMeta):
    __context: Optional[ts.Context] = None

    @property
    def context(self) -> ts.Context:
        assert self.__context is not None, "Context is not set"

        return self.__context

    @context.setter
    def context(self, context: ts.Context) -> None:
        self.__context = context
