import time
from collections import OrderedDict
from dataclasses import dataclass

import pandas as pd


@dataclass
class TimingTask:
    name: str
    start: float = time.time()
    end: float = 0.0

    @property
    def wall_dur(self) -> float:
        return self.end - self.start


class StopWatch:

    def __init__(self) -> None:
        self._tasks = OrderedDict()
        self._tasks['stopwatch_time'] = TimingTask('stopwatch_time')

    def start_task(self, name: str) -> None:
        if name not in self._tasks:
            self._tasks[name] = TimingTask(name)

    def wall_elapsed(self, name: str) -> float:
        if name in self._tasks:
            if self._tasks[name].wall_dur <= 0:
                return time.time() - self._tasks[name].start
            else:
                return self._tasks[name].wall_dur
        return 0.0

    def stop_task(self, name: str) -> None:
        self._tasks[name].end = time.time()

    def __repr__(self) -> str:
        df = pd.DataFrame(self._tasks)
        return str(df)
