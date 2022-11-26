from autosktime.data.benchmark.cmapss import CMAPSSBenchmark
from autosktime.data.benchmark.rul import RULBenchmark

BENCHMARKS = {
    CMAPSSBenchmark.name(): CMAPSSBenchmark(number=1),
    CMAPSSBenchmark.name() + "1": CMAPSSBenchmark(number=1),
    CMAPSSBenchmark.name() + "2": CMAPSSBenchmark(number=2),
    CMAPSSBenchmark.name() + "3": CMAPSSBenchmark(number=3),
    CMAPSSBenchmark.name() + "4": CMAPSSBenchmark(number=4),
    RULBenchmark.name(): RULBenchmark()
}
