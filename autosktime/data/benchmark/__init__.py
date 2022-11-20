from autosktime.data.benchmark.cmapss import CMAPSSBenchmark
from autosktime.data.benchmark.rul import RULBenchmark

BENCHMARKS = {
    CMAPSSBenchmark.name(): CMAPSSBenchmark,
    RULBenchmark.name(): RULBenchmark
}
