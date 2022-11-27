from autosktime.data.benchmark.cmapss import CMAPSSBenchmark, CMAPSS2Benchmark, CMAPSS1Benchmark, CMAPSS3Benchmark, \
    CMAPSS4Benchmark
from autosktime.data.benchmark.rul import RULBenchmark

BENCHMARKS = {
    CMAPSSBenchmark.name(): CMAPSS1Benchmark,
    CMAPSS1Benchmark.name(): CMAPSS1Benchmark,
    CMAPSS2Benchmark.name(): CMAPSS2Benchmark,
    CMAPSS3Benchmark.name(): CMAPSS3Benchmark,
    CMAPSS4Benchmark.name(): CMAPSS4Benchmark,
    RULBenchmark.name(): RULBenchmark
}
