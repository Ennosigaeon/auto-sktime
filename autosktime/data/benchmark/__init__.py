from autosktime.data.benchmark.cmapss import CMAPSSBenchmark, CMAPSS2Benchmark, CMAPSS1Benchmark, CMAPSS3Benchmark, \
    CMAPSS4Benchmark
from autosktime.data.benchmark.phme20 import PHME20Benchmark
from autosktime.data.benchmark.ppm import PPMBenchmark

BENCHMARKS = {
    CMAPSSBenchmark.name(): CMAPSS1Benchmark,
    CMAPSS1Benchmark.name(): CMAPSS1Benchmark,
    CMAPSS2Benchmark.name(): CMAPSS2Benchmark,
    CMAPSS3Benchmark.name(): CMAPSS3Benchmark,
    CMAPSS4Benchmark.name(): CMAPSS4Benchmark,
    PPMBenchmark.name(): PPMBenchmark,
    PHME20Benchmark.name(): PHME20Benchmark
}
