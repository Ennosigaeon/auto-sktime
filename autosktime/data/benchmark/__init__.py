from autosktime.data.benchmark.cmapss import CMAPSSBenchmark, CMAPSS2Benchmark, CMAPSS1Benchmark, CMAPSS3Benchmark, \
    CMAPSS4Benchmark
from autosktime.data.benchmark.femto_bearing import FemtoBenchmark
from autosktime.data.benchmark.phm08 import PHM08Benchmark
from autosktime.data.benchmark.phme20 import PHME20Benchmark
from autosktime.data.benchmark.filtration import FiltrationBenchmark

BENCHMARKS = {
    CMAPSS1Benchmark.name(): CMAPSS1Benchmark,
    CMAPSS2Benchmark.name(): CMAPSS2Benchmark,
    CMAPSS3Benchmark.name(): CMAPSS3Benchmark,
    CMAPSS4Benchmark.name(): CMAPSS4Benchmark,
    FemtoBenchmark.name(): FemtoBenchmark,
    FiltrationBenchmark.name(): FiltrationBenchmark,
    PHM08Benchmark.name(): PHM08Benchmark,
    PHME20Benchmark.name(): PHME20Benchmark
}
