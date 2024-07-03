'''Main function to run jobs based on command line arguments'''

import benchmarking.functions.helper as helper_funcs
import benchmarking.functions.benchmark as benchmark_funcs
import benchmarking.functions.binoculars as binocular_funcs

if __name__ == "__main__":

    # Parse command line arguments
    args=helper_funcs.parse_args()

    # Run binoculars
    if args.binoculars != 'False':

        binocular_funcs.binoculars()

    # Run benchmark
    if args.run_benchmark is not None:

        # Loop on the user specified benchmarks
        for benchmark in args.run_benchmark:

            # Parse arguments for this benchmark
            config_file = benchmark[0]
            resume = benchmark[1]

            # Run the benchmark
            benchmark_funcs.run(
                resume = resume,
                experiment_config_file = config_file
            )
