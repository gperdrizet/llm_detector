'''Main function to run jobs based on command line arguments'''

import benchmarking.functions.helper as helper_funcs
import benchmarking.functions.runner as runner_funcs
import benchmarking.functions.perplexity_ratio as perplexity_funcs

if __name__ == "__main__":

    logger = helper_funcs.start_logger(
        logfile_name = 'main.log',
        logger_name = 'main'
    )

    # Parse command line arguments
    args = helper_funcs.parse_args()

    # Run binoculars
    if args.perplexity_ratio != 'False':

        perplexity_funcs.perplexity_ratio_score()

    # Run benchmark
    if args.run_benchmark is not None:

        # Loop on the user specified benchmarks
        for benchmark in args.run_benchmark:

            # Parse arguments for this benchmark
            config_file = benchmark[0]
            resume = benchmark[1]

            if resume == 'True' or resume == 'true':
                RESUME = True

            else:
                RESUME = False

            logger.info('Running benchmark from configuration: %s', config_file)

            # Run the benchmark
            runner_funcs.run(
                experiment_config_file = config_file,
                resume = RESUME
            )
