'''Main function to run jobs based on command line arguments'''

import perplexity_ratio_score.functions.helper as helper_funcs
import perplexity_ratio_score.functions.runner as runner_funcs
import perplexity_ratio_score.functions.data_acquisition as data_funcs
import perplexity_ratio_score.functions.perplexity_ratio as perplexity_funcs
import perplexity_ratio_score.functions.perplexity_ratio_v2 as perplexity_funcs_v2

if __name__ == "__main__":

    logger=helper_funcs.start_logger(
        logfile_name='main.log',
        logger_name='main'
    )

    # Parse command line arguments
    args=helper_funcs.parse_args()

    # Run data acquisition pipeline
    if args.get_data != 'False':
        data_funcs.get_data()

    # Run perplexity ratio scoring of Hans 2024 data
    if args.perplexity_ratio != 'False':
        perplexity_funcs.perplexity_ratio_score()

    # Run perplexity ratio scoring v2 of Hans 2024 data
    if args.perplexity_ratio_v2 is not None:
        perplexity_funcs_v2.perplexity_ratio_score(args.perplexity_ratio_v2)

    # Run benchmark
    if args.run_benchmark is not None:

        # Loop on the user specified benchmarks
        for benchmark in args.run_benchmark:

            # Parse arguments for this benchmark
            config_file=benchmark[0]
            resume=benchmark[1]

            if resume == 'True' or resume == 'true':
                RESUME=True

            else:
                RESUME=False

            logger.info('Running benchmark from configuration: %s', config_file)

            # Run the benchmark
            runner_funcs.run(
                experiment_config_file=config_file,
                resume=RESUME
            )
