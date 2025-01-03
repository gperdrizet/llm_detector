'''Main function to run jobs based on command line arguments'''
import os

import functions.helper as helper_funcs
import functions.runner as runner_funcs
import functions.data_manipulation as data_funcs
import functions.perplexity_ratio as perplexity_funcs
import functions.perplexity_ratio_v2 as perplexity_funcs_v2
import functions.perplexity_ratio_score as perplexity_score

# Get path to this file so that we can define other paths relative to it
ROOT_PATH=os.path.dirname(os.path.realpath(__file__))

# Data paths
RAW_DATA_PATH=f'{ROOT_PATH}/data/raw_data'
INTERMEDIATE_DATA_PATH=f'{ROOT_PATH}/data/intermediate_data'
SCORED_DATA_PATH=f'{ROOT_PATH}/data/scored_data'

# Logs path
LOG_PATH=f'{ROOT_PATH}/logs'

if __name__ == "__main__":

    # Parse command line arguments
    args=helper_funcs.parse_args()

    # Run data acquisition pipeline
    if args.get_data != 'False':
        data_funcs.get_data()

    # Run text splitting using semantic splitter
    if args.semantic_split != 'False':
        data_funcs.semantic_split()

    # Run perplexity ratio scoring on split data
    if args.perplexity_ratio_score != 'False':
        perplexity_score.run(
            LOG_PATH,
            INTERMEDIATE_DATA_PATH,
            SCORED_DATA_PATH
        )

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
            RESUME=bool(resume)

            # Run the benchmark
            runner_funcs.run(
                experiment_config_file=config_file,
                resume=RESUME
            )
