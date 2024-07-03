'''Main function to run jobs based on command line arguments'''

import llm_detector_benchmarking.functions.helper as helper_funcs
import llm_detector_benchmarking.functions.benchmarking as benchmark_funcs
import llm_detector_benchmarking.functions.binoculars as binocular_funcs

if __name__ == "__main__":

    # Parse command line arguments
    args=helper_funcs.parse_args()

    # Start the default logger
    logger=helper_funcs.start_logger('llm_detector.log')

    # Run binoculars
    if args.binoculars != 'False':

        logger.info('Starting binoculars')

        binocular_funcs.binoculars()

    # Run model loading time benchmark
    if args.model_loading_benchmark != 'False':

        logger.info('Starting model loading time benchmark')

        benchmark_funcs.benchmark(
            benchmark_func=benchmark_funcs.load_time,
            resume=args.resume,
            experiment_config_file=args.model_loading_benchmark,
            logger=helper_funcs.start_logger('loading_time_benchmark.log')
        )

    # Run generation rate benchmark
    if args.generation_rate_benchmark != 'False':

        logger.info('Starting generation rate benchmark')

        benchmark_funcs.benchmark(
            benchmark_func=benchmark_funcs.generation_rate,
            resume=args.resume,
            experiment_config_file=args.generation_rate_benchmark,
            logger=helper_funcs.start_logger('generation_rate_benchmark.log')
        )

    # Run decoding benchmark
    if args.decoding_strategy_benchmark != 'False':

        logger.info('Starting decoding strategy benchmark')

        benchmark_funcs.benchmark(
            benchmark_func=benchmark_funcs.decoding_strategy,
            resume=args.resume,
            experiment_config_file=args.decoding_strategy_benchmark,
            logger=helper_funcs.start_logger('decoding_strategy_benchmark.log')
        )

    # Run encoding memory benchmark
    if args.encoding_memory_benchmark != 'False':

        logger.info('Starting encoding memory benchmark')

        benchmark_funcs.benchmark(
            benchmark_func=benchmark_funcs.encoding_memory,
            resume=args.resume,
            experiment_config_file=args.encoding_memory_benchmark,
            logger=helper_funcs.start_logger('encoding_memory_benchmark.log')
        )

    # Run logits calculation benchmark
    if args.logits_calculation_benchmark != 'False':

        logger.info('Starting logits calculation benchmark')

        benchmark_funcs.benchmark(
            benchmark_func=benchmark_funcs.logits_calculation,
            resume=args.resume,
            experiment_config_file=args.logits_calculation_benchmark,
            logger=helper_funcs.start_logger('logits_calculation_benchmark.log')
        )

    # Run binoculars model benchmark
    if args.binoculars_model_benchmark != 'False':

        logger.info('Starting binoculars model benchmark')

        benchmark_funcs.benchmark(
            benchmark_func=benchmark_funcs.binoculars_model,
            resume=args.resume,
            experiment_config_file=args.binoculars_model_benchmark,
            logger=helper_funcs.start_logger('binoculars_model_benchmark.log')
        )
