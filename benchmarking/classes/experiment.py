'''Class to hold objects and methods for benchmarking and optimization 
experiments'''

from __future__ import annotations
from typing import List

import os
import json
import logging
import itertools
import benchmarking.configuration as config

# Comment ##############################################################
# Code ########################################################################
class Experiment:
    '''Has generalized data structure for collecting data from
    experiments using two dicts for independent and dependent
    variables. Also holds other experiment metadata and data
    manipulation methods.'''

    def __init__(
            self,
            experiment_config_file: str = None,
            resume: bool = False
    ) -> None:

        # Load the experiment configuration
        with open(experiment_config_file, 'r', encoding='utf-8') as input_file:
            configuration = json.load(input_file)

        # Set the experiment's metadata

        # Add the name of the experiment, derived from the configuration
        # file's name
        config_file_basename = os.path.basename(experiment_config_file)
        self.experiment_name = config_file_basename.split('.')[0]

        # Set the name of the benchmark function to run
        self.benchmark_func = configuration['benchmark_function']

        # Set total iterations to run
        self.iteration = configuration['iteration']

        # Set the experiment description
        self.experiment_description = configuration['experiment_description']

        # Construct output data filename and path
        results_data_filename = f'{self.experiment_name}.jsonl'
        results_data_path = config.BENCHMARKING_DATA_PATH
        results_data_file = f'{results_data_path}/{results_data_filename}'
        self.results_data_file = results_data_file

        # Add the logger
        self.logger = logging.getLogger(f'{self.experiment_name}.Experiment')
        self.logger.debug('Experiment metadata loaded')

        # Set-up experimental data handling

        # If there is data needed for the benchmark, load it
        if 'input_datafile' in configuration.keys():
            self.data = self.load_data(
                input_datafile = configuration['input_datafile'])

            self.logger.debug('Input data loaded')

        else:
            self.data = None

        # Add independent and dependent variables
        self.independent_vars = configuration['independent_vars']
        self.dependent_vars = configuration['dependent_vars']

        # Generate the run list from the configuration file data
        self.run_dicts_list = self.generate_run_dicts_list()
        self.logger.debug('Run dictionaries list generated: %s runs',
                          len(self.run_dicts_list))

        # Initialize the results data structure, reading in data from
        # a previous run and removing completed runs from the run
        # dictionary list, if desired.
        output = self.initialize_results(resume = resume)
        (self.run_results_dicts_list, self.run_dict_list) = output
        self.logger.debug('Results initialized')

        # Batch the run dictionaries list by iteration
        self.run_batches_list = self.batch_run_dicts_list()
        self.logger.debug('Run dictionaries list batched: %s batches',
                         len(self.run_batches_list))

    def generate_run_dicts_list(self) -> List[dict]:
        '''Creates a list of dictionaries where each element dictionary
        represents an individual run with keys and a single value for 
        each independent variable'''

        # First we need to expand the lists of independent variable levels
        # from the configuration data to make runs that contain all of the
        # combinations, do this by creating a list of lists where each
        # element is a run.

        # First, loop on the independent variable dictionary, creating a
        # list of lists containing the variable levels and collecting
        # the variable name keys in order

        independent_var_names = []
        independent_var_levels_lists = []

        for independent_var_name, independent_var_levels in \
            self.independent_vars.items():

            independent_var_names.append(independent_var_name)
            independent_var_levels_lists.append(independent_var_levels)

        # Add the iteration number as an independent variable
        independent_var_names.append('iteration')
        independent_var_levels_lists.append(
            list(range(1, self.iteration + 1)))

        self.logger.debug('Independent variable level list of lists created')
        self.logger.debug('Independent variable level list of lists type: %s',
                           type(independent_var_levels_lists))
        self.logger.debug('Independent variable level list of lists element type: %s',
                           type(independent_var_levels_lists[0]))
        self.logger.debug('Independent variable level list of lists first element: %s',
                           independent_var_levels_lists[0])

        # Take the product of the independent variable levels list of lists - this
        # gives a list of tuples for each individual run
        runs_lists = list(itertools.product(*independent_var_levels_lists))

        self.logger.debug('Run list of lists created')
        self.logger.debug('Run list of lists type: %s',
                          type(runs_lists))
        self.logger.debug('Run list of lists element type: %s',
                          type(runs_lists[0]))
        self.logger.debug('Run list of lists first element: %s', runs_lists[0])

        # Convert the list of run tuples into a list of run dictionaries
        run_dicts_list = []

        for run_list in runs_lists:
            run_dict = {}

            for independent_var_name, independent_var_value in zip(
                independent_var_names, run_list):
                run_dict[independent_var_name] = independent_var_value

            run_dicts_list.append(run_dict)

        self.logger.debug('Run dictionary list created')
        self.logger.debug('Run dictionary list type: %s',
                          type(run_dicts_list))
        self.logger.debug('Run dictionary list element type: %s',
                          type(run_dicts_list[0]))
        self.logger.debug('Run dictionary list first element: %s',
                          run_dicts_list[0])

        return run_dicts_list

    def initialize_results(self, resume: bool = False) -> List[dict]:
        '''Creates results data structure as list of dicts. If resuming
        from prior run, populates results with data from disk and removes
        complete runs from the run dictionary list.'''

        # Read old data from disk and add it to the results dictionary
        # list so that we don't loose data when the data file gets
        # over-written with new results. If no old data exists, or we
        # are not resuming, make an empty holder for the results
        # dictionary list

        if resume is True and os.path.isfile(self.results_data_file) is True:
            self.logger.debug('Resuming from old data')

            results_dicts_list = []

            with open(self.results_data_file, 'r',
                      encoding='utf-8') as input_file:

                for line in input_file:
                    results_dicts_list.append(json.loads(line))

            self.logger.debug('%s old run results loaded',
                              len(results_dicts_list))
            self.logger.debug('Results type: %s',
                              type(results_dicts_list))
            self.logger.debug('Results element type: %s',
                              type(results_dicts_list[0]))
            self.logger.debug('Results first element: %s',
                              results_dicts_list[0])

        else:
            results_dicts_list = []
            self.logger.debug('Created empty list for results')

        # If we are resuming remove any completed runs in the results
        # dictionary list from the run dictionary list so we don't run
        # them again
        if resume is True and os.path.isfile(self.results_data_file) is True:

            # First, make a new list of dictionaries from the results data
            # which contains only the independent variable
            completed_run_dicts_list = []

            for results_dict in results_dicts_list:
                completed_independent_vars_dict = {}

                for key, value in results_dict.items():
                    if key in self.independent_vars.keys() or key == 'iteration':

                        completed_independent_vars_dict[key] = value

                completed_run_dicts_list.append(
                    completed_independent_vars_dict)

            self.logger.debug('Completed run dictionary list created from old results: %s runs', 
                              len(completed_run_dicts_list))
            self.logger.debug('Completed run dictionary list type: %s',
                              type(completed_run_dicts_list))
            self.logger.debug('Completed run dictionary list element type: %s',
                              type(completed_run_dicts_list[0]))
            self.logger.debug('Completed run dictionary list first element: %s',
                              completed_run_dicts_list[0])

            # Then, remove any runs found in the list of completed run
            # dictionaries from the list of run dictionaries
            new_run_dicts_list = []

            for run_dict in self.run_dicts_list:
                if run_dict not in completed_run_dicts_list:
                    new_run_dicts_list.append(run_dict)

            self.logger.debug('Completed runs removed from run dictionary list')
            self.logger.debug('%s total runs, %s runs remaining',
                            len(self.run_dicts_list), len(new_run_dicts_list))
            self.logger.debug('New run dictionary list type: %s',
                            type(new_run_dicts_list))
            self.logger.debug('New run dictionary list element type: %s',
                            type(new_run_dicts_list[0]))
            self.logger.debug('New run dictionary list first element: %s',
                            new_run_dicts_list[0])

        # If we are not resuming, just return the original run
        # dictionaries list
        else:
            new_run_dicts_list = self.run_dicts_list

        return results_dicts_list, new_run_dicts_list


    def batch_run_dicts_list(self) -> List[List[dict]]:
        '''Creates list of lists from run dictionaries list where
        each list element is a list of dictionaries forming a batch
        of iterations with all other independent variables the same.'''

        # Loop on the run dictionaries list, collecting the batches by
        # iteration number
        batches_lists = []
        batch_list = []

        self.logger.debug('Batching...')

        for run_dict in self.run_dicts_list:
            batch_list.append(run_dict)
            self.logger.debug('Run: %s', run_dict)

            if run_dict['iteration'] == self.iteration:
                batches_lists.append(batch_list)
                self.logger.debug('Batch complete')
                batch_list = []

        self.logger.debug('Run dictionary list batched: %s batches of %s runs',
                        len(batches_lists), len(batches_lists[0]))
        self.logger.debug('Batch list type: %s', type(batches_lists))
        self.logger.debug('Batch list element type: %s',
                        type(batches_lists[0]))
        self.logger.debug('First batch element type: %s',
                          type(batches_lists[0][0]))
        self.logger.debug('First batch element: %s', batches_lists[0][0])

        return batches_lists

    def extend_results(self, incoming_results: list=None):
        '''Adds results from batch to experiment class's results list'''

        self.run_results_dicts_list.extend(incoming_results)
        self.logger.debug('%s incoming results added to list', len(incoming_results))


    def save_results(self) -> None:
        '''Saves current contents of results data structure to disk
        as JSON lines.'''

        # Serialize collected results to JSON
        with open(self.results_data_file, 'w', encoding='utf-8') as output:
            for run_result_dict in self.run_results_dicts_list:
                json.dump(run_result_dict, output)
                output.write('\n')

        self.logger.debug('Results saved to file')


    def load_data(self, input_datafile: str = None):
        '''Loads input data for benchmark run'''

        self.logger.debug('Loading data from %s', input_datafile)

        # Get the full path to the datafile
        input_file_path = f'{config.DATA_PATH}/{input_datafile}'

        # Check the type of data we are loading via the file extension
        data_type = input_datafile.split('.')[-1]
        self.logger.debug('Datatype is %s', data_type)

        input_data = []

        if data_type == 'jsonl':

            with open(input_file_path, 'r', encoding='utf-8') as input_file:
                for line in input_file:
                    input_data.append(json.loads(line))

            self.logger.debug('Loaded %s json lines', len(input_data))

        return input_data
