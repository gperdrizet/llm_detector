'''Class to hold objects and methods for benchmarking 
and optimization experiments'''

from __future__ import annotations
from typing import Callable

import os
import json
import itertools
import llm_detector_benchmarking.configuration as config

class Experiment:
    '''Has generalized data structure for collecting data from experiments
    using two dicts for independent and dependent variables. Also holds
    other experiment metadata and data manipulation methods.'''

    def __init__(self,
        experiment_config_file: str=None,
        logger: Callable=None
    ) -> None:

        # Load the experiment configuration file
        with open(experiment_config_file, 'r', encoding='utf-8') as input_file:
            configuration=json.load(input_file)

        # Add the logger
        self.logger=logger

        # Initialize experiment metadata
        self.experiment_name=configuration['experiment_name']
        self.experiment_description=configuration['experiment_description']

        # Construct output data filename and path
        data_filename=f"{self.experiment_name.replace(' ', '_')}.json"
        self.data_file=f'{config.BENCHMARKING_DATA_PATH}/{data_filename}'

        # Add dicts for independent and dependent vars
        self.independent_vars=configuration['independent_vars']
        self.dependent_vars=configuration['dependent_vars']

        # Make a list of tuples, containing all of the experimental conditions
        # to use for looping during the run
        self.conditions=self.collect_independent_vars()

        # Now that we have captured the condition list, flush the independent variables
        # dict so that we can use the same data structure to record conditions as
        # we complete them during the run
        self.flush_independent_vars()

    def resume(self) -> None:
        '''Method to resume from prior data, if any. Reads prior data
        and adds it to current results. Removes any completed conditions
        from conditions list.'''

        # Holder for any completed conditions we may find
        completed_conditions=[]

        # If we have data to resume from...
        if os.path.isfile(self.data_file) is True:

            self.logger.info('Found old data for resume')

            # Read the prior run's data
            with open(self.data_file, 'r', encoding='utf-8') as input_file:
                old_results=json.load(input_file)
                self.logger.info('Read old run data')

            # Get the values of completed independent variable conditions into a list of lists

            # Loop on keys in the results dict
            for key in old_results.keys():

                # If the key is an independent variable
                if key in self.independent_vars.keys():

                    # Add its list of values to completed conditions
                    completed_conditions.append(old_results[key])

                    # And add the data to the independent vars dict so that
                    # when the results file is overwritten on the first run
                    # the old data is not lost
                    self.independent_vars[key]=old_results[key]

                # If the key is a dependent variable, just write the data
                # to the dependent variable dictionary
                if key in self.dependent_vars.keys():
                    self.dependent_vars[key]=old_results[key]

                self.logger.info(f' {key} has {len(old_results[key])} values')

            # Now expand and zip the list of list containing the completed
            # conditions, this will create a list containing a tuple for each
            # completed run matching the format of our run condition list
            completed_conditions=list(zip(*completed_conditions))
            self.logger.info(f'Collected {len(completed_conditions)} completed run tuples')

        # Then loop on the full conditions list and add only those conditions which
        # have not already been completed to a new list
        new_conditions=[]

        for condition in self.conditions:
            if condition not in completed_conditions:
                new_conditions.append(condition)

        self.logger.info('Created list of conditions left to run')

        # Finally, overwrite the conditions list with the list of new conditions
        # which still need to be completed
        self.conditions=new_conditions

    def collect_independent_vars(self) -> list:
        '''Returns values stored in the independent variables
        dictionary as list of lists'''

        # Make sure the iteration value is last in the independent_vars
        # dict, this will place iteration numbers last in the list of lists
        # so than when looping, all iterations of a given condition
        # will be sequential
        self.independent_vars['iteration']=self.independent_vars.pop('iteration')

        # Empty holder to accumulate results
        independent_var_lists=[]

        # Loop on independent variable dictionary and add each
        # list of values to the result
        for independent_var, independent_var_list in self.independent_vars.items():

            # Handel 'iteration' as a special case - in the config file
            # it contains a single int specifying the number of iterations
            # to run, so use it to construct a list containing iteration
            # numbers to loop on during the run
            if independent_var == 'iteration':
                independent_var_list=list(range(1, independent_var_list + 1))

            independent_var_lists.append(independent_var_list)

        # Take the product of the expanded list of lists - this
        # give a list of tuples for each individual run
        conditions=list(itertools.product(*independent_var_lists))

        return conditions

    def flush_independent_vars(self) -> None:
        '''Empties each list of values in independent variables
        dict'''

        for key in self.independent_vars.keys():
            self.independent_vars[key]=[]

    def save(self) -> None:
        '''Saves data in independent and dependent variable dictionaries to JSON file'''

        # Collect the independent and dependent variable data to a single results dictionary
        results=self.independent_vars

        # Add the dependent variable data
        for key, value in self.dependent_vars.items():
            results[key]=value

        # Serialize collected results to JSON
        with open(self.data_file, 'w', encoding='utf-8') as output:
            json.dump(results, output)
