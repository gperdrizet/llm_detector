'''Luigi feature engineering and XGBoost classifier training pipeling'''

import json
import luigi
from pickle import dump
from luigi.format import Nop

import classifier.functions.helper as helper_funcs
import classifier.functions.data_manipulation as data_funcs
import classifier.configuration as config

class LoadData(luigi.Task):

    def output(self):
        return luigi.LocalTarget(config.LOADED_DATA)

    def run(self):
        data = data_funcs.load_data()

        with self.output().open('w') as output_file:
            json.dump(data, output_file)


class PerplexityRatioKLD(luigi.Task):

    def requires(self):
        return LoadData()

    def output(self):
        return luigi.LocalTarget(config.PERPLEXITY_RATIO_KLD_KDE, format = Nop)

    def run(self):
        kl_kde = data_funcs.perplexity_ratio_kld_kde()

        with self.output().open('w') as output_file:
            dump(kl_kde, output_file, protocol = 5)


class AddPerplexityRatioKLDScore(luigi.Task):

    def requires(self):
        return PerplexityRatioKLD()
    
    def output(self):
        return luigi.LocalTarget(config.PERPLEXITY_RATIO_KLD_SCORE_ADDED)

    def run(self):
        data = data_funcs.add_perplexity_ratio_kld_score()

        with self.output().open('w') as output_file:
            json.dump(data, output_file)


class TFIDFScoreKLD(luigi.Task):

    def requires(self):
        return PerplexityRatioKLD()
    
    def output(self):
        return luigi.LocalTarget(config.TFIDF_SCORE_KLD_KDE, format = Nop)
    
    def run(self):
        kl_kde = data_funcs.tfidf_score_kld_kde()

        with self.output().open('w') as output_file:
            dump(kl_kde, output_file, protocol = 5)
    

class AddTFIDFScore(luigi.Task):
    x = luigi.IntParameter()
    y = luigi.IntParameter(default=1)
    z = luigi.IntParameter(default=2)

    def run(self):
        print(self.x * self.y * self.z)


class TrainXGBoost(luigi.Task):
    x = luigi.IntParameter()
    y = luigi.IntParameter(default=1)
    z = luigi.IntParameter(default=2)

    def run(self):
        print(self.x * self.y * self.z)


if __name__ == '__main__':

    helper_funcs.force_after('AddPerplexityRatioKLDScore')

    luigi.build(
        [
            LoadData(),
            PerplexityRatioKLD(),
            AddPerplexityRatioKLDScore(),
            TFIDFScoreKLD()
        ],
        local_scheduler = True
    )
