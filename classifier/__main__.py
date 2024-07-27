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


class MakeTFIDFLut(luigi.Task):

    def requires(self):
        return AddPerplexityRatioKLDScore()
    
    def output(self):
        return luigi.LocalTarget(config.TFIDF_LUT, format = Nop)

    def run(self):
        tfidf_lut = data_funcs.make_tfidf_lut()

        with self.output().open('w') as output_file:
            dump(tfidf_lut, output_file, protocol = 5)

class AddTFIDFScore(luigi.Task):

    def requires(self):
        return MakeTFIDFLut()
    
    def output(self):
        return luigi.LocalTarget(config.TFIDF_SCORE_ADDED)

    def run(self):
        data = data_funcs.add_tfidf_score()

        with self.output().open('w') as output_file:
            json.dump(data, output_file)


class TFIDFScoreKLD(luigi.Task):

    def requires(self):
        return AddTFIDFScore()
    
    def output(self):
        return luigi.LocalTarget(config.TFIDF_SCORE_KLD_KDE, format = Nop)
    
    def run(self):
        kl_kde = data_funcs.tfidf_score_kld_kde()

        with self.output().open('w') as output_file:
            dump(kl_kde, output_file, protocol = 5)
    

class AddTFIDFKLDScore(luigi.Task):

    def requires(self):
        return TFIDFScoreKLD()
    
    def output(self):
        return luigi.LocalTarget(config.TFIDF_KLD_SCORE_ADDED)

    def run(self):
        data = data_funcs.add_tfidf_kld_score()

        with self.output().open('w') as output_file:
            json.dump(data, output_file)


class TrainXGBoost(luigi.Task):

    def requires(self):
        return AddTFIDFKLDScore()
    
    def output(self):
        return luigi.LocalTarget(config.XGBOOST_CLASSIFIER, format = Nop)
    
    def run(self):
        model = data_funcs.train_xgboost_classifier()

        with self.output().open('w') as output_file:
            dump(model, output_file, protocol = 5)


if __name__ == '__main__':

    helper_funcs.force_after('MakeTFIDFLut')

    luigi.build(
        [
            LoadData(),
            PerplexityRatioKLD(),
            AddPerplexityRatioKLDScore(),
            MakeTFIDFLut(),
            AddTFIDFScore(),
            TFIDFScoreKLD(),
            AddTFIDFKLDScore(),
            TrainXGBoost()
        ],
        local_scheduler = True
    )
