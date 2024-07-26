import json
import luigi
import classifier.functions.data_manipulation as data_funcs
import classifier.configuration as config

class LoadData(luigi.Task):

    def output(self):
        return luigi.LocalTarget(config.LOADED_DATA)

    def run(self):
        data = data_funcs.load_data()

        with self.output().open('w') as output_file:
            json.dump(data, output_file)

class PerplexityRatioScoreKLDivergence(luigi.Task):
    x = luigi.IntParameter()
    y = luigi.IntParameter(default=1)
    z = luigi.IntParameter(default=2)

    def run(self):
        print(self.x * self.y * self.z)

class AddPerplexityRatioScore(luigi.Task):
    x = luigi.IntParameter()
    y = luigi.IntParameter(default=1)
    z = luigi.IntParameter(default=2)

    def run(self):
        print(self.x * self.y * self.z)

class TFIDFScoreKLDivergence(luigi.Task):
    x = luigi.IntParameter()
    y = luigi.IntParameter(default=1)
    z = luigi.IntParameter(default=2)

    def run(self):
        print(self.x * self.y * self.z)

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
    luigi.build(
        [
            LoadData()
        ], 
        local_scheduler=True
    )