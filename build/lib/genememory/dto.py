class TrainingRecord(object):
    def __init__(self, raw_pheno=None, linking_function=None, unit_map=None, meta_information=None, error=None):
        self.raw_pheno_type = [raw_pheno]
        self.linking_functions = linking_function
        self.unit_map = unit_map
        self.meta_information = meta_information
        self.error = error

    def get_raw_pheno_type(self):
        """
        Returns the raw phenotype consisting of a nested list
        """
        return self.raw_pheno_type

    def get_linking_function(self):
        return self.linking_functions

    def get_unit_map(self):
        return self.unit_map if self.unit_map is not None else {}

    def get_meta_information(self):
        return self.meta_information

    def get_error_information(self):
        return self.error if self.error is not None else 1
