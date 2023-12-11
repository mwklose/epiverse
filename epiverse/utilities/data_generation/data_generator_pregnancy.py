import polars as pl
from epiverse.utilities.data_generation import DataGenerator


class DataGeneratorPregnancy(DataGenerator):

    def generate_data(n: int, total_ga_weeks: int):
        df = pl.DataFrame({
            "id": range(n)
        })

    def _generate_birth_outcomes():
        # Do this in counterfactual sense; can assign if not treated, then assign if treated.
        pass

    def _generate_index_date():
        pass

    def _generate_treatment_arm():
        pass

    def _generate_preeclampsia():
        pass

    def _regenerate_preeclampsia_outcomes():
        pass

    def _generate_fetal_sga():
        pass

    def _generate_censoring():
        pass
