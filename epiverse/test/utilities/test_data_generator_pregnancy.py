from epiverse.utilities.data_generation import DataGeneratorPregnancy


def test_data_generator_pregnancy():
    seed = 700
    dgp = DataGeneratorPregnancy(seed=seed)
    pregnancy = dgp.generate_data(n=1000)

    dgp2 = DataGeneratorPregnancy(seed=seed)
    pregnancy2 = dgp2.generate_data(n=1000)

    pregnancy.to_csv(f"preg_{seed}.csv")

    assert pregnancy.equals(pregnancy2), "RNG able to be set for pregnancy. "
