from epiverse.utilities.data_generation import DataGeneratorPregnancy
from epiverse.survival import KaplanMeier, AalenJohansen
import pandas as pd
import numpy as np


def test_data_generator_pregnancy():
    seed = 700
    dgp = DataGeneratorPregnancy(seed=seed)
    pregnancy = dgp.generate_data(n=1000)

    dgp2 = DataGeneratorPregnancy(seed=seed)
    pregnancy2 = dgp2.generate_data(n=1000)

    pregnancy.to_csv(f"preg_{seed}.csv")

    assert pregnancy.equals(pregnancy2), "RNG able to be set for pregnancy."
    # And, there are all different event types.
    assert sorted(pregnancy["observed_event"].unique()) == [0, 1, 2]


def test_dgp_km():
    dgp = DataGeneratorPregnancy(seed=700)
    pregnancy = dgp.generate_data(n=1000)

    km = KaplanMeier(
        time=pregnancy["ga"], delta=pregnancy["observed_event"], weights=1, event_indicator=1)

    # From R, have that survival at 20w is 0.900, sqrt(0.01827)
    risk_20y = km.predict(20)

    assert abs(risk_20y[1] - 0.900) < 0.001, "20y risk estimate inconsistent."
    assert abs(risk_20y[2] - 0.01827 **
               2) < 0.001, "20y risk std err inconsistent."


def test_dgp_aj():
    dgp = dgp = DataGeneratorPregnancy(seed=700)
    pregnancy = dgp.generate_data(n=1000)

    aj = AalenJohansen(
        time=pregnancy["ga"], delta=pregnancy["observed_event"], weights=1, event_indicator=[1, 2])

    crisk_fd_30w = aj.predict(30, cause=1)
    crisk_lb_30w = aj.predict(30, cause=2)

    print(crisk_fd_30w)

    # Numbers from r: Surv(ga, observed_event, type="mstate") ~ 1
    assert abs(crisk_fd_30w[0, 0] -
               0.12901) <= 0.0001, "Fetal Death calculation incorrect."
    assert abs(np.sqrt(
        crisk_fd_30w[0, 1]) - 0.020712575) <= 0.0001, "Fetal Death variance incorrect."
    assert abs(crisk_lb_30w[0, 0] -
               0.1246) <= 0.0001, "Live birth calculation incorrect."
    assert abs(np.sqrt(crisk_lb_30w[0, 1]) -
               0.021277229) <= 0.0001, "Live birth variance incorrect."


def test_dgp_event_indicator():
    censoring = pd.Series([0, 0, 1, 1, 0, 1, 0])
    event1 = pd.Series([1, 0, 1, 0, 1, 0, 0])
    event2 = pd.Series([0, 1, 0, 0, 0, 1, 1])

    dgp = DataGeneratorPregnancy(seed=700)
    events = dgp._generate_event_indicator(censoring, event1, event2)

    assert (events == pd.Series([1, 2, 0, 0, 1, 0, 2])).all()
