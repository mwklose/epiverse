from epiverse.models import DiscreteDensity
import numpy as np
import pandas as pd


class TestDiscreteDensity():

    def test_example_data(self):

        A = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        L = np.array([1, 0, 1, 0, 1, 0, 0, 1])

        DiscreteDensity
        dd = DiscreteDensity(A=A, L=L)

        dd.fit(event_variable="A")

        # Test Marginal
        assert list(dd.predict(exposure=[1, 0]).values()) == [
            0.375, 0.625], "P[A] and P[^A] failed, using string"

        assert list(dd.predict(exposure=[0, 1]).values()) == [
            0.625, 0.375],  "P[^A] and P[A] failed"

        dd.fit(event_variable="L")

        assert list(dd.predict(exposure=[0, 1]).values()) == [
            0.5, 0.5], "P[L] and P[^L] failed, using string"

        dd.fit(event_variable=1)

        assert list(dd.predict(exposure=[0, 1]).values()) == [
            0.5, 0.5], "P[L] and P[^L] failed, using index"

        dd.fit(event_variable="A", conditioning_set=[])

        assert list(dd.predict([1, 0]).values()) == [
            0.375, 0.625], "P[A] failed, empty conditioning set"

        # Test Conditional
        both_exp = pd.DataFrame([1, 0], columns=["A"])
        exposed = pd.DataFrame([1], columns=["A"])

        dd.fit(event_variable="A", conditioning_set=["L"])

        assert dd.predict([1, 0]) == {(1, 1): 0.5,
                                      (1, 0): 0.25,
                                      (0, 1): 0.5,
                                      (0, 0): 0.75}, "P[A|L]failed"

        dd.fit(event_variable="A", conditioning_set=[
               "L"], conditioning_values=np.array([1]))

        assert dd.predict([1, 0]) == {(1, 1): 0.5,
                                      (0, 1): 0.5}, "P[A|L]failed for only L=1"

        dd.fit(event_variable="A", conditioning_set="L",
               conditioning_values=np.array([1]))

        assert dd.predict([1, 0]) == {(1, 1): 0.5,
                                      (0, 1): 0.5}, "P[A|L]failed for only L=1 as string"

        # Test More complicated conditional, need new dataset

        A1 = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0])
        L1 = np.array([1, 1, 0, 1, 0, 1, 0, 0, 1])
        L2 = np.array([1, 1, 0, 0, 0, 1, 0, 0, 0])

        dd2 = DiscreteDensity(A1=A1, L1=L1, L2=L2)

        dd2.fit(event_variable="A1", conditioning_set=["L1", "L2"],
                conditioning_values=np.array([[1, 1], [1, 0], [0, 0]]))

        assert dd2.predict(exposure=[1, 0]) == {(1, 1, 1): 2/3,
                                                (1, 1, 0): 1/2,
                                                (1, 0, 0): 2/4,
                                                (0, 1, 1): 1/3,
                                                (0, 1, 0): 1/2,
                                                (0, 0, 0): 2/4}, "Complex Pr[A1|L1,L2] failed."

        dd2.fit(event_variable="A1", conditioning_set=["L1", "L2"],
                conditioning_values=np.array([[1, 1], [0, 0]]))

        assert dd2.predict(exposure=[1, 0]) == {(1, 1, 1): 2/3,
                                                (1, 0, 0): 2/4,
                                                (0, 1, 1): 1/3,
                                                (0, 0, 0): 2/4}, "Complex Pr[A1|L1,L2] failed."
