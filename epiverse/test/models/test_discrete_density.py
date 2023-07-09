import unittest
from epiverse.models.discrete_density import DiscreteDensity
import numpy as np


class TestDiscreteDensity(unittest.TestCase):

    def test_example_data(self):
        A = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        L = np.array([1, 0, 1, 0, 1, 0, 0, 1])

        dd = DiscreteDensity(A=A, L=L).fit()

        # Test Marginal
        self.assertTrue(
            np.allclose(
                dd.predict(exposure=np.array([0, 1]), event_variable="A"),
                np.array([0.625, 0.375])
            ),
            "P[A] and P[^A] failed, using string"
        )

        self.assertTrue(
            np.allclose(
                dd.predict(exposure=np.array([0, 1]), event_variable=0),
                np.array([0.625, 0.375])
            ),
            "P[A] and P[^A] failed, using index"
        )

        self.assertTrue(
            np.allclose(
                dd.predict(exposure=np.array([0, 1]), event_variable="L"),
                np.array([0.5, 0.5])
            ),
            "P[L] and P[^L] failed, using string"
        )

        self.assertTrue(
            np.allclose(
                dd.predict(exposure=np.array([0, 1]), event_variable=1),
                np.array([0.5, 0.5])
            ),
            "P[L] and P[^L] failed, using index"
        )

        self.assertTrue(
            np.allclose(
                dd.predict(exposure=np.array([1]), event_variable="A"),
                np.array([0.375])
            ),
            "P[A] failed, using string"
        )

        self.assertTrue(
            np.allclose(
                dd.predict(exposure=np.array([1]), event_variable=0),
                np.array([0.375])
            ),
            "P[A] failed, using index"
        )

        # Test Conditional
        self.assertTrue(
            np.allclose(
                dd.predict(exposure=np.array(
                    [1]), event_variable=0, conditioning_set=["L"], conditioning_values=np.array([1])),
                np.array([0.5])
            ),
            "P[A|L]failed"
        )

        self.assertTrue(
            np.allclose(
                dd.predict(exposure=np.array(
                    [1]), event_variable=0, conditioning_set="L", conditioning_values=np.array([1])),
                np.array([0.5])
            ),
            "P[A|L]failed"
        )

        self.assertTrue(
            np.allclose(
                dd.predict(exposure=np.array(
                    [1]), event_variable=0, conditioning_set=1, conditioning_values=np.array([1])),
                np.array([0.5])
            ),
            "P[A|L]failed"
        )

        # Test More complicated conditional, need new dataset

        A1 = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0])
        L1 = np.array([1, 1, 0, 1, 0, 1, 0, 0, 1])
        L2 = np.array([1, 1, 0, 0, 0, 1, 0, 0, 0])

        dd2 = DiscreteDensity(A1=A1, L1=L1, L2=L2).fit()

        self.assertTrue(
            np.allclose(
                dd2.predict(exposure=np.array([1, 0]),
                            event_variable="A1",
                            conditioning_set=["L1", "L2"],
                            conditioning_values=np.array([[1, 1],
                                                          [1, 0],
                                                          [0, 0]])),
                np.array([[2/3, 1/3],
                         [0.5, 0.5],
                         [0.5, 0.5]])
            )
        )

        self.assertTrue(
            np.allclose(
                dd2.predict(exposure=np.array([1]),
                            event_variable="A1",
                            conditioning_set=["L1", "L2"],
                            conditioning_values=np.array([[1, 1],
                                                          [1, 0]])),
                np.array([[2/3],
                         [0.5]])
            )
        )

        self.assertTrue(
            dd2.predict(exposure=np.array([1]),
                        event_variable="A1",
                        conditioning_set=["L1", "L2"],
                        conditioning_values=np.array([[1, 1]])) +
            dd2.predict(exposure=np.array([0]),
                        event_variable="A1",
                        conditioning_set=["L1", "L2"],
                        conditioning_values=np.array([[1, 1]])) == 1
        )
