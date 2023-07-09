import numpy as np
import pandas as pd
from scipy.stats import logistic


def generate_potential(n):
    """Function to generate the potential outcome data for all observations
    """
    d = pd.DataFrame()
    d['W0'] = np.random.binomial(n=1, p=0.5, size=n)

    d['Y1a1'] = np.random.binomial(
        n=1, p=logistic.cdf(-1.5 + 0.5 - 2*d['W0']), size=n)
    d['Y1a0'] = np.random.binomial(
        n=1, p=logistic.cdf(-1.5 + 0 - 2*d['W0']), size=n)
    d['W1a1'] = np.random.binomial(
        n=1, p=logistic.cdf(-1 + d['W0'] - 1), size=n)
    d['W1a0'] = np.random.binomial(
        n=1, p=logistic.cdf(-1 + d['W0'] - 0), size=n)

    d['Y2a1a1'] = np.random.binomial(
        n=1, p=logistic.cdf(-1.5 + 0.1 + 1.2 - 0.5*d['W0'] - 2*d['W1a1']), size=n)
    d['Y2a0a1'] = np.random.binomial(
        n=1, p=logistic.cdf(-1.5 + 0.0 + 1.2 - 0.5*d['W0'] - 2*d['W1a0']), size=n)
    d['Y2a1a0'] = np.random.binomial(
        n=1, p=logistic.cdf(-1.5 + 0.1 + 0.0 - 0.5*d['W0'] - 2*d['W1a1']), size=n)
    d['Y2a0a0'] = np.random.binomial(
        n=1, p=logistic.cdf(-1.5 + 0.0 + 0.0 - 0.5*d['W0'] - 2*d['W1a0']), size=n)

    d['W2a1a1'] = np.random.binomial(
        n=1, p=logistic.cdf(-1 + d['W1a1'] + 0.5*d['W0'] - 0.2 - 1), size=n)
    d['W2a1a0'] = np.random.binomial(
        n=1, p=logistic.cdf(-1 + d['W1a1'] + 0.5*d['W0'] - 0.2 - 0), size=n)
    d['W2a0a1'] = np.random.binomial(
        n=1, p=logistic.cdf(-1 + d['W1a0'] + 0.5*d['W0'] - 0.0 - 1), size=n)
    d['W2a0a0'] = np.random.binomial(
        n=1, p=logistic.cdf(-1 + d['W1a0'] + 0.5*d['W0'] - 0.0 - 0), size=n)

    d['Y3a1a1a1'] = np.random.binomial(n=1, p=logistic.cdf(-1.2 + 0.0 + 0.1 + 1.2 - 0.5*d['W1a1'] - 2*d['W2a1a1']),
                                       size=n)
    d['Y3a0a1a1'] = np.random.binomial(n=1, p=logistic.cdf(-1.2 + 0.0 + 0.1 + 1.2 - 0.5*d['W1a0'] - 2*d['W2a0a1']),
                                       size=n)
    d['Y3a1a0a1'] = np.random.binomial(n=1, p=logistic.cdf(-1.2 + 0.0 + 0.0 + 1.2 - 0.5*d['W1a1'] - 2*d['W2a1a0']),
                                       size=n)
    d['Y3a0a0a1'] = np.random.binomial(n=1, p=logistic.cdf(-1.2 + 0.0 + 0.0 + 1.2 - 0.5*d['W1a0'] - 2*d['W2a0a0']),
                                       size=n)
    d['Y3a1a1a0'] = np.random.binomial(n=1, p=logistic.cdf(-1.2 + 0.0 + 0.1 + 0.0 - 0.5*d['W1a1'] - 2*d['W2a1a1']),
                                       size=n)
    d['Y3a0a1a0'] = np.random.binomial(n=1, p=logistic.cdf(-1.2 + 0.0 + 0.1 + 0.0 - 0.5*d['W1a0'] - 2*d['W2a0a1']),
                                       size=n)
    d['Y3a1a0a0'] = np.random.binomial(n=1, p=logistic.cdf(-1.2 + 0.0 + 0.0 + 0.0 - 0.5*d['W1a1'] - 2*d['W2a1a0']),
                                       size=n)
    d['Y3a0a0a0'] = np.random.binomial(n=1, p=logistic.cdf(-1.2 + 0.0 + 0.0 + 0.0 - 0.5*d['W1a0'] - 2*d['W2a0a0']),
                                       size=n)
    return d


def generate_observed(data):
    """Function to transform the potential outcome data for all observations into the observed data.
    """
    d = data.copy()
    n = d.shape[0]
    d['A0'] = np.random.binomial(n=1, p=logistic.cdf(1 - 2.0*d['W0']), size=n)
    d['W1'] = np.where(d['A0'] == 1, d['W1a1'], d['W1a0'])
    d['A1'] = np.random.binomial(
        n=1, p=logistic.cdf(-1 - 0.2*d['W0'] - d['W1'] + 1.75*d['A0']), size=n)
    d['W2'] = np.where((d['A0'] == 1) & (d['A1'] == 1), d['W2a1a1'], np.nan)
    d['W2'] = np.where((d['A0'] == 1) & (d['A1'] == 0), d['W2a1a0'], d['W2'])
    d['W2'] = np.where((d['A0'] == 0) & (d['A1'] == 1), d['W2a0a1'], d['W2'])
    d['W2'] = np.where((d['A0'] == 0) & (d['A1'] == 0), d['W2a0a0'], d['W2'])
    d['A2'] = np.random.binomial(
        n=1, p=logistic.cdf(-1 - 0.2*d['W1'] - d['W2'] + 1.75*d['A1']), size=n)

    # Generating observed outcomes
    d['Y1'] = np.where(d['A0'] == 1, d['Y1a1'], np.nan)
    d['Y1'] = np.where(d['A0'] == 0, d['Y1a0'], d['Y1'])
    d['Y2'] = np.where((d['A0'] == 1) & (d['A1'] == 1),
                       d['Y2a1a1'], np.nan)
    d['Y2'] = np.where((d['A0'] == 0) & (d['A1'] == 1),
                       d['Y2a0a1'], d['Y2'])
    d['Y2'] = np.where((d['A0'] == 1) & (d['A1'] == 0),
                       d['Y2a1a0'], d['Y2'])
    d['Y2'] = np.where((d['A0'] == 0) & (d['A1'] == 0),
                       d['Y2a0a0'], d['Y2'])
    d['Y3'] = np.where((d['A0'] == 1) & (d['A1'] == 1) & (d['A2'] == 1),
                       d['Y3a1a1a1'], np.nan)
    d['Y3'] = np.where((d['A0'] == 1) & (d['A1'] == 1) & (d['A2'] == 0),
                       d['Y3a1a1a0'], d['Y3'])
    d['Y3'] = np.where((d['A0'] == 1) & (d['A1'] == 0) & (d['A2'] == 1),
                       d['Y3a1a0a1'], d['Y3'])
    d['Y3'] = np.where((d['A0'] == 0) & (d['A1'] == 1) & (d['A2'] == 1),
                       d['Y3a0a1a1'], d['Y3'])
    d['Y3'] = np.where((d['A0'] == 1) & (d['A1'] == 0) & (d['A2'] == 0),
                       d['Y3a1a0a0'], d['Y3'])
    d['Y3'] = np.where((d['A0'] == 0) & (d['A1'] == 1) & (d['A2'] == 0),
                       d['Y3a0a1a0'], d['Y3'])
    d['Y3'] = np.where((d['A0'] == 0) & (d['A1'] == 0) & (d['A2'] == 1),
                       d['Y3a0a0a1'], d['Y3'])
    d['Y3'] = np.where((d['A0'] == 0) & (d['A1'] == 0) & (d['A2'] == 0),
                       d['Y3a0a0a0'], d['Y3'])

    # Generating censoring
    # d['C1'] = np.random.binomial(n=1, p=logistic.cdf(-2.5 - 0.5*d['A0']), size=n)
    # d['Y1'] = np.where(d['C1'] == 1, np.nan, d['Y1'])
    #
    # d['C2'] = np.random.binomial(n=1, p=logistic.cdf(-2.5 - 0.5*d['A1']), size=n)
    # d['W1'] = np.where((d['C2'] == 1) | (d['C1'] == 1), np.nan, d['W1'])
    # d['A1'] = np.where((d['C2'] == 1) | (d['C1'] == 1), np.nan, d['A1'])
    # d['Y2'] = np.where((d['C2'] == 1) | (d['C1'] == 1), np.nan, d['Y2'])
    #
    # d['C3'] = np.random.binomial(n=1, p=logistic.cdf(-2.5 - 0.5*d['A2']), size=n)
    # d['W2'] = np.where((d['C1'] == 1) | (d['C2'] == 1) | (d['C3'] == 1), np.nan, d['W2'])
    # d['A2'] = np.where((d['C1'] == 1) | (d['C2'] == 1) | (d['C3'] == 1), np.nan, d['A2'])
    # d['Y3'] = np.where((d['C1'] == 1) | (d['C2'] == 1) | (d['C3'] == 1), np.nan, d['Y3'])

    return d[['W0', 'A0', 'Y1', 'W1', 'A1', 'Y2', 'W2', 'A2', 'Y3']].copy()
