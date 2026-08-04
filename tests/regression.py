"""Shared regression helpers for expanded-household comparisons.

Why these helpers exist
-----------------------
The integerizer LP is degenerate: for a given set of controls there are many
household selections with an identical objective value, and which one CBC
returns depends on floating-point detail that is not portable between
machines.  Perturbing the integerizer's inputs by a relative 1e-16 -- below
double-precision epsilon (2.2e-16), i.e. a fraction of one ULP -- is enough to
swap ~1% of the selected households.

Crucially, the *aggregate* result is unaffected.  Measured on
``examples/example_test`` across input perturbations from 1e-16 to 1e-8:

    perturbation   rows      per-TAZ counts   households swapped
    0              1500      exact match      0
    1e-16..1e-8    1500      exact match      11-15  (<= 1.00%)
    1e-6           1498      differ by 1      53     (3.53%)

So row counts and per-zone counts are stable across eight orders of magnitude
of numerical noise, while household *identity* is not stable at any level.

Asserting exact frame equality against a stored baseline therefore tests the
CI runner's hardware as much as it tests PopulationSim: the same commit with a
byte-identical dependency set has passed and failed on different GitHub runner
images.  These helpers assert the invariants that genuinely hold, and bound how
far household selection may drift, instead.

See ActivitySim/populationsim#182 for the underlying nondeterminism.
"""

import warnings

import pandas as pd

# Selection drift observed under noise up to 1e-8 is <= 1.00% of households;
# 2% leaves headroom without letting a real regression through.
MAX_MOVED_FRACTION = 0.02


def count_moved_households(actual, expected):
    """Number of households present in one frame but not the other.

    Both frames are treated as multisets of rows, so this counts genuine
    swaps rather than ordering differences.
    """
    columns = list(expected.columns)
    actual_counts = actual.value_counts(columns)
    expected_counts = expected.value_counts(columns)
    delta = actual_counts.sub(expected_counts, fill_value=0).abs().sum()
    # every swap shows up twice: once as a removal, once as an addition
    return int(delta) // 2


def assert_expanded_regression(
    actual,
    expected,
    zone_col="TAZ",
    max_moved_fraction=MAX_MOVED_FRACTION,
):
    """Assert the portable invariants of an expanded household table.

    Checks total row count, exact per-zone household counts, and that no more
    than ``max_moved_fraction`` of households differ from the baseline.  An
    exact-match difference is reported as a warning so that vertex swaps stay
    visible in CI logs without failing the build.

    Parameters
    ----------
    actual : pandas.DataFrame
        expanded_household_ids produced by the pipeline
    expected : pandas.DataFrame
        baseline loaded from tests/expected/
    zone_col : str
        column holding the finest geography to check counts for
    max_moved_fraction : float
        upper bound on the share of households allowed to differ

    Returns
    -------
    int
        number of households that differ from the baseline
    """
    if len(expected) == 0:
        assert len(actual) == 0, f"expected an empty table, got {len(actual)} rows"
        return 0

    assert len(actual) == len(expected), (
        f"row count {len(actual)} != baseline {len(expected)}"
    )

    actual_counts = actual.groupby(zone_col).size()
    expected_counts = expected.groupby(zone_col).size()
    pd.testing.assert_series_equal(
        actual_counts,
        expected_counts,
        check_names=False,
        obj=f"per-{zone_col} household counts",
    )

    moved = count_moved_households(actual, expected)
    limit = max(1, int(max_moved_fraction * len(expected)))
    assert moved <= limit, (
        f"{moved} households differ from baseline (limit {limit}); "
        f"per-{zone_col} counts still match, so this is a larger selection "
        f"change than solver degeneracy alone explains"
    )

    if moved:
        warnings.warn(
            f"{moved} of {len(expected)} households differ from the stored "
            f"baseline while all aggregates match; this is expected solver "
            f"degeneracy (see ActivitySim/populationsim#182), not a regression",
            stacklevel=2,
        )

    return moved
