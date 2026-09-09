import numpy as np
import pandas as pd
from pathlib import Path

from populationsim.core import tracing, inject, pipeline, mp_tasks


def setup_function(func):
    example_dir = Path(__file__).parent.parent / "examples" / "example_survey_weighting"
    configs_dir = example_dir / "configs"
    mp_configs_dir = example_dir / "configs_mp"
    data_dir = example_dir / "data"
    output_dir = Path(__file__).parent / "output"

    inject.add_injectable("configs_dir", [str(mp_configs_dir), str(configs_dir)])
    inject.add_injectable("data_dir", str(data_dir))
    inject.add_injectable("output_dir", str(output_dir))

    inject.clear_cache()

    tracing.config_logger()


def teardown_function(func):
    if pipeline.is_open():
        pipeline.close_pipeline()
    inject.clear_cache()
    inject.reinject_decorated_tables()


def test_weighting_mp():
    """
    Same run as test_weighting, but driven through mp_tasks.

    example_survey_weighting has no sub-geography below the seed geography, so
    configs_mp slices on the seed geography itself: with every control defined
    at SUBREGCluster the seed zones balance independently, and splitting them
    over two processes has to reproduce the serial loop exactly.
    """

    # Debugging ----------------------
    run_list = mp_tasks.get_run_list()
    mp_tasks.print_run_list(run_list)
    # --------------------------------

    injectables = ["data_dir", "configs_dir", "output_dir"]
    injectables = {k: inject.get_injectable(k) for k in injectables}

    mp_tasks.run_multiprocess(injectables)

    pipeline.open_pipeline("_")

    # the sliced/coalesced weight table must cover every household exactly once
    seed_weights = pipeline.get_table("SUBREGCluster_weights")
    seed_households = pipeline.get_table("households")
    assert len(seed_weights) == len(seed_households)
    assert seed_weights.index.is_unique
    assert not seed_weights["balanced_weight"].isna().any()

    summary_hh_weights = pipeline.get_table("summary_hh_weights")
    total_summary_hh_weights = summary_hh_weights["SUBREGCluster_balanced_weight"].sum()
    total_seed_households_weights = seed_households["HHweight"].sum()

    # Should be pretty close but not exact.
    assert abs(total_summary_hh_weights - total_seed_households_weights) < 1

    # multiprocessing must not move the weights: compare against the same
    # baseline test_weighting checks the single process run against.
    # (coalesce does not have to preserve row order, so align on hh_id)
    expected_wts = pd.read_parquet(Path(__file__).parent / "expected" / "weights.parquet")
    actual = summary_hh_weights["SUBREGCluster_balanced_weight"].reindex(
        expected_wts.index
    )
    assert not actual.isna().any()

    assert np.allclose(
        actual.values,
        expected_wts["SUBREGCluster_balanced_weight"].values,
        rtol=1e-8,
        atol=1e-8,
    )

    pipeline.close_pipeline()

    inject.clear_cache()
