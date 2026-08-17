from pathlib import Path

import pytest

from populationsim.core import inject, pipeline, tracing

# examples/example_test/configs_intermediate declares
#     geographies: [REGION, DISTRICT, PUMA, TRACT, TAZ]
#     seed_geography: PUMA
# so DISTRICT sits between the meta geography (REGION) and the seed geography
# (PUMA). Its controls.csv puts persons_occ_1 at DISTRICT and persons_occ_2 and
# persons_occ_3 at REGION, which is what lets this example tell the two apart.
SEED_GEOGRAPHY = "PUMA"
INTERMEDIATE_GEOGRAPHY = "DISTRICT"
INTERMEDIATE_CONTROL = "persons_occ_1"
META_CONTROLS = ("persons_occ_2", "persons_occ_3")

_MODELS = [
    "input_pre_processor",
    "setup_data_structures",
    "initial_seed_balancing",
    "meta_control_factoring",
    "final_seed_balancing",
    "integerize_final_seed_weights",
    "sub_balancing.geography=TRACT",
    "sub_balancing.geography=TAZ",
    "expand_households",
    "summarize",
    "write_tables",
    "write_synthetic_population",
]


def setup_function():
    # Other test modules pin `settings` as a plain injectable via
    # config.override_setting; restore the decorated one so this example's own
    # settings -- in particular its five-level `geographies` list -- are read.
    inject.reinject_decorated_tables()


def teardown_function(func):
    if pipeline.is_open():
        pipeline.close_pipeline()
    inject.clear_cache()
    inject.reinject_decorated_tables()


def test_intermediate_geography():

    example_dir = Path(__file__).parent.parent / "examples" / "example_test"

    inject.add_injectable("data_dir", example_dir / "data_intermediate")
    inject.add_injectable("configs_dir", example_dir / "configs_intermediate")
    inject.add_injectable("output_dir", Path(__file__).parent / "output")

    inject.clear_cache()

    tracing.config_logger()

    # Completing at all is the primary regression. Without the control_spec
    # filtering in meta_control_factoring / final_seed_balancing /
    # integerize_final_seed_weights, the seed-level controls are indexed with
    # the intermediate geography's target and this raises
    # KeyError: "['persons_occ_1'] not in index".
    pipeline.run(models=_MODELS, resume_after=None)

    crosswalk = pipeline.get_table("crosswalk")
    assert list(crosswalk.columns) == [
        "REGION",
        INTERMEDIATE_GEOGRAPHY,
        SEED_GEOGRAPHY,
        "TRACT",
        "TAZ",
    ]

    # The behaviour under test: controls belonging to a geography between the
    # meta and seed geographies are kept out of the seed-level control table,
    # while the meta geography's controls are still factored into it.
    seed_controls = pipeline.get_table(f"{SEED_GEOGRAPHY}_controls")
    assert INTERMEDIATE_CONTROL not in seed_controls.columns
    for target in META_CONTROLS:
        assert target in seed_controls.columns

    # The intermediate control is excluded from the seed level, not discarded:
    # it still constrains its own geography.
    intermediate_controls = pipeline.get_table(f"{INTERMEDIATE_GEOGRAPHY}_controls")
    assert INTERMEDIATE_CONTROL in intermediate_controls.columns

    # Sub-geography balancing ran and preserved the seed-level household total.
    seed_total = pipeline.get_table(f"{SEED_GEOGRAPHY}_weights")[
        "balanced_weight"
    ].sum()
    assert seed_total > 0
    for geography in ("TRACT", "TAZ"):
        weights = pipeline.get_table(f"{geography}_weights")["balanced_weight"]
        assert (weights >= 0).all()
        assert weights.sum() == pytest.approx(seed_total, rel=1e-6)

    # This example sets NO_INTEGERIZATION_EVER, so expand_households produces
    # nothing -- assert that explicitly rather than leaving it implied.
    assert pipeline.get_table("expanded_household_ids").empty
