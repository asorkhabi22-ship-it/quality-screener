"""scripts/build_panels.py: main-sample split."""

import pandas as pd

from scripts.build_panels import split_main_sample


def test_split_main_sample_routes_unknown_sic_to_neither() -> None:
    firms = pd.DataFrame(
        {
            "cik": ["1", "2", "3"],
            "excluded_from_main": pd.array([False, True, pd.NA], dtype="boolean"),
        }
    )
    pit = pd.DataFrame({"cik": ["1", "1", "2", "3"], "revenue": [1.0, 2.0, 3.0, 4.0]})
    main, fin_util = split_main_sample(pit, firms)
    assert main["cik"].tolist() == ["1", "1"]
    assert fin_util["cik"].tolist() == ["2"]
