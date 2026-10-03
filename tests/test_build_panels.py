"""scripts/build_panels.py and scripts/compute_scores.py helpers."""

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


def test_compute_scores_helpers() -> None:
    from scripts.compute_scores import main_sample, top_m_scores

    scored = pd.DataFrame(
        {
            "excluded_from_main": pd.array([False, True, pd.NA, False], dtype="boolean"),
            "ticker": ["A", "B", "C", "D"], "name": ["a", "b", "c", "d"], "fiscal_year": [2020] * 4,
            "m_score": [-1.0, 5.0, 4.0, -3.0], "m_sgi_w": [1.0] * 4, "m_tata_w": [0.0] * 4,
            "m_dsri_w": [1.0] * 4, "m_aqi_w": [1.0] * 4, "m_gmi_w": [1.0] * 4,
            "m_imputed_components": pd.array([pd.NA] * 4, dtype="string"),
        }
    )
    main = main_sample(scored)
    assert main["ticker"].tolist() == ["A", "D"]
    assert top_m_scores(main, n=1)["ticker"].tolist() == ["A"]
