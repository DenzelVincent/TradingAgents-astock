"""Regression tests for the Sina financial-statement response shape.

Sina moved the payload from ``result.data.<fzb|lrb|llb>`` (a flat list of rows)
to ``result.data.report_list{报告期: {..., data: [{item_title, item_value}]}}``.

The old parser only read the former, so all three statements silently came back
as "No ... data found" instead of raising — the agent then reported "this stock
has no financial statements" as a fact.
"""

import copy
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tradingagents.dataflows import a_stock

# Shape returned by the API since ~2026-10.
NEW_SHAPE = {
    "result": {
        "status": {"code": 0},
        "data": {
            "report_count": "2",
            "report_date": [
                {
                    "date_value": "20260630",
                    "date_description": "2026半年报",
                    "date_type": 2,
                },
                {
                    "date_value": "20260331",
                    "date_description": "2026一季报",
                    "date_type": 1,
                },
            ],
            "report_list": {
                "20260630": {
                    "rType": "合并期末",
                    "data": [
                        {"item_field": "", "item_title": "流动资产", "item_value": ""},
                        {
                            "item_field": "CURFDS",
                            "item_title": "货币资金",
                            "item_value": "53518798979.08",
                        },
                    ],
                },
                "20260331": {
                    "rType": "合并期末",
                    "data": [
                        {
                            "item_field": "CURFDS",
                            "item_title": "货币资金",
                            "item_value": "50000000000.0",
                        },
                    ],
                },
            },
        },
    }
}

# Shape the parser was originally written against.
LEGACY_SHAPE = {
    "result": {
        "data": {
            "fzb": [
                {"报告日": "20260630", "货币资金": "53518798979.08"},
                {"报告日": "20260331", "货币资金": "50000000000.0"},
            ]
        }
    }
}


def _response(payload):
    resp = MagicMock()
    resp.json.return_value = payload
    return resp


@pytest.mark.unit
@patch("tradingagents.dataflows.a_stock._requests.get")
def test_new_report_list_shape_is_parsed(mock_get):
    """The newer payload shape must produce real rows, not an empty table."""
    mock_get.return_value = _response(NEW_SHAPE)

    out = a_stock.get_balance_sheet("600519", "quarterly", "2026-09-30")

    assert "No balance sheet data found" not in out
    assert "货币资金" in out
    assert "53518798979.08" in out
    assert "2026-06-30" in out


@pytest.mark.unit
@patch("tradingagents.dataflows.a_stock._requests.get")
def test_legacy_shape_still_parsed(mock_get):
    """Compatibility: the shape the parser was written for must keep working."""
    mock_get.return_value = _response(LEGACY_SHAPE)

    out = a_stock.get_balance_sheet("600519", "quarterly", "2026-09-30")

    assert "No balance sheet data found" not in out
    assert "53518798979.08" in out


@pytest.mark.unit
@patch("tradingagents.dataflows.a_stock._requests.get")
def test_all_three_statements_parse_the_new_shape(mock_get):
    mock_get.return_value = _response(NEW_SHAPE)

    for getter, empty_msg in (
        (a_stock.get_balance_sheet, "No balance sheet data found"),
        (a_stock.get_cashflow, "No cash flow data found"),
        (a_stock.get_income_statement, "No income statement data found"),
    ):
        out = getter("600519", "quarterly", "2026-09-30")
        assert empty_msg not in out, out
        assert "货币资金" in out, out


@pytest.mark.unit
@patch("tradingagents.dataflows.a_stock._requests.get")
def test_unrecognised_shape_degrades_to_empty_table(mock_get):
    """An unknown payload must not raise; it degrades to "no data"."""
    mock_get.return_value = _response({"result": {"data": {"unexpected": []}}})

    out = a_stock.get_balance_sheet("600519", "quarterly", "2026-09-30")

    assert "No balance sheet data found" in out



def _with_publish_dates(h1_published, q1_published):
    """NEW_SHAPE plus the per-period publish_date the live API returns."""
    payload = copy.deepcopy(NEW_SHAPE)
    report_list = payload["result"]["data"]["report_list"]
    report_list["20260630"]["publish_date"] = h1_published
    report_list["20260331"]["publish_date"] = q1_published
    return payload


@pytest.mark.unit
def test_report_deadline_follows_disclosure_rules():
    """一季报 / 三季报 +1 个月、半年报 +2 个月、年报次年 4 月底；非标准期按 4 个月，缺失仍为缺失。"""
    dates = pd.to_datetime(
        pd.Series(["20260331", "20260630", "20260930", "20261231", "20260531", None]),
        errors="coerce",
    )
    got = a_stock._report_deadline(dates)
    assert [d.strftime("%Y-%m-%d") for d in got[:5]] == [
        "2026-04-30", "2026-08-31", "2026-10-31", "2027-04-30", "2026-09-30",
    ]
    assert pd.isna(got.iloc[5])


@pytest.mark.unit
@patch("tradingagents.dataflows.a_stock._requests.get")
def test_report_not_yet_published_on_analysis_date_is_excluded(mock_get):
    """No look-ahead: on 07-15 the 06-30 half-year report (published 08-15) is not out yet."""
    mock_get.return_value = _response(_with_publish_dates("20260815", "20260425"))

    out = a_stock.get_balance_sheet("600519", "quarterly", "2026-07-15")

    assert "53518798979.08" not in out
    assert "2026-06-30" not in out
    assert "50000000000.0" in out
    assert "2026-03-31" in out


@pytest.mark.unit
@patch("tradingagents.dataflows.a_stock._requests.get")
def test_report_published_on_or_before_analysis_date_is_kept(mock_get):
    mock_get.return_value = _response(_with_publish_dates("20260815", "20260425"))

    for curr_date in ("2026-08-15", "2026-09-30"):
        out = a_stock.get_balance_sheet("600519", "quarterly", curr_date)
        assert "53518798979.08" in out, (curr_date, out)
        assert "50000000000.0" in out, (curr_date, out)


@pytest.mark.unit
@patch("tradingagents.dataflows.a_stock._requests.get")
def test_late_republish_date_does_not_hide_report_past_deadline(mock_get):
    """Sina stamps older periods with the date they were re-printed as next year's comparatives.

    The 2026 half-year report is long out by 2026-09-30 (deadline 08-31) even though its
    publish_date says 2027-08-14; it must not be dropped.
    """
    mock_get.return_value = _response(_with_publish_dates("20270814", "20270424"))

    out = a_stock.get_balance_sheet("600519", "quarterly", "2026-09-30")

    assert "53518798979.08" in out
    assert "50000000000.0" in out


@pytest.mark.unit
@patch("tradingagents.dataflows.a_stock._requests.get")
def test_publish_date_column_is_not_returned(mock_get):
    """公告日 of older periods is a re-print date; showing it would mislead the analyst."""
    mock_get.return_value = _response(_with_publish_dates("20260815", "20260425"))

    for curr_date in (None, "2026-09-30"):
        out = a_stock.get_balance_sheet("600519", "quarterly", curr_date)
        assert "公告日" not in out, out
        assert "53518798979.08" in out, out


@pytest.mark.unit
@patch("tradingagents.dataflows.a_stock._requests.get")
def test_legacy_shape_without_publish_date_waits_for_deadline(mock_get):
    mock_get.return_value = _response(LEGACY_SHAPE)

    out = a_stock.get_balance_sheet("600519", "quarterly", "2026-07-15")

    assert "53518798979.08" not in out
    assert "50000000000.0" in out


@pytest.mark.unit
@patch("tradingagents.dataflows.a_stock._requests.get")
def test_all_three_statements_respect_publication(mock_get):
    mock_get.return_value = _response(_with_publish_dates("20260815", "20260425"))

    for getter in (a_stock.get_balance_sheet, a_stock.get_cashflow, a_stock.get_income_statement):
        out = getter("600519", "quarterly", "2026-07-15")
        assert "53518798979.08" not in out, out
        assert "50000000000.0" in out, out


@pytest.mark.unit
@patch("tradingagents.dataflows.a_stock._requests.get")
def test_annual_filter_still_applies_with_publish_dates(mock_get):
    """freq='annual' keeps only December report periods, publish dates or not."""
    mock_get.return_value = _response(_with_publish_dates("20260815", "20260425"))

    out = a_stock.get_balance_sheet("600519", "annual", "2026-09-30")

    assert "No balance sheet data found" in out


@pytest.mark.unit
@patch("tradingagents.dataflows.a_stock._requests.get")
def test_deadline_day_itself_is_not_enough(mock_get):
    """Reports due by 08-31 may come out that evening; only the day after counts without a publish date."""
    mock_get.return_value = _response(LEGACY_SHAPE)

    assert "53518798979.08" not in a_stock.get_balance_sheet("600519", "quarterly", "2026-08-31")
    assert "53518798979.08" in a_stock.get_balance_sheet("600519", "quarterly", "2026-09-01")
