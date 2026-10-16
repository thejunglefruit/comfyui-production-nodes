# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 The Jungle Fruit
import pytest

from jf_nodes.core.csv_batch import CsvBatchError, RowDefaults, parse_prompt_csv, select_row

SHEET = """prompt,negative,seed,cfg,steps,width,height,tag
"a red fox in snow, 35mm",blurry,42,6.5,28,1024,768,fox
"neon street, rain",,7,,,,,street
"""


def test_parses_typed_values_and_defaults():
    rows = parse_prompt_csv(SHEET, RowDefaults(cfg=7.0, steps=20, width=512, height=512))
    assert len(rows) == 2
    fox, street = rows
    assert fox.prompt == "a red fox in snow, 35mm"
    assert (fox.seed, fox.cfg, fox.steps, fox.width, fox.height, fox.tag) == (42, 6.5, 28, 1024, 768, "fox")
    assert isinstance(fox.seed, int) and isinstance(fox.cfg, float)
    assert street.negative == ""
    assert (street.cfg, street.steps, street.width, street.height) == (7.0, 20, 512, 512)


def test_only_required_columns_and_case_insensitive_headers():
    rows = parse_prompt_csv("Prompt,NEGATIVE,Seed\ncat,dog,1\n")
    assert rows[0].prompt == "cat" and rows[0].seed == 1 and rows[0].steps == 20


def test_semicolon_and_bom_and_blank_lines():
    rows = parse_prompt_csv("﻿prompt;negative;seed\n\nwide shot;;5\n\n")
    assert [r.prompt for r in rows] == ["wide shot"]


def test_64bit_seed_is_exact():
    big = 0xFFFFFFFFFFFFFFFF
    assert parse_prompt_csv(f"prompt,negative,seed\nx,,{big}\n")[0].seed == big


def test_spreadsheet_style_integer_float():
    assert parse_prompt_csv("prompt,negative,seed,steps\nx,,3.0,25.0\n")[0].steps == 25


def test_missing_column_message_lists_what_is_missing():
    with pytest.raises(CsvBatchError) as err:
        parse_prompt_csv("prompt,seed\nx,1\n")
    assert "negative" in str(err.value) and "Found: prompt, seed" in str(err.value)


@pytest.mark.parametrize("body, needle", [
    ("prompt,negative,seed\nx,,abc\n", "line 2: column 'seed' must be an integer"),
    ("prompt,negative,seed\nx,,-1\n", "between 0 and"),
    ("prompt,negative,seed,cfg\nx,,1,high\n", "column 'cfg' must be a number"),
    ("prompt,negative,seed\n,,1\n", "'prompt' is empty"),
    ("prompt,negative,seed\nx,,\n", "'seed' is empty"),
    ("prompt,negative,seed\n", "no data rows"),
    ("   ", "empty"),
])
def test_bad_data_errors_are_specific(body, needle):
    with pytest.raises(CsvBatchError, match=needle):
        parse_prompt_csv(body)


def test_select_row_modes():
    rows = parse_prompt_csv("prompt,negative,seed\na,,1\nb,,2\nc,,3\n")
    assert select_row(rows, 1)[0].prompt == "b"
    assert select_row(rows, 4, "wrap") == (rows[1], 1)
    assert select_row(rows, 9, "clamp") == (rows[2], 2)
    with pytest.raises(CsvBatchError, match="past the end"):
        select_row(rows, 3, "error")
    with pytest.raises(CsvBatchError):
        select_row(rows, -1)
