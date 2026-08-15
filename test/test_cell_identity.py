"""Cell identity: what may a certification cell's banked evidence depend on?

Today a cell checkpoint is guarded by a fleet-wide hash of a whole-file digest,
so editing a comment, a validator, or another cell's plan invalidates all
fourteen cells and forces a 36-hour re-run. These tests pin the two properties
that make a narrower guard trustworthy:

  * the projection's DEFAULT is "numerical" -- a new or renamed symbol is
    included and invalidates, so the exemption list cannot silently rot;
  * an exempt symbol's body genuinely drops out, so validation and reporting
    edits cost nothing.
"""

from __future__ import annotations

import pytest

from quantark.validation.cell_identity import (
    cell_identity_sha256,
    project_source,
    source_projection_sha256,
)

BEFORE = '''
"""Module docstring."""

CONSTANT = 3

def arithmetic(x):
    return x * CONSTANT

def reporting(payload):
    return f"report: {payload}"
'''


def _swap(source: str, old: str, new: str) -> str:
    assert old in source
    return source.replace(old, new)


def test_an_exempt_body_drops_out_of_the_projection():
    """Editing exempt code must not change the projection."""
    edited = _swap(BEFORE, 'return f"report: {payload}"', 'return "totally different"')
    assert BEFORE != edited
    assert project_source(BEFORE, exempt=["reporting"]) == project_source(
        edited, exempt=["reporting"]
    )


def test_a_non_exempt_body_is_part_of_the_projection():
    edited = _swap(BEFORE, "return x * CONSTANT", "return x + CONSTANT")
    assert project_source(BEFORE, exempt=["reporting"]) != project_source(
        edited, exempt=["reporting"]
    )


def test_module_level_constants_stay_in_the_projection():
    """Constants are plan and policy; a changed constant must invalidate."""
    edited = _swap(BEFORE, "CONSTANT = 3", "CONSTANT = 4")
    assert project_source(BEFORE, exempt=["reporting"]) != project_source(
        edited, exempt=["reporting"]
    )


def test_adding_an_exempt_definition_does_not_move_the_projection():
    """The projection must equal 'as if the exempt symbols never existed'.

    Found by probe_numerical_projection_equivalence: deleting a definition's
    lines left its PEP8 blank separators behind, so a revision that ADDED two
    exempt functions projected differently from one that never had them -- a diff
    made entirely of blank lines. That would have forced a 36-hour re-run to pay
    for adding a validator helper, which is the exact failure this whole
    mechanism exists to prevent.
    """
    grown = BEFORE.replace(
        "def reporting(payload):",
        "def also_reporting(payload):\n    return 1\n\n\ndef reporting(payload):",
    )
    assert project_source(BEFORE, exempt=["reporting"]) == project_source(
        grown, exempt=["reporting", "also_reporting"]
    )


def test_blank_line_changes_alone_never_move_the_projection():
    """Whitespace cannot change arithmetic, so it must not gate reuse."""
    spaced = BEFORE.replace("CONSTANT = 3", "CONSTANT = 3\n\n")
    assert project_source(BEFORE, exempt=["reporting"]) == project_source(
        spaced, exempt=["reporting"]
    )


def test_a_new_function_is_numerical_by_default():
    """The default direction: unlisted code counts, so the list cannot rot."""
    edited = BEFORE + "\n\ndef freshly_added(x):\n    return x - 1\n"
    assert project_source(BEFORE, exempt=["reporting"]) != project_source(
        edited, exempt=["reporting"]
    )


def test_bookkeeping_constants_can_be_exempted_so_exempting_is_not_self_defeating():
    """The exemption list must be able to sit outside the projection it governs.

    NON_NUMERICAL_SYMBOLS and the input lists are module-level constants. While
    they are inside the projection, adding a validator helper and listing it as
    exempt CHANGES the list, which invalidates every banked cell -- the act of
    exempting defeats itself, which is how a 36-hour re-run gets charged for a
    validator tweak.

    Exempting them is sound because their effect stays visible in the projection
    anyway: exempting a pre-existing function still removes its body, and a
    newly added exempt function was never present to begin with.
    """
    source = (
        "EXEMPT_LIST = ('reporting',)\n\n"
        "def arithmetic(x):\n    return x\n\n"
        "def reporting(p):\n    return p\n"
    )
    grown = (
        "EXEMPT_LIST = ('reporting', 'also_reporting')\n\n"
        "def arithmetic(x):\n    return x\n\n"
        "def reporting(p):\n    return p\n\n"
        "def also_reporting(p):\n    return p\n"
    )
    exempt_before = ["EXEMPT_LIST", "reporting"]
    exempt_after = ["EXEMPT_LIST", "reporting", "also_reporting"]
    assert project_source(source, exempt=exempt_before) == project_source(
        grown, exempt=exempt_after
    )


def test_exempting_a_pre_existing_function_still_invalidates():
    """The other direction: hiding code that WAS numerical must be visible."""
    source = (
        "def arithmetic(x):\n    return x * 2\n\n" "def helper(x):\n    return x + 1\n"
    )
    assert project_source(source, exempt=["helper"]) != project_source(source, exempt=[])


def test_an_annotated_constant_can_be_exempted():
    source = "LIST: tuple = ('a',)\n\ndef arithmetic(x):\n    return x\n"
    projected = project_source(source, exempt=["LIST"])
    assert "LIST" not in projected
    assert "def arithmetic" in projected


def test_comments_never_move_the_projection():
    """Prose cannot change arithmetic, so it must not cost a re-run.

    This codebase documents its reasoning heavily; if a comment invalidates
    fourteen cells, the rational response is to stop writing comments, which is
    a worse outcome than any digest precision gained.
    """
    commented = BEFORE.replace(
        "def arithmetic(x):", "# explain the scaling below\ndef arithmetic(x):"
    ).replace("return x * CONSTANT", "return x * CONSTANT  # trailing note")
    assert project_source(BEFORE, exempt=["reporting"]) == project_source(
        commented, exempt=["reporting"]
    )


def test_a_hash_inside_a_string_is_not_treated_as_a_comment():
    """Comment stripping must be tokenizer-based, not textual."""
    source = 'KEY = "value # not a comment"\n\ndef arithmetic(x):\n    return x\n'
    assert "value # not a comment" in project_source(source, exempt=[])
    changed = source.replace("value # not a comment", "value # different")
    assert project_source(source, exempt=[]) != project_source(changed, exempt=[])


def test_docstrings_never_move_the_projection():
    source = (
        '"""Module docstring."""\n\n'
        "def arithmetic(x):\n"
        '    """What it does."""\n'
        "    return x * 2\n"
    )
    reworded = (
        '"""Rewritten module docstring, much longer than before."""\n\n'
        "def arithmetic(x):\n"
        '    """Completely different wording."""\n'
        "    return x * 2\n"
    )
    assert project_source(source, exempt=[]) == project_source(reworded, exempt=[])
    # ...but the code around them still counts.
    assert project_source(source, exempt=[]) != project_source(
        source.replace("return x * 2", "return x * 3"), exempt=[]
    )


def test_an_exemption_naming_a_missing_symbol_is_refused():
    """A renamed exempt function must break the build, not silently widen it."""
    with pytest.raises(ValueError, match="not a top-level"):
        project_source(BEFORE, exempt=["reporting", "renamed_away"])


def test_a_duplicated_symbol_is_refused():
    """Two definitions of one name make 'exempt that symbol' ambiguous."""
    doubled = BEFORE + "\n\ndef reporting(other):\n    return 0\n"
    with pytest.raises(ValueError, match="defined more than once"):
        project_source(doubled, exempt=["reporting"])


def test_decorated_and_async_definitions_are_removed_whole():
    source = (
        "import functools\n\n"
        "@functools.cache\n"
        "def decorated(x):\n"
        "    return x\n\n"
        "async def waiter(x):\n"
        "    return x\n\n"
        "def kept(x):\n"
        "    return x\n"
    )
    projected = project_source(source, exempt=["decorated", "waiter"])
    assert "functools.cache" not in projected
    assert "waiter" not in projected
    assert "def kept" in projected
    # The import is module level, so it stays.
    assert "import functools" in projected


def test_classes_can_be_exempted():
    source = (
        "class Reporter:\n    def render(self):\n        return 1\n\n"
        "def arithmetic(x):\n    return x\n"
    )
    projected = project_source(source, exempt=["Reporter"])
    assert "Reporter" not in projected
    assert "def arithmetic" in projected


def test_projection_digest_covers_the_path_as_well_as_the_bytes(tmp_path):
    """Renaming a file must change the digest even if its bytes are identical."""
    one = tmp_path / "one.py"
    two = tmp_path / "two.py"
    one.write_text(BEFORE)
    two.write_text(BEFORE)
    first = source_projection_sha256(
        [(one, ("reporting",))], root=tmp_path
    )
    second = source_projection_sha256(
        [(two, ("reporting",))], root=tmp_path
    )
    assert first != second
    # ...and it is stable for the same input.
    assert first == source_projection_sha256([(one, ("reporting",))], root=tmp_path)


def test_a_non_python_input_is_hashed_whole(tmp_path):
    """Only Python sources can be projected; anything else must hash entirely."""
    data = tmp_path / "table.json"
    data.write_text('{"a": 1}')
    first = source_projection_sha256([(data, ())], root=tmp_path)
    data.write_text('{"a": 2}')
    assert first != source_projection_sha256([(data, ())], root=tmp_path)


def test_exempting_a_symbol_in_a_non_python_input_is_refused(tmp_path):
    data = tmp_path / "table.json"
    data.write_text("{}")
    with pytest.raises(ValueError, match="cannot project"):
        source_projection_sha256([(data, ("something",))], root=tmp_path)


class TestCellIdentity:
    """The identity binds arithmetic, this cell's plan, runtime, and consumers."""

    BASE = dict(
        numerical_sha256="a" * 64,
        plan={"case": "near_ko", "batches": 128},
        runtime={"numpy_version": "1.26.4"},
        consumed={},
    )

    def test_identity_is_stable_and_order_independent(self):
        first = cell_identity_sha256(**self.BASE)
        reordered = dict(self.BASE)
        reordered["plan"] = {"batches": 128, "case": "near_ko"}
        assert first == cell_identity_sha256(**reordered)

    @pytest.mark.parametrize(
        "field,value",
        [
            ("numerical_sha256", "b" * 64),
            ("plan", {"case": "near_ko", "batches": 256}),
            ("runtime", {"numpy_version": "2.0.0"}),
            ("consumed", {"heston/near_ki": "c" * 64}),
        ],
    )
    def test_every_component_moves_the_identity(self, field, value):
        changed = dict(self.BASE)
        changed[field] = value
        assert cell_identity_sha256(**changed) != cell_identity_sha256(**self.BASE)

    def test_a_consumed_cell_changing_invalidates_the_consumer(self):
        """The structural fix: a control's identity is part of its consumer's.

        The 35.5h fleet stopped heston/near_ki early and fed the truncated mean
        into heston_slv/near_ki's telescoping estimator. Nothing invalidated the
        consumer, because no link between them existed in any hash.
        """
        with_control = dict(self.BASE, consumed={"heston/near_ki": "c" * 64})
        moved_control = dict(self.BASE, consumed={"heston/near_ki": "d" * 64})
        assert cell_identity_sha256(**with_control) != cell_identity_sha256(
            **moved_control
        )
