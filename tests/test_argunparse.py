"""Tests for argunparse module - strategy refactor."""

from __future__ import annotations

import argparse
import sys
import types


import importlib.util
import pathlib

_ROOT = pathlib.Path(__file__).parent.parent  # d:/Repos/Perso/pytools


def _load_argunparse():
    """Load argunparse with its relative imports stubbed out."""
    pkg = types.ModuleType("pytools")
    sys.modules.setdefault("pytools", pkg)

    spec_t = importlib.util.spec_from_file_location(
        "pytools._typing", _ROOT / "_typing.py"
    )
    mod_t = importlib.util.module_from_spec(spec_t)
    sys.modules["pytools._typing"] = mod_t
    pkg._typing = mod_t
    spec_t.loader.exec_module(mod_t)

    spec_a = importlib.util.spec_from_file_location(
        "pytools.argunparse", _ROOT / "argunparse.py"
    )
    mod_a = importlib.util.module_from_spec(spec_a)
    sys.modules["pytools.argunparse"] = mod_a
    spec_a.loader.exec_module(mod_a)
    return mod_a


_mod = _load_argunparse()
ArgumentUnparser = _mod.ArgumentUnparser
_UnparseContext = _mod._UnparseContext
_BooleanStrategy = _mod._BooleanStrategy
_AppendStrategy = _mod._AppendStrategy
_StandardStrategy = _mod._StandardStrategy
ActionCodec = _mod.ActionCodec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx(parser, quote=False):
    return _UnparseContext(parser=parser, quoteArgs=quote)


def _bool_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--debug", action="store_true")
    return p


def _append_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--files", action="append", nargs="+", type=str)
    return p


def _standard_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--amount", type=int)
    return p


# ---------------------------------------------------------------------------
# _UnparseContext
# ---------------------------------------------------------------------------

class TestUnparseContext:
    def test_holds_parser_and_quoteArgs(self):
        p = argparse.ArgumentParser()
        ctx = _UnparseContext(parser=p, quoteArgs=True)
        assert ctx.parser is p
        assert ctx.quoteArgs is True


# ---------------------------------------------------------------------------
# _BooleanStrategy
# ---------------------------------------------------------------------------

class TestBooleanStrategy:
    def test_emits_flag_when_value_is_true(self):
        p = _bool_parser()
        action = next(a for a in p._actions if a.dest == "debug")
        result = _BooleanStrategy().unparse(action, True, _ctx(p))
        assert result == ["--debug"]

    def test_emits_nothing_when_value_is_false(self):
        p = _bool_parser()
        action = next(a for a in p._actions if a.dest == "debug")
        result = _BooleanStrategy().unparse(action, False, _ctx(p))
        assert result == []

    def test_matches_nargs_zero_action(self):
        p = _bool_parser()
        action = next(a for a in p._actions if a.dest == "debug")
        assert _BooleanStrategy().matches(action) is True

    def test_does_not_match_standard_action(self):
        p = _standard_parser()
        action = next(a for a in p._actions if a.dest == "amount")
        assert _BooleanStrategy().matches(action) is False


# ---------------------------------------------------------------------------
# _AppendStrategy
# ---------------------------------------------------------------------------

class TestAppendStrategy:
    def test_emits_flag_and_values_for_each_invocation(self):
        p = _append_parser()
        action = next(a for a in p._actions if a.dest == "files")
        result = _AppendStrategy().unparse(action, [["a.txt"], ["b.txt"]], _ctx(p))
        assert result == ["--files", "a.txt", "--files", "b.txt"]

    def test_single_invocation(self):
        p = _append_parser()
        action = next(a for a in p._actions if a.dest == "files")
        result = _AppendStrategy().unparse(action, [["a.txt"]], _ctx(p))
        assert result == ["--files", "a.txt"]

    def test_matches_append_action(self):
        p = _append_parser()
        action = next(a for a in p._actions if a.dest == "files")
        assert _AppendStrategy().matches(action) is True

    def test_does_not_match_boolean_action(self):
        p = _bool_parser()
        action = next(a for a in p._actions if a.dest == "debug")
        assert _AppendStrategy().matches(action) is False


# ---------------------------------------------------------------------------
# _StandardStrategy
# ---------------------------------------------------------------------------

class TestStandardStrategy:
    def test_emits_flag_and_scalar_value(self):
        p = _standard_parser()
        action = next(a for a in p._actions if a.dest == "amount")
        result = _StandardStrategy().unparse(action, 12, _ctx(p))
        assert result == ["--amount", "12"]

    def test_matches_non_boolean_non_append_action(self):
        p = _standard_parser()
        action = next(a for a in p._actions if a.dest == "amount")
        assert _StandardStrategy().matches(action) is True

    def test_omits_arg_equal_to_default(self):
        p = _standard_parser()
        action = next(a for a in p._actions if a.dest == "amount")
        result = _StandardStrategy().unparse(action, None, _ctx(p))
        assert result == []


# ---------------------------------------------------------------------------
# Dispatcher (via public API)
# ---------------------------------------------------------------------------

class TestDispatcher:
    def test_boolean_path(self):
        p = _bool_parser()
        result = ArgumentUnparser(p).unparseArgs(debug=True)
        assert result == ["--debug"]

    def test_append_path(self):
        p = _append_parser()
        result = ArgumentUnparser(p).unparseArgs(files=[["a.txt"], ["b.txt"]])
        assert result == ["--files", "a.txt", "--files", "b.txt"]

    def test_standard_path(self):
        p = _standard_parser()
        result = ArgumentUnparser(p).unparseArgs(amount=12)
        assert result == ["--amount", "12"]


# ---------------------------------------------------------------------------
# ActionCodec
# ---------------------------------------------------------------------------

class TestActionCodec:
    def test_has_parse_and_serialize(self):
        codec = ActionCodec(parse=int, serialize=str)
        assert codec.parse("42") == 42
        assert codec.serialize(42) == "42"


# ---------------------------------------------------------------------------
# register_codec - round-tripping
# ---------------------------------------------------------------------------

class TestRegisterCodec:
    def test_no_codec_falls_back_to_str(self):
        p = _standard_parser()
        result = ArgumentUnparser(p).unparseArgs(amount=12)
        assert result == ["--amount", "12"]

    def test_list_type_round_trips_with_codec(self):
        p = argparse.ArgumentParser()
        p.add_argument("value", type=list)
        unparser = ArgumentUnparser(p)
        unparser.register_codec(list, ActionCodec(parse=list, serialize="".join))
        args = unparser.unparseArgs("abc")
        assert args == ["abc"]
        assert p.parse_args(args).value == list("abc")

    def test_csv_type_round_trips_with_codec(self):
        p = argparse.ArgumentParser()
        p.add_argument("--tags", type=lambda s: s.split(","))
        csv_codec = ActionCodec(
            parse=lambda s: s.split(","),
            serialize=",".join,
        )
        unparser = ArgumentUnparser(p)
        unparser.register_codec(p._actions[-1].type, csv_codec)
        args = unparser.unparseArgs(tags=["a", "b", "c"])
        assert args == ["--tags", "a,b,c"]
        assert p.parse_args(args).tags == ["a", "b", "c"]

    def test_codec_does_not_affect_other_actions(self):
        p = argparse.ArgumentParser()
        p.add_argument("--amount", type=int)
        p.add_argument("value", type=list)
        unparser = ArgumentUnparser(p)
        unparser.register_codec(list, ActionCodec(parse=list, serialize="".join))
        args = unparser.unparseArgs("abc", amount=12)
        assert args == ["abc", "--amount", "12"]


# ---------------------------------------------------------------------------
# from_spec with ActionCodec as type
# ---------------------------------------------------------------------------

class TestFromSpecCodec:
    def test_codec_as_type_registers_serialize(self):
        csv_codec = ActionCodec(
            parse=lambda s: s.split(","),
            serialize=",".join,
        )
        unparser = ArgumentUnparser.from_spec([
            {"flags": ["--tags"], "type": csv_codec},
        ])
        args = unparser.unparseArgs(tags=["a", "b", "c"])
        assert args == ["--tags", "a,b,c"]

    def test_codec_as_type_parse_is_used_by_argparse(self):
        csv_codec = ActionCodec(
            parse=lambda s: s.split(","),
            serialize=",".join,
        )
        unparser = ArgumentUnparser.from_spec([
            {"flags": ["--tags"], "type": csv_codec},
        ])
        args = unparser.unparseArgs(tags=["a", "b", "c"])
        reparsed = unparser.parser.parse_args(args)
        assert reparsed.tags == ["a", "b", "c"]
