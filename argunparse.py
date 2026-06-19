"""
Low-level module for argument parsing utilities.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import shlex
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, EnumMeta
from typing import Dict, List, Optional, Tuple, TypeVar

import _types
from _types import Undefined

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

_ValueT = TypeVar("_ValueT")
""" Represents any python object than is parsable into a bash argument representation.

Disclaimer:
    This is a conscious misuse of `TypeVar`, as ValueT does not represent one single specific type,
    but serves as a practical way to document code using this concept throughout this module.
"""

_ArgDictT = Dict[argparse.Action, List[str]]
# Type for a dictionary with actions as keys, and the required list of arguments to pass.


class ParserError(RuntimeError):
    """Raise if incorrect arguments are passed to a parser."""

    def __init__(self, parser: argparse.ArgumentParser, message: str) -> None:
        message = f"{parser.prog}: {message}\n{parser.format_usage()}"
        super().__init__(message)


# ---------------------------------------------------------------------------
# Private argparse API helpers
# ---------------------------------------------------------------------------

def _get_parser_actions(parser: argparse.ArgumentParser) -> List[argparse.Action]:
    """Return all actions registered on *parser*.

    Isolates the single use of `parser._actions` so the rest of the module
    has no direct dependency on that private attribute.

    Args:
        parser (argparse.ArgumentParser): Parser whose actions to retrieve.

    Returns:
        list[argparse.Action]: All actions on the parser.
    """
    return parser._actions  # type: ignore[attr-defined]


def _is_append_action(action: argparse.Action) -> bool:
    """Return `True` if *action* behaves like argparse's built-in append action.

    Args:
        action (argparse.Action): Action to test.

    Returns:
        bool: `True` when the action accumulates repeated flag values into a list.
    """
    return action.__class__.__name__ == "_AppendAction"


def _is_help_action(action: argparse.Action) -> bool:
    """Return `True` if *action* is the standard `-h/--help` action.

    Args:
        action (argparse.Action): Action to test.

    Returns:
        bool: `True` when the action represents the built-in help flag.
    """
    return action.dest == "help" and action.option_strings == ["-h", "--help"]


def _cast_value(action: argparse.Action, value: _ValueT) -> _ValueT:
    """Attempt to cast *value* through *action*'s type callable.

    Args:
        action (argparse.Action): Action whose `type` to apply.
        value (_ValueT): Value to cast.

    Returns:
        _ValueT: The cast value, or *value* unchanged when `action.type` is `None`.

    Raises:
        Exception: Re-raises whatever `action.type(value)` raises.
    """
    if action.type is None:
        return value
    return action.type(value)


@dataclass
class ActionCodec:
    """Pairs the parse and serialize directions for a single argument type.

    Args:
        parse: Converts a shell string token to a Python value (replaces `action.type`).
        serialize: Converts a Python value back to a shell string token (the inverse).

    Without codecs, you could run into this situation:

        >>> parser = argparse.ArgumentParser()
        >>> _ = parser.add_argument("supported", type=int)     # '123' -> 123
        >>> _ = parser.add_argument("unsupported", type=list)  # 'abc' -> ['a', 'b', 'c']
        >>> ArgumentUnparser(parser).unparseArgs(123, ['a', 'b', 'c'])
        ['123', '\'[\'"\'"\'a\'"\'"\', \'"\'"\'b\'"\'"\', \'"\'"\'c\'"\'"\']\'']

    With codecs:

        >>> LIST_CODEC = ActionCodec(
        ...     construct=list,
        ...     represent="".join,
        ... )

        >>> parser = argparse.ArgumentParser()
        >>> _ = parser.add_argument("list", type=list)
        >>> unparser = ArgumentUnparser(parser)
        >>> _ = unparser.register_codec(LIST_CODEC.construct, LIST_CODEC)
        >>> unparser.unparseArgs(list=["a", "b", "c"])
        ['abc']

        >>> parser.parse_args(unparser.unparseArgs(list='abc')).list
        ['a', 'b', 'c']

        Round-trip successful!

    Other examples, with csv and paths:

        >>> CSV_CODEC = ActionCodec(
        ...     construct=lambda s: s.split(","),
        ...     represent=",".join,
        ... )

        >>> PATH_CODEC = ActionCodec(
        ...     construct=Path,
        ...     represent=lambda s: Path(s).as_posix(),
        ... )

        >>> parser = argparse.ArgumentParser()
        >>> _ = parser.add_argument("--tags", type=lambda s: s.split(","))
        >>> _ = parser.add_argument("--path", type=Path)
        >>> unparser = ArgumentUnparser(parser)
        >>> _ = unparser.register_codec(CSV_CODEC.construct, CSV_CODEC)
        >>> _ = unparser.register_codec(PATH_CODEC.construct, PATH_CODEC)

        >>> args = unparser.unparseArgs(tags="a,b,c", path=Path("foo/bar"))
        >>> args
        ['--tags', 'a,b,c', '--path', 'foo/bar']

        >>> reparsed = parser.parse_args(args)
        >>> reparsed.tags
        ['a', 'b', 'c']

        >>> reparsed.path
        WindowsPath('foo/bar')

    """
    construct: object
    represent: object


# ---------------------------------------------------------------------------
# Unparse context
# ---------------------------------------------------------------------------

@dataclass
class _UnparseContext:
    """Carries parser state needed by strategy classes during unparsing.

    Args:
        parser: The argument parser that defines the schema.
        quoteArgs: Whether to run values through :func:`shlex.quote`.
        codecs: Per-type codec registry, keyed by the `action.type` callable.
    """
    parser: argparse.ArgumentParser
    quoteArgs: bool
    codecs: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Shared low-level helpers (called by strategies)
# ---------------------------------------------------------------------------

def _unparse_argument_value(action: argparse.Action, value: _ValueT, ctx: _UnparseContext) -> str:
    """Unparse a single leaf value for *action*.

    Args:
        action: Action to unparse value for.
        value: Python value to convert.
        ctx: Current unparse context.

    Returns:
        str: Shell-safe string representation of *value*.

    Raises:
        ParserError: If *action* takes no value but one was supplied.
        ParserError: If *value* cannot be cast by `action.type`.
    """
    default = ctx.parser.get_default(action.dest)

    if action.nargs == 0 and value != default:
        usage = formatActionAsString(action)
        raise ParserError(ctx.parser, f"{usage}: Does not accept values, yet {value!r} was passed.")

    codec = ctx.codecs.get(action.type)
    if codec is None:
        try:
            _cast_value(action, value)
        except Exception as exception:
            usage = formatActionAsString(action)
            message = f"{usage}: Could not cast {type(value)} value ({value!r}) to {action.type}."
            raise ParserError(ctx.parser, message) from exception

    unparsed = codec.represent(value) if codec else str(value)  # type: ignore[operator]
    if ctx.quoteArgs:
        unparsed = shlex.quote(unparsed)
    return unparsed


def _unparse_values(action: argparse.Action, values: List[_ValueT], ctx: _UnparseContext) -> List[str]:
    """Unparse one repetition's worth of values for *action*.

    Args:
        action: Action to unparse arguments for.
        values: Python values for a single invocation of the action.
        ctx: Current unparse context.

    Returns:
        List[str]: Unparsed arguments for this one invocation.
    """
    default = ctx.parser.get_default(action.dest)
    if action.type:
        try:
            default = action.type(default)
        except Exception:
            usage = formatActionAsString(action)
            log.warning(f"{usage}: could not cast default {default} to {action.type}")

    if values == [default]:
        return []

    requiresOpStr = isOptionStringRequired(action)

    if action.nargs == "+":
        nargs = max(len(values), 1)
    elif action.nargs == "*":
        nargs = len(values)
    elif action.nargs == "?":
        nargs = min(len(values), 1)
    elif action.nargs is None:
        nargs = 1
    else:
        nargs = int(action.nargs)

    unparsedArgs = []
    for index, value_ in enumerate(values, start=1):
        if isOptionStringRequired(action, value_):
            requiresOpStr = True
        if index > nargs:
            continue
        unparsedArgs.append(_unparse_argument_value(action, value_, ctx))

    if requiresOpStr:
        unparsedArgs.insert(0, action.option_strings[0])

    return unparsedArgs


# ---------------------------------------------------------------------------
# Strategy classes - one per action family
# ---------------------------------------------------------------------------

class _ActionStrategy:
    """Base class for action-unparsing strategies."""

    def matches(self, action: argparse.Action) -> bool:
        """Return `True` if this strategy handles *action*."""
        raise NotImplementedError

    def unparse(self, action: argparse.Action, value: _ValueT, ctx: _UnparseContext) -> List[str]:
        """Unparse *value* for *action* given *ctx*."""
        raise NotImplementedError


class _BooleanStrategy(_ActionStrategy):
    """Handles `store_true` / `store_false` actions (`nargs == 0`)."""

    def matches(self, action: argparse.Action) -> bool:
        return action.nargs == 0

    def unparse(self, action: argparse.Action, value: _ValueT, ctx: _UnparseContext) -> List[str]:
        if isOptionStringRequired(action, value):
            return [action.option_strings[0]]
        return []


class _AppendStrategy(_ActionStrategy):
    """Handles `action="append"` - a list-of-lists value."""

    def matches(self, action: argparse.Action) -> bool:
        return _is_append_action(action)

    def unparse(self, action: argparse.Action, value: _ValueT, ctx: _UnparseContext) -> List[str]:
        usage = formatActionAsString(action)
        if not _types.is_collection(value):
            raise ParserError(
                ctx.parser, f"{usage}: only takes iterable values. invalid value: {value}"
            )
        unparsedArgs = []
        for value_ in value:
            unparsedArgs.extend(_unparse_values(action, value_, ctx))
        return unparsedArgs


class _StandardStrategy(_ActionStrategy):
    """Handles all remaining action types (fallback)."""

    def matches(self, action: argparse.Action) -> bool:
        return True

    def unparse(self, action: argparse.Action, value: _ValueT, ctx: _UnparseContext) -> List[str]:
        usage = formatActionAsString(action)
        if action.nargs in ("?", None):
            values = [value]
        elif not _types.is_collection(value):
            raise ParserError(
                ctx.parser, f"{usage}: only takes iterable values. invalid value: {value}"
            )
        else:
            nargs = action.nargs
            if nargs not in ("+", "*") and len(value) != nargs:  # type: ignore[arg-type]
                raise ParserError(
                    ctx.parser, f"{usage}: takes exactly {nargs} values, {len(value)} passed."  # type: ignore[arg-type]
                )
            values = value  # type: ignore[assignment]
        return _unparse_values(action, values, ctx)  # type: ignore[arg-type]


_STRATEGIES: List[_ActionStrategy] = [
    _BooleanStrategy(),
    _AppendStrategy(),
    _StandardStrategy(),
]


# ---------------------------------------------------------------------------
# ArgumentUnparser - engine
# ---------------------------------------------------------------------------

class ArgumentUnparser:
    """For a given `ArgumentParser` and python values, can parse a list of shell string arguments.

    Examples:
        >>> parser = argparse.ArgumentParser()
        >>> _ = parser.add_argument("--files", type=str, action="append", nargs="+")
        >>> _ = parser.add_argument("--amount", type=int)
        >>> _ = parser.add_argument("--debug", action="store_true")

        >>> ArgumentUnparser(parser).unparseArgs([["/path/to/file"], ["/path/to/file3"]], 12)
        ['--files', '/path/to/file', '--files', '/path/to/file3', '--amount', '12']

        >>> ArgumentUnparser(parser).unparseArgs(files=[["/path/to/file"]], amount=12, debug=True)
        ['--files', '/path/to/file', '--amount', '12', '--debug']

        >>> ArgumentUnparser(parser).unparseArgs([["/path/to/file"]])
        ['--files', '/path/to/file']

    Todo:
        TODO: Missing a round-tripping system:

        Obviously, this is a bad example, because we wouldn't usually set the type to list,
        we'd probably use nargs/append instead. Please ignore that.

        >>> parser = argparse.ArgumentParser()
        >>> _ = parser.add_argument("supported", type=int)     # '123' -> 123
        >>> _ = parser.add_argument("unsupported", type=list)  # 'abc' -> ['a', 'b', 'c']
        >>> ArgumentUnparser(parser).unparseArgs(123, ['a', 'b', 'c'])
        ['123', '\\'[\\'"\\'"\\'a\\'"\\'"\\', \\'"\\'"\\'b\\'"\\'"\\', \\'"\\'"\\'c\\'"\\'"\\']\\'']

        Why this result? Obviously, 123 -> '123' is not a problem.
        However, "['a', 'b', 'c']" cannot be round-tripped to the parser action's `list` type.

        Something like a `overrideAction(actionName, **kwargs)` method would maybe suffice?
    """

    def __init__(self, parser: argparse.ArgumentParser, quoteArgs: bool = True):
        """
        Args:
            parser (argparse.ArgumentParser): Parser to use to unparse.
            quoteArgs (bool, optional): Pass `False` to return args as is, without quoting.
                `True` (default) runs args through `shlex.quote` before returning,
                which quotes only when necessary.
        """
        self.parser = parser
        self.quoteArgs = quoteArgs
        self._codecs: dict = {}

    def register_codec(self, type_fn: object, codec: ActionCodec) -> None:
        """Register a codec for actions whose `type` is *type_fn*.

        Args:
            type_fn: The callable passed as `type=` to `add_argument`.
            codec: Codec whose `serialize` will be used instead of `str`.
        """
        self._codecs[type_fn] = codec

    @classmethod
    def from_spec(cls, spec: List[dict], **kwargs) -> "ArgumentUnparser":
        """Construct an `ArgumentUnparser` from a declarative list of argument specs.

        Each entry in *spec* is a plain dict whose `flags` key supplies the
        positional `*name_or_flags` for :py:meth:`argparse.ArgumentParser.add_argument`.
        Every other key is forwarded verbatim as a keyword argument.

        Args:
            spec (list[dict]): Sequence of argument descriptors.  Each dict must
                contain a `flags` key (a list of flag strings, e.g.
                `["--files"]` or `["-f", "--files"]`).  All remaining keys
                are passed directly to `add_argument` (e.g. `type`,
                `action`, `nargs`, `default`, `required`, ...).
            **kwargs: Forwarded to :py:class:`ArgumentUnparser.__init__` (e.g.
                `quoteArgs=False`).

        Returns:
            ArgumentUnparser: A fully initialised unparser backed by the parser
            built from *spec*.

        Raises:
            ValueError: If any entry in *spec* is not a dict, is missing the
                `flags` key, or `flags` is not a non-empty list.

        Example:
            >>> unparser = ArgumentUnparser.from_spec([
            ...     {"flags": ["--files"], "type": str, "action": "append", "nargs": "+"},
            ...     {"flags": ["--amount"], "type": int},
            ...     {"flags": ["--debug"], "action": "store_true"},
            ... ])
            >>> unparser.unparseArgs(files=[["/path/to/file"]], amount=12)
            ['--files', '/path/to/file', '--amount', '12']
        """
        parser = argparse.ArgumentParser()
        codecs = {}

        for index, entry in enumerate(spec):
            if not isinstance(entry, dict):
                raise ValueError(
                    f"spec[{index}]: expected a dict, got {type(entry).__name__!r}"
                )

            entry = dict(entry)

            flags = entry.pop("flags", None)

            if flags is None:
                raise ValueError(f"spec[{index}]: missing required key 'flags'")

            if not isinstance(flags, list) or not flags:
                raise ValueError(
                    f"spec[{index}]: 'flags' must be a non-empty list of strings, got {flags!r}"
                )

            if isinstance(entry.get("type"), ActionCodec):
                codec = entry["type"]
                entry["type"] = codec.construct
                codecs[codec.construct] = codec

            parser.add_argument(*flags, **entry)

        unparser = cls(parser, **kwargs)
        for type_fn, codec in codecs.items():
            unparser.register_codec(type_fn, codec)
        return unparser

    def unparseArgs(self, *args, **kwargs) -> List[str]:
        """Unparse the given args/kwargs using an argparse parser.

        Args:
            parser (argparse.ArgumentParser): Parser to use to unparse arguments.

        Raises:
            ParserError: If too many positional arguments are passed.
            ParserError: If insufficient or too many arguments are passed.

        Returns:
            list[str]: List of unparsed shell arguments.
        """
        argsDict = self._unparsePositionalArgs(*args)
        kwargsDict = self._unparseKeywordArgs(**kwargs)
        overlap = set(argsDict) & set(kwargsDict)

        if overlap:
            raise ParserError(
                self.parser, f"Keyword parameters already passed as positional values: {overlap}"
            )

        argsDict.update(self._unparseKeywordArgs(**kwargs))

        # Verify if any required arguments are missing.
        requiredActions = {action for action in _get_parser_actions(self.parser) if action.required}
        missingActions = requiredActions - {*argsDict}

        if missingActions:
            fActions = ", ".join(map(lambda action: action.dest, missingActions))
            raise ParserError(self.parser, f"Insufficient arguments passed, missing: {fActions}")

        unparsedArgs = list(itertools.chain.from_iterable(argsDict.values()))

        return unparsedArgs

    def _unparsePositionalArgs(self, *args: _ValueT) -> _ArgDictT:
        """Distribute *args into actions.

        Note:
            If more *args are passed than positional actions exist on the parser,
            the extra args are spilled into keyword actions,
            in which case they are unparsed into the keyword argument return dict instead of positional.

            This diverges from argparse, which would simply crash, but feels more pythonic.

        Args:
            args (_ValueT): Python value to unparse to shell argument(s).

        Returns:
            _ArgDictT: Actions and their parsed values.
        """
        positionalActions, keywordActions = _getActionLists(self.parser)
        argsDict = {}

        for action, arg in zip((*positionalActions, *keywordActions), args):
            unparsed = self._unparseArgument(action, arg)
            argsDict[action] = unparsed

        return argsDict

    def _unparseKeywordArgs(self, **kwargs: _ValueT) -> _ArgDictT:
        """Unparse **kwargs into the parsers actions.

        Note:
            **kwargs is in the python sense, not argparse/shell sense.
            Ie: A kwarg can match a positional action's name,
            in which case the argument is unparsed to the positional return dict instead of keyword.

            This diverges from argparse, which would simply crash, but feels more pythonic.

        Args:
            kwargs (_ValueT): Python value to unparse to shell argument(s).

        Returns:
            List[str]: Actions and their parsed values.
        """
        argsDict = defaultdict(list)

        for name, value in kwargs.items():
            action = self.findAction(name)
            previouslyPassed = argsDict.get(action, Undefined)

            if previouslyPassed is not Undefined:
                usage = formatActionAsString(action)
                log.warning(
                    f"{usage}: argument passed more than once: {previouslyPassed!r} // {value!r}"
                )

            argsDict[action].extend(self._unparseArgument(action, value))

        return argsDict

    def findAction(self, name: str, strict: bool = True) -> Optional[argparse.Action]:
        """Find an action matching the given name.

        Args:
            name (str): Name of the action to find.
            strict (bool, optional): Pass `False` to return None if no action is found.
                Defaults to `True`, which raises an exception.

        Raises:
            ParserError: If the name doesn't match any of the parser's actions and strict is `True`.

        Returns:
            argparse.Action | None: Action matching the given name, if any.
        """
        action = None

        # WATCHME: Destination might be tricky when you take actions into account?
        #   It works fine for boolean actions (store true/false)
        #   but hasn't been tested into other weird actions.
        for action_ in _get_parser_actions(self.parser):
            actionNames = (action_.dest, *action_.option_strings)
            if name in actionNames:
                action = action_
                break

        if action is None and strict:
            raise ParserError(self.parser, f"'{name}' does not match any of the parser's arguments.")

        return action

    def _unparseArgument(self, action: argparse.Action, value: _ValueT) -> List[str]:
        """Dispatch to the correct strategy for the given *action* type.

        Args:
            action (argparse.Action): Action to unparse arguments for.
            value (_ValueT): Python value of the action to unparse.

        Returns:
            List[str]: Unparsed arguments.
        """
        ctx = _UnparseContext(parser=self.parser, quoteArgs=self.quoteArgs, codecs=self._codecs)
        for strategy in _STRATEGIES:
            if strategy.matches(action):
                return strategy.unparse(action, value, ctx)
        return []  # unreachable: _StandardStrategy always matches


def isOptionStringRequired(action: argparse.Action, value: _ValueT = None) -> bool:
    """Returns whether the given `value` means the `action` needs to be passed as a flag.

    Args:
        action (argparse.Action): Action to compare to value.
        value (_ValueT, optional): Boolean-able value to compare against action.

    Returns:
        bool: `True` if the `value` means the argument flag needs to be specified, else `False`.
    """
    if not action.option_strings:
        return False

    if action.nargs == 0:
        type_ = type(action.const)

        # Recast is necessary for python-like truthiness.
        #   Without recast: 2 == True --> False
        return action.const == type_(value)

    return True


def hasFallbackValue(action: argparse.Action) -> bool:
    """
    Returns:
        bool: Whether the action has a fallback value.
    """
    return action.default or action.const


def _getActionLists(
    parser: argparse.ArgumentParser, skipHelp: bool = True
) -> Tuple[List[argparse.Action], List[argparse.Action]]:
    """Returns the positional and keyword actions of the given parser as separate lists.

    Args:
        parser (argparse.ArgumentParser): Argument parser whose actions to parse.
        skipHelp (bool, optional): If `True` (default), skips the help action.

    Returns:
        list[argparse.Action], list[argparse.Action]: Positional actions and keyword actions.
    """
    positionalActions = []
    keywordActions = []
    for action in _get_parser_actions(parser):
        if skipHelp and _is_help_action(action):
            continue

        if action.option_strings:
            keywordActions.append(action)
        else:
            positionalActions.append(action)

    return positionalActions, keywordActions


class ChoiceEnumMeta(EnumMeta):
    """ Enum metaclass to simplify round-tripping with argparse.

    See `ChoiceEnum` for an example.
    """
    def __call__(cls, value, *args, **kwargs):
        """ Override call to make values round-trippable."""
        member = cls._member_map_.get(value)
        if member:
            return super().__getitem__(value)

        return super().__call__(value, *args, **kwargs)


class ChoiceEnum(Enum, metaclass=ChoiceEnumMeta):
    """ Enum class to facilitate `ArgumentUnparser` compatible "choice" arguments.

    Examples:
        Define a choice enum:
            >>> class Choice(ChoiceEnum):
            ...     A = 0

        It allows is to pass a name instead of a value when calling:
            >>> str(Choice("A"))
            'A'

        Pass it to a parser's action:
            >>> parser = argparse.ArgumentParser()
            >>> action = parser.add_argument("choice", type=Choice,  choices=Choice)

        Arg is parsed to an enum correctly:
            >>> namespace = parser.parse_args(["A"])
            >>> namespace.choice
            <Choice.A: 0>

        And can be round tripped from an enum:
            >>> action.type(namespace.choice)
            <Choice.A: 0>

        Which allows ingestion by the unparser:
            >>> ArgumentUnparser(parser).unparseArgs(choice=namespace.choice)
            ['A']
    """
    def __str__(self) -> str:
        """Overriding str makes round-tripping easier.

        The unparser can simply str(enumValue) -> 'enumName'
        Which can then be reingested into a CLI (see examples in class docstring).

        Returns:
            str: The member's name.
        """
        return self.name


def formatActionAsString(action: argparse.Action) -> str:
    """Represents an action as a user-readable string.

    Exists because `action.format_usage()` doesn't cover positional actions.

    Args:
        action (argparse.Action): Parser `Action` to format.

    Returns:
        str: A user-readable string representation of an action.
    """
    if action.option_strings:
        actionUsage = action.format_usage()
    else:
        actionUsage = action.dest

    return actionUsage


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
