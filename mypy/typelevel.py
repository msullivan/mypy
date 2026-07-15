"""Type-level computation evaluation.

This module provides the evaluation functions for type-level computations
(TypeOperatorType, TypeForComprehension).

Note: Conditional types are now represented as _Cond[...] TypeOperatorType.

"""

from __future__ import annotations

import functools
import inspect
import itertools
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, NamedTuple

from mypy.expandtype import expand_type, expand_type_by_instance
from mypy.maptype import map_instance_to_supertype
from mypy.mro import calculate_mro
from mypy.nodes import (
    ARG_NAMED,
    ARG_NAMED_OPT,
    ARG_OPT,
    ARG_POS,
    ARG_STAR,
    ARG_STAR2,
    MDEF,
    ArgKind,
    Argument,
    Block,
    ClassDef,
    Context,
    Decorator,
    FuncDef,
    RefExpr,
    SymbolTable,
    SymbolTableNode,
    TypeInfo,
    Var,
)
from mypy.subtypes import is_subtype
from mypy.typeops import make_simplified_union, tuple_fallback
from mypy.types import (
    AnyType,
    CallableType,
    ComputedType,
    Instance,
    LiteralType,
    NoneType,
    ProperType,
    TupleType,
    Type,
    TypeAliasType,
    TypedDictType,
    TypeForComprehension,
    TypeOfAny,
    TypeOperatorType,
    TypeType,
    TypeVarLikeType,
    UnboundType,
    UninhabitedType,
    UnionType,
    UnpackType,
    get_proper_type,
    has_type_vars,
    is_stuck_expansion,
)

if TYPE_CHECKING:
    from mypy.semanal_shared import SemanticAnalyzerInterface


MAX_DEPTH = 100


class TypeLevelContext:
    """Holds the context for type-level computation evaluation.

    This is a global mutable state that provides access to the semantic analyzer
    API during type operator expansion. The context is set via a context manager
    before type analysis and cleared afterward.
    """

    def __init__(self) -> None:
        self._api: SemanticAnalyzerInterface | None = None
        # Make an evaluator part of this state also, so that we can
        # maintain a depth tracker and an outer error message context.
        #
        # XXX: but maybe we should always thread the evaluator back
        # ourselves or something instead?
        self._evaluator: TypeLevelEvaluator | None = None
        self._suppress_errors: bool = False
        # True while the main semantic-analysis passes (module top levels and
        # function bodies) are running, before class transformations such as
        # UpdateClass have been applied. See defer_class_schema() below.
        self._defer_class_schema: bool = False

    @property
    def api(self) -> SemanticAnalyzerInterface | None:
        """Get the current semantic analyzer API, or None if not in context."""
        return self._api

    @contextmanager
    def set_api(self, api: SemanticAnalyzerInterface) -> Iterator[None]:
        """Context manager to set the API for type-level evaluation.

        Usage:
            with typelevel_ctx.set_api(self.api):
                # Type operators can now access the API via typelevel_ctx.api
                result = get_proper_type(some_type)
        """
        saved = self._api
        self._api = api
        try:
            yield
        finally:
            self._api = saved

    @contextmanager
    def suppress_errors(self) -> Iterator[None]:
        """Suppress side-effectful errors (e.g. from RaiseError) during evaluation.

        Used during type formatting to prevent spurious errors when
        get_proper_type evaluates TypeOperatorTypes for display purposes.
        """
        saved = self._suppress_errors
        self._suppress_errors = True
        try:
            yield
        finally:
            self._suppress_errors = saved

    @contextmanager
    def defer_class_schema(self) -> Iterator[None]:
        """Defer type-level reads of class member schemas during semantic analysis.

        Class transformations such as UpdateClass are applied in a dedicated
        pass (apply_class_plugin_hooks) that runs *after* the main semantic
        analysis passes which resolve annotations and type aliases. Those
        earlier passes can nonetheless force evaluation of type-level
        computations (e.g. via has_placeholder() or annotation analysis).

        While this context is active, reading the member schema of a class that
        is subject to a not-yet-applied UpdateClass effect raises
        EvaluationStuck, so the surrounding computation stays unevaluated (and
        is retried later, once the schema is final) instead of observing the
        stale pre-transformation schema and emitting spurious errors.
        """
        saved = self._defer_class_schema
        self._defer_class_schema = True
        try:
            yield
        finally:
            self._defer_class_schema = saved


# Global context instance for type-level computation
typelevel_ctx: Final = TypeLevelContext()


# Registry mapping operator names (not full!) to their evaluation functions
OperatorFunc = Callable[..., Type]


@dataclass(frozen=True)
class OperatorInfo:
    """Metadata about a registered operator evaluator."""

    func: OperatorFunc
    expected_argc: int | None  # None for variadic


_OPERATOR_EVALUATORS: dict[str, OperatorInfo] = {}


EXPANSION_ANY = AnyType(TypeOfAny.expansion_stuck)

EXPANSION_OVERFLOW = AnyType(TypeOfAny.from_error)


def register_operator(name: str) -> Callable[[OperatorFunc], OperatorFunc]:
    """Decorator to register an operator evaluator.

    The function signature is inspected to determine the expected number of
    positional arguments (excluding the keyword-only ``evaluator`` parameter).
    If a ``*args`` parameter is found, the operator is treated as variadic.
    """

    def decorator(func: OperatorFunc) -> OperatorFunc:
        # inspect.signature follows __wrapped__ set by functools.wraps,
        # so this sees through the lift_over_unions wrapper automatically.
        sig = inspect.signature(func)
        argc = (
            None
            if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
            else len(sig.parameters) - 1
        )

        _OPERATOR_EVALUATORS[name] = OperatorInfo(func=func, expected_argc=argc)
        return func

    return decorator


def lift_over_unions(func: OperatorFunc) -> OperatorFunc:
    """Decorator that lifts an operator to work over union types.

    If any argument is a union type, the operator is applied to each
    combination of union elements and the results are combined into a union.

    For example, Concat[Literal['a'] | Literal['b'], Literal['c']]
    becomes Literal['ac'] | Literal['bc'].

    Works at the unpacked-args level: the wrapped function receives individual
    ``Type`` positional arguments plus a keyword-only ``evaluator``.
    """

    @functools.wraps(func)
    def wrapper(*args: Type, evaluator: TypeLevelEvaluator) -> Type:
        expanded: list[list[Type]] = []
        for arg in args:
            proper = get_proper_type(arg)
            if isinstance(proper, UnionType):
                expanded.append(list(proper.items))
            elif isinstance(proper, UninhabitedType):
                # Never decomposes to an empty union: the operator never runs.
                expanded.append([])
            else:
                expanded.append([arg])

        combinations = list(itertools.product(*expanded))

        # Fast path: no unions to fan out over, just call the operator directly.
        if len(combinations) == 1:
            return func(*args, evaluator=evaluator)

        results: list[Type] = []
        for combo in combinations:
            result = func(*combo, evaluator=evaluator)
            if not (isinstance(result, ProperType) and isinstance(result, UninhabitedType)):
                results.append(result)

        return UnionType.make_union(results)

    return wrapper


class EvaluationStuck(Exception):
    pass


class EvaluationOverflow(Exception):
    pass


class TypeLevelError(Exception):
    """Raised by an operator evaluator when its arguments are invalid.

    Caught in eval_operator, which emits the message via api.fail and
    returns UninhabitedType.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class TypeLevelEvaluator:
    """Evaluates type-level computations to concrete types."""

    def __init__(self, api: SemanticAnalyzerInterface, ctx: Context | None) -> None:
        self.api = api
        self.ctx = ctx
        self.depth = 0
        self._current_op: TypeOperatorType | None = None

        self.cache: dict[Type, Type] = {}

    @property
    def error_ctx(self) -> Context:
        """Get the best available error context for reporting."""
        ctx = self.ctx or self._current_op
        assert ctx is not None, "No error context available"
        return ctx

    def evaluate(self, typ: Type) -> Type:
        """Main entry point: evaluate a type to its simplified form."""

        if typ in self.cache:
            return self.cache[typ]

        if self.depth >= MAX_DEPTH:
            ctx = self.ctx or typ
            # Use serious=True to bypass in_checked_function() check which requires
            # self.options to be set on the SemanticAnalyzer
            self.api.fail("Type expansion is too deep; producing Any", ctx, serious=True)
            raise EvaluationOverflow()

        try:
            self.depth += 1
            if isinstance(typ, TypeOperatorType):
                rtyp = self.eval_operator(typ)
            elif isinstance(typ, TypeForComprehension):
                rtyp = evaluate_comprehension(self, typ)
            else:
                rtyp = typ  # Already a concrete type or can't be evaluated

            self.cache[typ] = rtyp

            return rtyp
        finally:
            self.depth -= 1

    def eval_proper(self, typ: Type) -> ProperType:
        """Main entry point: evaluate a type to its simplified form."""
        typ = get_proper_type(self.evaluate(typ))
        # A call to another expansion via an alias got stuck, reraise here
        if is_stuck_expansion(typ):
            raise EvaluationStuck
        if isinstance(typ, (TypeVarLikeType, UnboundType, ComputedType)):
            raise EvaluationStuck
        if isinstance(typ, UnpackType) and isinstance(typ.type, TypeVarLikeType):
            raise EvaluationStuck

        return typ

    def eval_operator(self, typ: TypeOperatorType) -> Type:
        """Evaluate a type operator by dispatching to registered handler."""
        name = typ.type.name

        info = _OPERATOR_EVALUATORS.get(name)
        if info is None:
            return EXPANSION_ANY

        old_op = self._current_op
        self._current_op = typ
        try:
            args = typ.args
            if info.expected_argc is None:
                args = self.flatten_args(args)
            elif len(args) != info.expected_argc:
                return UninhabitedType()
            try:
                return info.func(*args, evaluator=self)
            except TypeLevelError as err:
                if not typelevel_ctx._suppress_errors:
                    self.api.fail(err.message, self.error_ctx, serious=True)
                return AnyType(TypeOfAny.from_error)
        finally:
            self._current_op = old_op

    # --- Type construction helpers ---

    def literal_bool(self, value: bool) -> LiteralType:
        """Create a Literal[True] or Literal[False] type."""
        return LiteralType(value, self.api.named_type("builtins.bool"))

    def literal_int(self, value: int) -> LiteralType:
        """Create a Literal[int] type."""
        return LiteralType(value, self.api.named_type("builtins.int"))

    def flatten_args(self, args: list[Type]) -> list[Type]:
        """Flatten type arguments, evaluating and unpacking as needed.

        Handles UnpackType from comprehensions by expanding the inner TupleType.
        """
        flat_args: list[Type] = []
        for arg in args:
            evaluated = self.eval_proper(arg)
            if isinstance(evaluated, UnpackType):
                inner = get_proper_type(evaluated.type)
                if isinstance(inner, TupleType):
                    flat_args.extend(inner.items)
                else:
                    flat_args.append(evaluated)
            else:
                flat_args.append(evaluated)
        return flat_args

    def literal_str(self, value: str) -> LiteralType:
        """Create a Literal[str] type."""
        return LiteralType(value, self.api.named_type("builtins.str"))

    def tuple_type(self, items: list[Type]) -> TupleType:
        """Create a tuple type with the given items."""
        return TupleType(items, self.api.named_type("builtins.tuple"))

    def get_typemap_type(self, name: str) -> Instance:
        # They are always in _typeshed.typemap in normal runs, but are
        # sometimes missing from typing, depending on the version.
        # But _typeshed.typemap doesn't exist in tests, so...
        if typ := self.api.named_type_or_none(f"typing.{name}"):
            return typ
        else:
            return self.api.named_type(f"_typeshed.typemap.{name}")


def _call_by_value(evaluator: TypeLevelEvaluator, typ: Type) -> Type:
    """Make sure alias arguments are evaluated before expansion.

    Currently this is used in conditional bodies, which should protect
    any recursive uses, to make sure that arguments to potentially
    recursive aliases get evaluated before substituted in, to make
    sure that they don't grow without bound.

    This shouldn't be necessary for correctness, but can be important
    for performance.

    We should *maybe* do it in more places! Possibly everywhere?  Or
    maybe we should do it *never* and just do a better job of caching.
    """
    if isinstance(typ, TypeAliasType):
        typ = typ.copy_modified(
            args=[get_proper_type(_call_by_value(evaluator, st)) for st in typ.args]
        )

    # Evaluate recursively instead of letting it get handled in the
    # get_proper_type loop to help maintain better error contexts.
    return evaluator.eval_proper(typ)


@register_operator("_Cond")
def _eval_cond(
    condition: Type, true_type: Type, false_type: Type, *, evaluator: TypeLevelEvaluator
) -> Type:
    """Evaluate _Cond[condition, TrueType, FalseType]."""
    result = extract_literal_bool(evaluator.eval_proper(condition))

    if result is True:
        return _call_by_value(evaluator, true_type)
    elif result is False:
        return _call_by_value(evaluator, false_type)
    else:
        # Undecidable - return Any for now
        # In the future, we might want to keep the conditional and defer evaluation
        return EXPANSION_ANY


@register_operator("_And")
def _eval_and(left_arg: Type, right_arg: Type, *, evaluator: TypeLevelEvaluator) -> Type:
    """Evaluate _And[cond1, cond2] - logical AND of type booleans."""
    left = extract_literal_bool(evaluator.eval_proper(left_arg))
    if left is False:
        # Short-circuit: False and X = False
        return evaluator.literal_bool(False)
    if left is None:
        return UninhabitedType()

    right = extract_literal_bool(evaluator.eval_proper(right_arg))
    if right is None:
        return UninhabitedType()

    return evaluator.literal_bool(right)


@register_operator("_Or")
def _eval_or(left_arg: Type, right_arg: Type, *, evaluator: TypeLevelEvaluator) -> Type:
    """Evaluate _Or[cond1, cond2] - logical OR of type booleans."""
    left = extract_literal_bool(evaluator.eval_proper(left_arg))
    if left is True:
        # Short-circuit: True or X = True
        return evaluator.literal_bool(True)
    if left is None:
        return UninhabitedType()

    right = extract_literal_bool(evaluator.eval_proper(right_arg))
    if right is None:
        return UninhabitedType()

    return evaluator.literal_bool(right)


@register_operator("_Not")
def _eval_not(arg: Type, *, evaluator: TypeLevelEvaluator) -> Type:
    """Evaluate _Not[cond] - logical NOT of a type boolean."""
    result = extract_literal_bool(evaluator.eval_proper(arg))
    if result is None:
        return UninhabitedType()

    return evaluator.literal_bool(not result)


@register_operator("_DictEntry")
def _eval_dict_entry(name_type: Type, value_type: Type, *, evaluator: TypeLevelEvaluator) -> Type:
    """Evaluate _DictEntry[name, typ] -> Member[name, typ, Never, Never, Never].

    This is the internal type operator for dict comprehension syntax in type context.
    {k: v for x in foo} desugars to *[_DictEntry[k, v] for x in foo].
    """
    member_info = evaluator.get_typemap_type("Member")
    never = UninhabitedType()
    return Instance(member_info.type, [name_type, value_type, never, never, never])


@register_operator("Iter")
def _eval_iter(arg: Type, *, evaluator: TypeLevelEvaluator) -> Type:
    """Evaluate a type-level iterator (Iter[T])."""
    target = evaluator.eval_proper(arg)
    if isinstance(target, TupleType):
        return target
    else:
        return UninhabitedType()


@register_operator("IsAssignable")
def _eval_isass(lhs: Type, rhs: Type, *, evaluator: TypeLevelEvaluator) -> Type:
    """Evaluate a type-level condition (IsAssignable[T, Base])."""
    left_proper = evaluator.eval_proper(lhs)
    right_proper = evaluator.eval_proper(rhs)

    # Handle type variables - may be undecidable
    if has_type_vars(left_proper) or has_type_vars(right_proper):
        return EXPANSION_ANY

    result = is_subtype(left_proper, right_proper)

    return evaluator.literal_bool(result)


@register_operator("IsEquivalent")
def _eval_isequiv(lhs: Type, rhs: Type, *, evaluator: TypeLevelEvaluator) -> Type:
    """Evaluate IsEquivalent[T, S] - check if T and S are equivalent types.

    Returns Literal[True] if T is a subtype of S AND S is a subtype of T.
    Equivalent to: IsAssignable[T, S] and IsAssignable[S, T]
    """
    left_proper = evaluator.eval_proper(lhs)
    right_proper = evaluator.eval_proper(rhs)

    # Handle type variables - may be undecidable
    if has_type_vars(left_proper) or has_type_vars(right_proper):
        return EXPANSION_ANY

    # Both directions must hold for type equivalence
    result = is_subtype(left_proper, right_proper) and is_subtype(right_proper, left_proper)

    return evaluator.literal_bool(result)


@register_operator("Bool")
def _eval_bool(arg: Type, *, evaluator: TypeLevelEvaluator) -> Type:
    """Evaluate Bool[T] - check if T contains Literal[True].

    Returns Literal[True] if T is Literal[True] or a union containing Literal[True].
    Equivalent to: IsAssignable[Literal[True], T] and not IsAssignable[T, Never]
    """
    arg_proper = evaluator.eval_proper(arg)

    # Check if Literal[True] is a subtype of arg (i.e., arg contains True)
    # and arg is not Never
    literal_true = evaluator.literal_bool(True)
    contains_true = is_subtype(literal_true, arg_proper)
    is_never = isinstance(arg_proper, UninhabitedType)

    return evaluator.literal_bool(contains_true and not is_never)


def extract_literal_bool(typ: Type) -> bool | None:
    """Extract bool value from LiteralType."""
    typ = get_proper_type(typ)
    if isinstance(typ, LiteralType) and isinstance(typ.value, bool):
        return typ.value
    return None


def extract_literal_int(typ: Type) -> int | None:
    """Extract int value from LiteralType."""
    typ = get_proper_type(typ)
    if (
        isinstance(typ, LiteralType)
        and isinstance(typ.value, int)
        and not isinstance(typ.value, bool)
    ):
        return typ.value
    return None


def extract_literal_string(typ: Type) -> str | None:
    """Extract string value from LiteralType."""
    typ = get_proper_type(typ)
    if isinstance(typ, LiteralType) and isinstance(typ.value, str):
        return typ.value
    return None


def extract_qualifier_strings(typ: Type) -> list[str]:
    """Extract qualifier strings from a type that may be a Literal or Union of Literals.

    Used to extract qualifiers from Member[..., quals, ...] where quals can be:
    - A single Literal[str] like Literal["ClassVar"]
    - A Union of Literals like Literal["ClassVar"] | Literal["Final"]
    - Never (no qualifiers) - returns empty list
    """
    typ = get_proper_type(typ)
    qual_strings: list[str] = []

    if isinstance(typ, LiteralType) and isinstance(typ.value, str):
        qual_strings.append(typ.value)
    elif isinstance(typ, UnionType):
        for item in typ.items:
            item_proper = get_proper_type(item)
            if isinstance(item_proper, LiteralType) and isinstance(item_proper.value, str):
                qual_strings.append(item_proper.value)

    return qual_strings


def _callable_to_params(evaluator: TypeLevelEvaluator, target: CallableType) -> list[Type]:
    """Convert a CallableType's parameters into a list of Param[name, type, kind, default] instances."""
    param_info = evaluator.get_typemap_type("Param")
    never = UninhabitedType()
    params: list[Type] = []

    # Align target.arg_* with the original FuncDef.arguments by position, so we
    # can pull the default initializer for each slot (covering pos-only args too).
    defn = target.definition
    if isinstance(defn, Decorator):
        defn = defn.func
    defn_args: list[Argument] = []
    if isinstance(defn, FuncDef):
        defn_args = list(defn.arguments)
        # Bound methods drop self/cls from arg_types; trim leading defn args
        # until the lengths match.
        while len(defn_args) > len(target.arg_types):
            defn_args.pop(0)
        # If a decorator transformed the signature, names won't agree;
        # bail out of positional alignment in that case.
        if any(
            n is not None and n != a.variable.name for n, a in zip(target.arg_names, defn_args)
        ):
            defn_args = []

    for i, (arg_type, arg_kind, arg_name) in enumerate(
        zip(target.arg_types, target.arg_kinds, target.arg_names)
    ):
        if arg_name is not None:
            name_type: Type = evaluator.literal_str(arg_name)
        else:
            name_type = NoneType()

        kind: str
        has_default = False
        if arg_kind in (ARG_POS, ARG_OPT):
            # name=None marks a positional-only parameter; otherwise it's a
            # regular positional-or-keyword parameter.
            kind = "positional" if arg_name is None else "positional_or_keyword"
            if arg_kind == ARG_OPT:
                has_default = True
        elif arg_kind == ARG_STAR:
            kind = "*"
        elif arg_kind == ARG_NAMED:
            kind = "keyword"
        elif arg_kind == ARG_NAMED_OPT:
            kind = "keyword"
            has_default = True
        elif arg_kind == ARG_STAR2:
            kind = "**"
        else:
            kind = "positional_or_keyword"

        kind_type: Type = evaluator.literal_str(kind)

        default_type: Type
        if has_default:
            # Use the literal-refined init_type populated by check_default_params
            # if it's available (after type checking); otherwise fall back to
            # the annotation type (during semanal-time evaluation).
            init_type = defn_args[i].variable.init_type if i < len(defn_args) else None
            default_type = init_type if init_type is not None else arg_type
        else:
            default_type = never

        params.append(Instance(param_info.type, [name_type, arg_type, kind_type, default_type]))

    return params


def _get_args(evaluator: TypeLevelEvaluator, target: Type, base: Type) -> Sequence[Type] | None:
    target = evaluator.eval_proper(target)
    base = evaluator.eval_proper(base)

    # TODO: Other cases:
    # * Overloaded
    # * classmethod/staticmethod (decorated callables)

    if isinstance(target, CallableType) and isinstance(base, CallableType):
        if not is_subtype(target, base):
            return None
        params = _callable_to_params(evaluator, target)
        return [evaluator.tuple_type(params), target.ret_type]

    if isinstance(target, Instance) and isinstance(base, Instance):
        # TODO: base.is_protocol!!
        # Probably implement it by filling in base with TypeVars and
        # calling infer_constraints and solve.

        return get_type_args_for_base(target, base.type)

    if isinstance(target, NoneType):
        return _get_args(evaluator, evaluator.api.named_type("builtins.object"), base)

    if isinstance(target, TupleType):
        # TODO: tuple v tuple?
        # TODO: Do a real check against more classes?
        if isinstance(base, Instance) and target.partial_fallback == base:
            return target.items

        return _get_args(evaluator, tuple_fallback(target), base)

    if isinstance(target, TypedDictType):
        return _get_args(evaluator, target.fallback, base)

    if isinstance(target, TypeType):
        if isinstance(base, Instance) and base.type.fullname == "builtins.type":
            return [target.item]
        # TODO: metaclasses, protocols
        return None

    return None


@register_operator("GetArg")
@lift_over_unions
def _eval_get_arg(
    target: Type, base: Type, index_arg: Type, *, evaluator: TypeLevelEvaluator
) -> Type:
    """Evaluate GetArg[T, Base, Idx] - get type argument at index from T as Base."""
    eval_target = evaluator.eval_proper(target)
    eval_base = evaluator.eval_proper(base)
    if isinstance(eval_target, AnyType):
        return eval_target
    args = _get_args(evaluator, eval_target, eval_base)

    if args is None:
        raise TypeLevelError(f"GetArg: {eval_target} is not a subclass of {eval_base}")

    # Extract index as int
    index = extract_literal_int(evaluator.eval_proper(index_arg))
    if index is None:
        return UninhabitedType()  # Can't evaluate without literal index

    if index < 0:
        index += len(args)
    if 0 <= index < len(args):
        return args[index]

    raise TypeLevelError(f"GetArg: index out of range for {eval_target} as {eval_base}")


@register_operator("GetArgs")
@lift_over_unions
def _eval_get_args(target: Type, base: Type, *, evaluator: TypeLevelEvaluator) -> Type:
    """Evaluate GetArgs[T, Base] -> tuple of all type args from T as Base."""
    eval_target = evaluator.eval_proper(target)
    eval_base = evaluator.eval_proper(base)
    if isinstance(eval_target, AnyType):
        return eval_target
    args = _get_args(evaluator, eval_target, eval_base)

    if args is None:
        raise TypeLevelError(f"GetArgs: {eval_target} is not a subclass of {eval_base}")
    return evaluator.tuple_type(list(args))


@register_operator("FromUnion")
def _eval_from_union(arg: Type, *, evaluator: TypeLevelEvaluator) -> Type:
    """Evaluate FromUnion[T] -> tuple of union elements."""
    target = evaluator.eval_proper(arg)

    if isinstance(target, UnionType):
        return evaluator.tuple_type(list(target.items))
    else:
        # Non-union becomes 1-tuple
        return evaluator.tuple_type([target])


@register_operator("_NewUnion")
def _eval_new_union(*args: Type, evaluator: TypeLevelEvaluator) -> Type:
    """Evaluate _NewUnion[*Ts] -> union of all type arguments."""
    return make_simplified_union(list(args))


@register_operator("_NewCallable")
def _eval_new_callable(*args: Type, evaluator: TypeLevelEvaluator) -> Type:
    """Evaluate _NewCallable[Param1, Param2, ..., ReturnType] -> CallableType.

    The last argument is the return type. All preceding arguments should be
    Param[name, type, kind, default] instances describing the callable's parameters.
    """
    if len(args) == 0:
        return AnyType(TypeOfAny.from_error)

    ret_type = args[-1]
    param_args = args[:-1]

    arg_types: list[Type] = []
    arg_kinds: list[ArgKind] = []
    arg_names: list[str | None] = []

    for param in param_args:
        param = evaluator.eval_proper(param)
        if not isinstance(param, Instance):
            # Not a Param instance, skip
            continue
        if param.type.fullname not in ("typing.Param", "_typeshed.typemap.Param"):
            continue

        # Param[name, type, kind, default]
        p_args = param.args
        if len(p_args) < 2:
            continue

        name = extract_literal_string(get_proper_type(p_args[0]))
        param_type = p_args[1]
        if len(p_args) > 2 and isinstance(get_proper_type(p_args[2]), UninhabitedType):
            raise TypeLevelError("Param kind cannot be Never; must be a ParamKind")
        kinds = extract_qualifier_strings(p_args[2]) if len(p_args) > 2 else []
        if len(kinds) > 1:
            raise TypeLevelError(f"Param kind must be a single Literal, got Literal{kinds!r}")
        has_default = len(p_args) > 3 and not isinstance(
            get_proper_type(p_args[3]), UninhabitedType
        )

        kind_set = set(kinds)
        if "*" in kind_set:
            arg_kinds.append(ARG_STAR)
            arg_names.append(None)
        elif "**" in kind_set:
            arg_kinds.append(ARG_STAR2)
            arg_names.append(None)
        elif "keyword" in kind_set:
            if has_default:
                arg_kinds.append(ARG_NAMED_OPT)
            else:
                arg_kinds.append(ARG_NAMED)
            arg_names.append(name)
        elif has_default:
            arg_kinds.append(ARG_OPT)
            arg_names.append(name)
        else:
            arg_kinds.append(ARG_POS)
            arg_names.append(name)

        arg_types.append(param_type)

    fallback = evaluator.api.named_type("builtins.function")
    return CallableType(
        arg_types=arg_types,
        arg_kinds=arg_kinds,
        arg_names=arg_names,
        ret_type=ret_type,
        fallback=fallback,
    )


@register_operator("GetMember")
@lift_over_unions
def _eval_get_member(target_arg: Type, name_arg: Type, *, evaluator: TypeLevelEvaluator) -> Type:
    """Evaluate GetMember[T, Name] - get Member type for named member from T."""
    name = extract_literal_string(evaluator.eval_proper(name_arg))
    if name is None:
        return UninhabitedType()

    eval_target = evaluator.eval_proper(target_arg)
    if isinstance(eval_target, AnyType):
        return eval_target
    members = _get_members_dict(eval_target, evaluator=evaluator, attrs_only=False)
    member = members.get(name)
    if member is None:
        raise TypeLevelError(f"GetMember: {name!r} not found in {eval_target}")
    return member


@register_operator("GetMemberType")
@lift_over_unions
def _eval_get_member_type(
    target_arg: Type, name_arg: Type, *, evaluator: TypeLevelEvaluator
) -> Type:
    """Evaluate GetMemberType[T, Name] - get attribute type from T."""
    name = extract_literal_string(evaluator.eval_proper(name_arg))
    if name is None:
        return UninhabitedType()

    eval_target = evaluator.eval_proper(target_arg)
    if isinstance(eval_target, AnyType):
        return eval_target
    members = _get_members_dict(eval_target, evaluator=evaluator, attrs_only=False)
    member = members.get(name)
    if member is not None:
        # Extract the type argument (index 1) from Member[name, typ, quals, init, definer]
        member = get_proper_type(member)
        if isinstance(member, Instance) and len(member.args) > 1:
            return member.args[1]
    return UninhabitedType()


@register_operator("_TypeGetAttr")
def _eval_type_get_attr(
    target_arg: Type, name_arg: Type, *, evaluator: TypeLevelEvaluator
) -> Type:
    """Evaluate _TypeGetAttr[T, Name] - get attribute from a Member.

    Internal operator for dot notation: m.attr desugars to _TypeGetAttr[m, Literal["attr"]].
    Unlike GetMemberType, this only works on Member instances, not arbitrary types.
    """
    target = evaluator.eval_proper(target_arg)

    member_info = evaluator.get_typemap_type("Member")
    param_info = evaluator.get_typemap_type("Param")
    if not isinstance(target, Instance) or target.type not in (member_info.type, param_info.type):
        name_type = evaluator.eval_proper(name_arg)
        name = extract_literal_string(name_type)
        evaluator.api.fail(
            f"Dot notation .{name} requires a Member or Param type, got {target}",
            evaluator.error_ctx,
            serious=True,
        )
        return UninhabitedType()

    # Direct call bypasses union lifting, which is correct for Member/Param access
    return _eval_get_member_type(target_arg, name_arg, evaluator=evaluator)


@register_operator("Slice")
@lift_over_unions
def _eval_slice(
    target_arg: Type, start_arg: Type, end_arg: Type, *, evaluator: TypeLevelEvaluator
) -> Type:
    """Evaluate Slice[S, Start, End] - slice a literal string or tuple type."""
    target = evaluator.eval_proper(target_arg)
    if isinstance(target, AnyType):
        return target

    # Handle start - can be int or None
    start_type = evaluator.eval_proper(start_arg)
    if isinstance(start_type, NoneType):
        start: int | None = None
    else:
        start = extract_literal_int(start_type)
        if start is None:
            return UninhabitedType()

    # Handle end - can be int or None
    end_type = evaluator.eval_proper(end_arg)
    if isinstance(end_type, NoneType):
        end: int | None = None
    else:
        end = extract_literal_int(end_type)
        if end is None:
            return UninhabitedType()

    # Handle literal string slicing
    s = extract_literal_string(target)
    if s is not None:
        result = s[start:end]
        return evaluator.literal_str(result)

    # Handle tuple type slicing
    if isinstance(target, TupleType):
        sliced_items = target.items[start:end]
        return evaluator.tuple_type(sliced_items)

    raise TypeLevelError(f"Slice: {target} is not a tuple or string Literal")


@register_operator("Concat")
@lift_over_unions
def _eval_concat(left_arg: Type, right_arg: Type, *, evaluator: TypeLevelEvaluator) -> Type:
    """Evaluate Concat[S1, S2] - concatenate two literal strings."""
    left = extract_literal_string(evaluator.eval_proper(left_arg))
    right = extract_literal_string(evaluator.eval_proper(right_arg))

    if left is not None and right is not None:
        return evaluator.literal_str(left + right)

    return UninhabitedType()


@register_operator("Uppercase")
@lift_over_unions
def _eval_uppercase(arg: Type, *, evaluator: TypeLevelEvaluator) -> Type:
    """Evaluate Uppercase[S] - convert literal string to uppercase."""
    s = extract_literal_string(evaluator.eval_proper(arg))
    if s is not None:
        return evaluator.literal_str(s.upper())

    return UninhabitedType()


@register_operator("Lowercase")
@lift_over_unions
def _eval_lowercase(arg: Type, *, evaluator: TypeLevelEvaluator) -> Type:
    """Evaluate Lowercase[S] - convert literal string to lowercase."""
    s = extract_literal_string(evaluator.eval_proper(arg))
    if s is not None:
        return evaluator.literal_str(s.lower())

    return UninhabitedType()


@register_operator("Capitalize")
@lift_over_unions
def _eval_capitalize(arg: Type, *, evaluator: TypeLevelEvaluator) -> Type:
    """Evaluate Capitalize[S] - capitalize first character of literal string."""
    s = extract_literal_string(evaluator.eval_proper(arg))
    if s is not None:
        return evaluator.literal_str(s.capitalize())

    return UninhabitedType()


@register_operator("Uncapitalize")
@lift_over_unions
def _eval_uncapitalize(arg: Type, *, evaluator: TypeLevelEvaluator) -> Type:
    """Evaluate Uncapitalize[S] - lowercase first character of literal string."""
    s = extract_literal_string(evaluator.eval_proper(arg))
    if s is not None:
        result = s[0].lower() + s[1:] if s else s
        return evaluator.literal_str(result)

    return UninhabitedType()


@register_operator("Members")
@lift_over_unions
def _eval_members(target: Type, *, evaluator: TypeLevelEvaluator) -> Type:
    """Evaluate Members[T] -> tuple of Member types for all members of T.

    Includes methods, class variables, and instance attributes.
    """
    return _eval_members_impl(target, evaluator=evaluator, attrs_only=False)


@register_operator("Attrs")
@lift_over_unions
def _eval_attrs(target: Type, *, evaluator: TypeLevelEvaluator) -> Type:
    """Evaluate Attrs[T] -> tuple of Member types for annotated attributes only.

    Excludes methods but includes ClassVar members.
    """
    return _eval_members_impl(target, evaluator=evaluator, attrs_only=True)


def _eval_members_impl(
    target_arg: Type, *, evaluator: TypeLevelEvaluator, attrs_only: bool
) -> Type:
    """Common implementation for Members and Attrs operators."""
    members = _get_members_dict(target_arg, evaluator=evaluator, attrs_only=attrs_only)
    return evaluator.tuple_type(list(members.values()))


def _should_skip_type_info(type_info: TypeInfo, api: SemanticAnalyzerInterface) -> bool:
    """Determine whether to skip a type_info when collecting members.

    HACK: The rules here need to be more clearly defined. For now, we skip
    anything from a stub file except typing.Member (which needs to be
    introspectable for GetMemberType/dot notation on Member instances).
    """
    # TODO: figure out the real rules for this
    if type_info.fullname in (
        "typing.Member",
        "_typeshed.typemap.Member",
        "typing.Param",
        "_typeshed.typemap.Param",
    ):
        return False
    module = api.modules.get(type_info.module_name)
    if module is not None and module.is_stub:
        return True
    return False


def _returns_update_class(func_type: CallableType | None) -> bool:
    """Whether a callable's return type is an (unevaluated) UpdateClass[...]."""
    if func_type is None:
        return False
    ret_type = func_type.ret_type
    # Don't use get_proper_type here - it would eagerly expand the operator.
    return isinstance(ret_type, TypeOperatorType) and ret_type.type.name == "UpdateClass"


def _update_class_func_type(node: object) -> CallableType | None:
    """Extract the CallableType of a (possibly decorated) function definition."""
    if isinstance(node, Decorator):
        node = node.func
    if isinstance(node, FuncDef) and isinstance(node.type, CallableType):
        return node.type
    return None


def _class_has_pending_update_class(info: TypeInfo) -> bool:
    """Whether `info` is (or will be) transformed by an UpdateClass effect.

    Mirrors the detection in semanal_main._apply_update_class_effects: a class
    is a target of UpdateClass if a class decorator returns UpdateClass, or if
    an __init_subclass__ in one of its base classes returns UpdateClass.

    Used together with typelevel_ctx._defer_class_schema to hold off reading a
    class's member schema until the transformation has actually been applied.
    """
    for decorator in info.defn.decorators:
        if isinstance(decorator, RefExpr) and decorator.node is not None:
            if _returns_update_class(_update_class_func_type(decorator.node)):
                return True
    for base in info.mro[1:]:
        sym = base.names.get("__init_subclass__")
        if sym is not None and _returns_update_class(_update_class_func_type(sym.node)):
            return True
    return False


def _get_members_dict(
    target_arg: Type, *, evaluator: TypeLevelEvaluator, attrs_only: bool
) -> dict[str, Type]:
    """Build a dict of member name -> Member type for all members of target.

    Args:
        attrs_only: If True, filter to attributes only (excludes methods).
                    If False, include all members.

    Returns a dict mapping member names to Member[name, typ, quals, init, definer]
    instance types.
    """
    target = evaluator.eval_proper(target_arg)

    member_info = evaluator.get_typemap_type("Member")

    if isinstance(target, TypedDictType):
        return _get_typeddict_members_dict(target, member_info.type, evaluator=evaluator)

    if not isinstance(target, Instance):
        return {}

    if typelevel_ctx._defer_class_schema and _class_has_pending_update_class(target.type):
        # This class will be transformed by an UpdateClass effect that has not
        # been applied yet. Stay unevaluated so we retry once the schema is
        # final, rather than reading the stale pre-transformation schema.
        raise EvaluationStuck

    members: dict[str, Type] = {}

    # Iterate through MRO in reverse (base classes first) to include inherited members
    for type_info in reversed(target.type.mro):
        if _should_skip_type_info(type_info, evaluator.api):
            continue

        for name, sym in type_info.names.items():
            if sym.type is None:
                continue

            # Skip inferred attributes (those without explicit type annotations),
            # but include them for enums since enum members are always inferred.
            if isinstance(sym.node, Var) and sym.node.is_inferred and not target.type.is_enum:
                continue

            if attrs_only:
                # Attrs filters to attributes only (excludes methods).
                # Methods are FuncDef nodes; Callable-typed attributes are Var nodes.
                if isinstance(sym.node, FuncDef):
                    continue

            # Map type_info to get correct type args as seen from target
            if type_info == target.type:
                definer = target
            else:
                definer = map_instance_to_supertype(target, type_info)

            # Expand the member type to substitute type variables with actual args
            member_typ = expand_type_by_instance(sym.type, definer)

            member_type = create_member_type(
                evaluator,
                member_info.type,
                name=name,
                typ=member_typ,
                node=sym.node,
                definer=definer,
            )
            members[name] = member_type

    return members


def _get_typeddict_members_dict(
    target: TypedDictType, member_type_info: TypeInfo, *, evaluator: TypeLevelEvaluator
) -> dict[str, Type]:
    """Build a dict of member name -> Member type for a TypedDict."""
    members: dict[str, Type] = {}

    for name, item_type in target.items.items():
        # Build qualifiers for TypedDict keys
        # Required is the default, so only add NotRequired when not required
        quals: list[str] = []
        if name not in target.required_keys:
            quals.append("NotRequired")
        if name in target.readonly_keys:
            quals.append("ReadOnly")

        quals_type = UnionType.make_union([evaluator.literal_str(q) for q in quals])

        members[name] = Instance(
            member_type_info,
            [
                evaluator.literal_str(name),  # name
                item_type,  # typ
                quals_type,  # quals
                UninhabitedType(),  # init (not tracked for TypedDict)
                UninhabitedType(),  # definer (not tracked for TypedDict)
            ],
        )

    return members


def create_member_type(
    evaluator: TypeLevelEvaluator,
    member_type_info: TypeInfo,
    name: str,
    typ: Type,
    node: object,
    definer: Instance,
) -> Instance:
    """Create a Member[name, typ, quals, init, definer] instance type."""
    # Determine qualifiers
    quals: Type
    if isinstance(node, Var):
        if node.is_classvar:
            quals = evaluator.literal_str("ClassVar")
        elif node.is_final:
            quals = evaluator.literal_str("Final")
        else:
            quals = UninhabitedType()  # Never = no qualifiers
    elif isinstance(node, FuncDef):
        # Methods are class-level, so they have ClassVar qualifier
        quals = evaluator.literal_str("ClassVar")
    else:
        quals = UninhabitedType()

    # For init, use init_type when available (set during type checking for class members).
    # The literal type extraction is done in checker.py when init_type is set.
    init: Type = UninhabitedType()
    if isinstance(node, Var) and node.init_type is not None:
        init = node.init_type

    return Instance(
        member_type_info,
        [
            evaluator.literal_str(name),  # name
            typ,  # typ
            quals,  # quals
            init,  # init
            definer,  # definer
        ],
    )


@register_operator("NewTypedDict")
def _eval_new_typeddict(*args: Type, evaluator: TypeLevelEvaluator) -> Type:
    """Evaluate NewTypedDict[*Members] -> create a new TypedDict from Member types.

    This is the inverse of Members[TypedDict].
    """
    # Get the Member TypeInfo to verify arguments
    member_info = evaluator.get_typemap_type("Member")

    items: dict[str, Type] = {}
    required_keys: set[str] = set()
    readonly_keys: set[str] = set()

    for arg in args:
        arg = get_proper_type(arg)

        # Each argument should be a Member[name, typ, quals, init, definer]
        if not isinstance(arg, Instance) or arg.type != member_info.type:
            # Not a Member type - can't construct TypedDict
            return UninhabitedType()

        if len(arg.args) < 3:
            return UninhabitedType()

        # Extract name, type, and qualifiers from Member args
        name_type, item_type, quals, *_ = arg.args
        name = extract_literal_string(name_type)
        if name is None:
            return UninhabitedType()
        is_required = True  # Default is Required
        is_readonly = False

        for qual in extract_qualifier_strings(quals):
            if qual == "NotRequired":
                is_required = False
            elif qual == "Required":
                is_required = True
            elif qual == "ReadOnly":
                is_readonly = True

        items[name] = item_type
        if is_required:
            required_keys.add(name)
        if is_readonly:
            readonly_keys.add(name)

    # Get the TypedDict fallback
    fallback = evaluator.api.named_type_or_none("typing._TypedDict")
    if fallback is None:
        # Fallback to Mapping[str, object] if _TypedDict not available
        fallback = evaluator.api.named_type("builtins.dict")

    return TypedDictType(
        items=items, required_keys=required_keys, readonly_keys=readonly_keys, fallback=fallback
    )


class MemberDef(NamedTuple):
    """Extracted member definition from a Member[name, typ, quals, init, definer] type."""

    name: str
    type: Type
    init_type: Type
    is_classvar: bool
    is_final: bool


def _extract_members(
    args: tuple[Type, ...], evaluator: TypeLevelEvaluator, *, eval_types: bool = False
) -> list[MemberDef] | None:
    """Extract member definitions from Member type arguments.

    Returns a list of MemberDef, or None if any argument is not a valid Member.

    If eval_types is True, member types are eagerly evaluated via the
    evaluator (needed for UpdateClass where the types are stored on a
    real TypeInfo and must be fully resolved).
    """
    member_info = evaluator.get_typemap_type("Member")
    members: list[MemberDef] = []

    for arg in args:
        arg = get_proper_type(arg)

        if not isinstance(arg, Instance) or arg.type != member_info.type:
            return None
        if len(arg.args) < 2:
            return None

        name = extract_literal_string(arg.args[0])
        if name is None:
            return None

        item_type = arg.args[1]
        if eval_types:
            item_type = evaluator.eval_proper(item_type)

        is_classvar = False
        is_final = False
        if len(arg.args) >= 3:
            for qual in extract_qualifier_strings(arg.args[2]):
                if qual == "ClassVar":
                    is_classvar = True
                elif qual == "Final":
                    is_final = True

        init_type: Type = UninhabitedType()
        if len(arg.args) >= 4:
            init_type = arg.args[3]

        members.append(MemberDef(name, item_type, init_type, is_classvar, is_final))

    return members


_synthetic_type_counter = 0


def _build_synthetic_typeinfo(
    class_name: str, members: list[MemberDef], evaluator: TypeLevelEvaluator
) -> TypeInfo:
    """Create a synthetic TypeInfo populated with members.

    Used by NewProtocol to build a TypeInfo carrying member definitions
    extracted from Member type arguments.
    """
    global _synthetic_type_counter
    _synthetic_type_counter += 1

    # HACK: We create a ClassDef with an empty Block because TypeInfo requires one.
    # Each synthetic type needs a unique fullname so that nominal subtype checks
    # (has_base) don't incorrectly treat distinct synthetic types as related.
    class_def = ClassDef(class_name, Block([]))
    class_def.fullname = f"__typelevel__.{class_name}.{_synthetic_type_counter}"

    info = TypeInfo(SymbolTable(), class_def, "__typelevel__")
    class_def.info = info

    object_type = evaluator.api.named_type("builtins.object")
    info.bases = [object_type]
    try:
        calculate_mro(info)
    except Exception:
        # HACK: Minimal MRO setup when calculate_mro fails
        info.mro = [info, object_type.type]

    for m in members:
        var = Var(m.name, m.type)
        var.info = info
        var._fullname = f"{info.fullname}.{m.name}"
        var.is_classvar = m.is_classvar
        var.is_final = m.is_final
        var.is_initialized_in_class = True
        var.init_type = m.init_type
        var.is_inferred = False
        info.names[m.name] = SymbolTableNode(MDEF, var)

    return info


@register_operator("NewProtocol")
def _eval_new_protocol(*args: Type, evaluator: TypeLevelEvaluator) -> Type:
    """Evaluate NewProtocol[*Members] -> create a new structural protocol type.

    This creates a synthetic protocol class with members defined by the Member arguments.
    The protocol type uses structural subtyping.
    """
    # TODO: methods are probably in bad shape

    members = _extract_members(args, evaluator)
    if members is None:
        return UninhabitedType()

    info = _build_synthetic_typeinfo("NewProtocol", members, evaluator)

    info.is_protocol = True
    assert evaluator._current_op is not None
    info.new_protocol_constructor = evaluator._current_op
    info.runtime_protocol = False

    return Instance(info, [])


@register_operator("UpdateClass")
def _eval_update_class(*args: Type, evaluator: TypeLevelEvaluator) -> Type:
    """UpdateClass should not be evaluated as a normal type operator.

    It is only valid as the return type of a class decorator or
    __init_subclass__, and is handled specially by semanal_main.py
    via evaluate_update_class().  If we get here (e.g. via get_proper_type
    expanding the return type annotation), return NoneType since the
    decorated function/method semantically returns None at runtime.
    Returning Never here would cause the checker to treat subsequent
    code as unreachable.
    """
    return NoneType()


def evaluate_update_class(
    typ: TypeOperatorType, api: SemanticAnalyzerInterface, ctx: Context | None = None
) -> list[MemberDef] | None:
    """Evaluate an UpdateClass TypeOperatorType and return its member definitions.

    Called from semanal_main.py during the post-semanal pass. Eagerly evaluates
    member types so they are fully resolved before being stored on the target
    class's TypeInfo.

    Returns None if evaluation fails.
    """
    evaluator = TypeLevelEvaluator(api, ctx)
    try:
        args = evaluator.flatten_args(typ.args)
    except (EvaluationStuck, EvaluationOverflow):
        return None
    return _extract_members(tuple(args), evaluator, eval_types=True)


@register_operator("Length")
@lift_over_unions
def _eval_length(arg: Type, *, evaluator: TypeLevelEvaluator) -> Type:
    """Evaluate Length[T] -> Literal[int] for tuple length."""
    target = evaluator.eval_proper(arg)

    if isinstance(target, TupleType):
        # Need to evaluate the elements before we inspect them
        items = [evaluator.eval_proper(st) for st in target.items]

        # If there is an Unpack, it must be of an unbounded tuple, or
        # it would have been substituted out.
        if any(isinstance(st, UnpackType) for st in items):
            return NoneType()
        return evaluator.literal_int(len(target.items))
    if isinstance(target, Instance) and target.type.has_base("builtins.tuple"):
        return NoneType()

    return UninhabitedType()


@register_operator("RaiseError")
def _eval_raise_error(*args: Type, evaluator: TypeLevelEvaluator) -> Type:
    """Evaluate RaiseError[S] -> emit a type error with message S.

    RaiseError is used to emit custom type errors during type-level computation.
    The argument must be a Literal[str] containing the error message.
    Returns Never after emitting the error.
    """

    if not args:
        msg = "RaiseError called without arguments!"
    else:
        msg = extract_literal_string(args[0]) or str(args[0])

    if args[1:]:
        msg += ": " + ", ".join(str(t) for t in args[1:])

    # TODO: We could also print a stack trace?
    # Use serious=True to bypass in_checked_function() check which requires
    # self.options to be set on the SemanticAnalyzer
    if not typelevel_ctx._suppress_errors:
        evaluator.api.fail(msg, evaluator.error_ctx, serious=True)

    return UninhabitedType()


def evaluate_comprehension(evaluator: TypeLevelEvaluator, typ: TypeForComprehension) -> Type:
    """Evaluate a TypeForComprehension.

    Evaluates *[Expr for var in Iter if Cond] to UnpackType(TupleType([...])).
    """

    # Get the iterable type and expand it to a TupleType
    iter_proper = evaluator.eval_proper(typ.iter_type)

    if not isinstance(iter_proper, TupleType):
        # Can only iterate over tuple types
        return UninhabitedType()

    # Process each item in the tuple
    result_items: list[Type] = []
    assert typ.iter_var
    for item in iter_proper.items:
        # Substitute iter_var with item in element_expr and conditions
        env = {typ.iter_var.id: item}
        substituted_expr = expand_type(typ.element_expr, env)
        substituted_conditions = [expand_type(cond, env) for cond in typ.conditions]

        # Evaluate all conditions
        all_pass = True
        for cond in substituted_conditions:
            cond_result = extract_literal_bool(evaluator.evaluate(cond))
            if cond_result is False:
                all_pass = False
                break
            elif cond_result is None:
                # Undecidable condition - raise Stuck
                raise EvaluationStuck

        if all_pass:
            # Include this element in the result
            result_items.append(substituted_expr)

    return UnpackType(evaluator.tuple_type(result_items))


# --- Helper Functions ---


def get_type_args_for_base(instance: Instance, base_type: TypeInfo) -> tuple[Type, ...] | None:
    """Get type args when viewing instance as base class.

    Returns None if instance is not a subtype of base_type.
    """
    # Check if base_type is in the MRO. (map_instance_to_supertype
    # doesn't have a way to signal when it isn't; it just fills the
    # type with Anys)
    if base_type not in instance.type.mro:
        return None

    return map_instance_to_supertype(instance, base_type).args


# --- Public API ---


def evaluate_computed_type(typ: ComputedType, ctx: Context | None = None) -> Type:
    """Evaluate a ComputedType. Called from ComputedType.expand().

    Uses typelevel_ctx.api to access the semantic analyzer.

    The ctx argument indicates where an error message from RaiseError
    ought to be placed.  TODO: Make it a stack of contexts maybe?

    """
    if typelevel_ctx.api is None:
        raise AssertionError("No access to semantic analyzer!")

    old_evaluator = typelevel_ctx._evaluator
    if not typelevel_ctx._evaluator:
        typelevel_ctx._evaluator = TypeLevelEvaluator(typelevel_ctx.api, ctx)
    try:
        res = typelevel_ctx._evaluator.evaluate(typ)
    except EvaluationOverflow:
        # If this is not the top level of type evaluation, re-raise.
        if old_evaluator is not None:
            raise
        res = EXPANSION_OVERFLOW
    except EvaluationStuck:
        # TODO: Should we do the same top level thing as above?
        res = EXPANSION_ANY
    finally:
        typelevel_ctx._evaluator = old_evaluator

    # print("EVALED!!", typ, "====>", res)
    return res
