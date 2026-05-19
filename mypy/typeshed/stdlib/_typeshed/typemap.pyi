"""Declarations from the typemap PEP proposal.

These are here so that we can also easily export them from typing_extensions
and typemap.typing.
"""

import typing_extensions
from typing import Any, Generic, Literal, TypeVar, TypedDict
from typing_extensions import TypeVarTuple, Unpack, Never

class BaseTypedDict(TypedDict):
    pass


_S = TypeVar("_S")
_T = TypeVar("_T")

_KwargDict = TypeVar('_KwargDict', bound=BaseTypedDict)

# Inherit from Any to allow the assignments.
# TODO: Should we do this in a more principled way?
class InitField(Generic[_KwargDict], Any):
    def __init__(self, **kwargs: Unpack[_KwargDict]) -> None:
        ...

    def _get_kwargs(self) -> _KwargDict:
        ...

# Marker decorator for type operators. Classes decorated with this are treated
# specially by the type checker as type-level computation operators.
def _type_operator(cls: type[_T]) -> type[_T]: ...

# MemberQuals: qualifiers that can apply to a Member
MemberQuals: typing_extensions.TypeAlias = Literal["ClassVar", "Final", "Required", "NotRequired", "ReadOnly"]

# ParamKind: kind markers that can apply to a Param
ParamKind: typing_extensions.TypeAlias = Literal["positional", "keyword", "*", "**"]

# --- Data Types (used in type computations) ---

_Name = TypeVar("_Name")
_Type = TypeVar("_Type")
_Quals = TypeVar("_Quals", default=Never)
_Kind = TypeVar("_Kind", default=Never)
_Init = TypeVar("_Init", default=Never)
_Definer = TypeVar("_Definer", default=Never)
_Default = TypeVar("_Default", default=Never)

class Member(Generic[_Name, _Type, _Quals, _Init, _Definer]):
    """
    Represents a class member with name, type, qualifiers, initializer, and definer.
    - _Name: Literal[str] - the member name
    - _Type: the member's type
    - _Quals: Literal['ClassVar'] | Literal['Final'] | Never - qualifiers
    - _Init: the literal type of the initializer expression
    - _Definer: the class that defined this member
    """

    name: _Name
    type: _Type
    quals: _Quals
    init: _Init
    definer: _Definer

class Param(Generic[_Name, _Type, _Kind, _Default]):
    """
    Represents a function parameter for extended callable syntax.
    - _Name: Literal[str] | None - the parameter name
    - _Type: the parameter's type
    - _Kind: Literal['positional', 'keyword', '*', '**'] - the parameter kind
    - _Default: the parameter's default type (Never if no default)
    """

    name: _Name
    type: _Type
    kind: _Kind
    default: _Default


_N = TypeVar("_N", bound=str)

# Convenience aliases for Param

# XXX: For mysterious reasons, if I mark this as `:
# typing_extensions.TypeAlias`, mypy thinks _N and _T are unbound...
PosParam = Param[None, _T, Literal["positional"]]
PosDefaultParam = Param[None, _T, Literal["positional"], _T]
DefaultParam = Param[_N, _T, Never, _T]
NamedParam = Param[_N, _T, Literal["keyword"]]
NamedDefaultParam = Param[_N, _T, Literal["keyword"], _T]
ArgsParam = Param[None, _T, Literal["*"]]
KwargsParam = Param[None, _T, Literal["**"]]

class Params(Generic[Unpack[_Ts]]):
    """
    Wraps a sequence of Param types as the first argument to Callable
    to distinguish extended callable format from the standard format.
    """

    ...

# --- Type Introspection Operators ---

_Base = TypeVar("_Base")
_Idx = TypeVar("_Idx")
_S1 = TypeVar("_S1")
_S2 = TypeVar("_S2")
_Start = TypeVar("_Start")
_End = TypeVar("_End")

@_type_operator
class GetArg(Generic[_T, _Base, _Idx]):
    """
    Get type argument at index _Idx from _T when viewed as _Base.
    Returns Never if _T does not inherit from _Base or index is out of bounds.
    """

    ...

@_type_operator
class GetArgs(Generic[_T, _Base]):
    """
    Get all type arguments from _T when viewed as _Base, as a tuple.
    Returns Never if _T does not inherit from _Base.
    """

    ...

@_type_operator
class GetMember(Generic[_T, _Name]):
    """
    Get the Member type for attribute _Name from type _T.
    _Name must be a Literal[str].
    Returns Never if the member does not exist.
    """

    ...

@_type_operator
class GetMemberType(Generic[_T, _Name]):
    """
    Get the type of attribute _Name from type _T.
    _Name must be a Literal[str].
    """

    ...

@_type_operator
class Members(Generic[_T]):
    """
    Get all members of type _T as a tuple of Member types.
    Includes methods, class variables, and instance attributes.
    """

    ...

@_type_operator
class Attrs(Generic[_T]):
    """
    Get annotated instance attributes of _T as a tuple of Member types.
    Excludes methods and ClassVar members.
    """

    ...

@_type_operator
class FromUnion(Generic[_T]):
    """
    Convert a union type to a tuple of its constituent types.
    If _T is not a union, returns a 1-tuple containing _T.
    """

    ...

# --- Member/Param Accessors (defined as type aliases using GetMemberType) ---

# _MP = TypeVar("_MP", bound=Member[Any, Any, Any, Any, Any] | Param[Any, Any, Any])
# _M = TypeVar("_M", bound=Member[Any, Any, Any, Any, Any])

_MP = TypeVar("_MP")
_M = TypeVar("_M")
_P = TypeVar("_P")


GetName = GetMemberType[_MP, Literal["name"]]
GetType = GetMemberType[_MP, Literal["type"]]
GetQuals = GetMemberType[_M, Literal["quals"]]
GetKind = GetMemberType[_P, Literal["kind"]]
GetInit = GetMemberType[_M, Literal["init"]]
GetDefiner = GetMemberType[_M, Literal["definer"]]
GetDefault = GetMemberType[_P, Literal["default"]]

# --- Type Construction Operators ---

_Ts = TypeVarTuple("_Ts")

@_type_operator
class NewProtocol(Generic[Unpack[_Ts]]):
    """
    Construct a new structural (protocol) type from Member types.
    NewProtocol[Member[...], Member[...], ...] creates an anonymous protocol.
    """

    ...

@_type_operator
class NewTypedDict(Generic[Unpack[_Ts]]):
    """
    Construct a new TypedDict from Member types.
    NewTypedDict[Member[...], Member[...], ...] creates an anonymous TypedDict.
    """

    ...

@_type_operator
class UpdateClass(Generic[Unpack[_Ts]]):
    """
    Update an existing class with new members.
    Can only be used as the return type of a class decorator or __init_subclass__.
    Members with type Never are removed from the class.
    """

    ...

@_type_operator
class _NewUnion(Generic[Unpack[_Ts]]):
    """
    Construct a union type from the given type arguments.
    _NewUnion[int, str, bool] evaluates to int | str | bool.
    """

    ...

@_type_operator
class _NewCallable(Generic[Unpack[_Ts]]):
    """
    Construct a callable type from Param types and a return type.
    _NewCallable[Param[...], ..., ReturnType] evaluates to a Callable.
    """

    ...

# --- Boolean/Conditional Operators ---

@_type_operator
class IsAssignable(Generic[_T, _Base]):
    """
    Type-level assignability check. Evaluates to a type-level boolean.
    Used in conditional type expressions: `Foo if IsAssignable[T, Base] else Bar`
    """

    ...

@_type_operator
class IsEquivalent(Generic[_T, _S]):
    """
    Type equivalence check. Returns Literal[True] if T is a subtype of S
    AND S is a subtype of T.
    Equivalent to: IsAssignable[T, S] and IsAssignable[S, T]
    """

    ...

@_type_operator
class Bool(Generic[_T]):
    """
    Check if T contains Literal[True].
    Returns Literal[True] if T is Literal[True] or a union containing it.
    Equivalent to: IsAssignable[Literal[True], T] and not IsAssignable[T, Never]
    """

    ...

@_type_operator
class Iter(Generic[_T]):
    """
    Marks a type for iteration in type comprehensions.
    `for x in Iter[T]` iterates over elements of tuple type T.
    """

    ...

# --- String Operations ---

@_type_operator
class Slice(Generic[_S, _Start, _End]):
    """
    Slice a literal string type.
    Slice[Literal["hello"], Literal[1], Literal[3]] = Literal["el"]
    """

    ...

@_type_operator
class Concat(Generic[_S1, _S2]):
    """
    Concatenate two literal string types.
    Concat[Literal["hello"], Literal["world"]] = Literal["helloworld"]
    """

    ...

@_type_operator
class Uppercase(Generic[_S]):
    """Convert literal string to uppercase."""

    ...

@_type_operator
class Lowercase(Generic[_S]):
    """Convert literal string to lowercase."""

    ...

@_type_operator
class Capitalize(Generic[_S]):
    """Capitalize first character of literal string."""

    ...

@_type_operator
class Uncapitalize(Generic[_S]):
    """Lowercase first character of literal string."""

    ...

# --- Annotated Operations ---

@_type_operator
class GetAnnotations(Generic[_T]):
    """
    Extract Annotated metadata from a type.
    GetAnnotations[Annotated[int, 'foo', 'bar']] = Literal['foo', 'bar']
    GetAnnotations[int] = Never
    """

    ...

@_type_operator
class DropAnnotations(Generic[_T]):
    """
    Strip Annotated wrapper from a type.
    DropAnnotations[Annotated[int, 'foo']] = int
    DropAnnotations[int] = int
    """

    ...

# --- Utility Operators ---

@_type_operator
class Length(Generic[_T]):
    """
    Get the length of a tuple type as a Literal[int].
    Returns Literal[None] for unbounded tuples.
    """

    ...

@_type_operator
class RaiseError(Generic[_S, Unpack[_Ts]]):
    """
    Emit a type error with the given message.
    RaiseError[Literal["error message"]] emits the error and returns Never.
    """

    ...
