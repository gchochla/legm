import inspect
from typing import List, Optional, Callable, Any, Dict, Tuple, Union
from types import SimpleNamespace
from argparse import Namespace


def _get_kwargs_from_params_and_namespace(
    args: Tuple[Any],
    kwargs: Dict[str, Any],
    varnames: List[str],
    defaults: Tuple[Any],
    namespace: Optional[Union[SimpleNamespace, Namespace]],
    ancestor_varnames: Optional[List[List[str]]] = None,
    ancestor_defaults: Optional[List[Tuple[Any]]] = None,
) -> Dict[str, Any]:
    """Creates a new kwargs dict for any function based on
    given parameters plus a namespace.

    Args:
        args: from *args.
        kwargs: from **kwargs.
        varnames: the names of the arguments of the function.
        defaults: the defaults of the arguments of the function.
        namespace: the namespace used to enhance kwargs with.

    Returns:
        The enhanced set of kwargs.
    """
    # TODO: might need to separate kwonly from rest

    kwargs.update({varname: arg for varname, arg in zip(varnames, args)})

    if namespace is not None:
        all_varnames = [varnames] + (ancestor_varnames or [])
        all_defaults = [defaults] + (ancestor_defaults or [])

        for i, (varnames, defaults) in enumerate(
            zip(all_varnames, all_defaults)
        ):
            strict = i == 0  # i == 0 only for arguments in method itself

            # __defaults__ is tuple with defaults for arguments that actually have it =>
            # len(__defaults__) does not equal len(varnames) necessarily =>
            # extend defaults to match varnames length, second argument in tuple is
            # whether value is actual default or dummy
            defaults = [(None, False)] * (len(varnames) - len(defaults)) + [
                (default, True) for default in defaults
            ]

            # we either collect values for arguments from
            # namespace, the enhanced kwargs, or the defaults,
            # otherwise there should be a TypeError
            new_kwargs = {}
            for i, varname in enumerate(varnames):
                default, not_dummy = defaults[i]

                # if not already provided in args, kwargs
                if varname not in kwargs:
                    # grab it from namespace
                    try:
                        new_kwargs[varname] = getattr(namespace, varname)

                    except AttributeError:
                        # if arguments of function itself and doesn't have default,
                        # then raise error
                        if not not_dummy and strict:
                            raise TypeError(f"No value for argument {varname}")

            # this will include parameters in kwargs that do not appear in signature
            kwargs.update(new_kwargs)

    return kwargs


def from_namespace(func: Callable) -> Callable:
    """Decorator that adds ability in any function to optionally accept a
    namespace/list of namespaces `init__namespace`, which can contain some of
    the arguments.

    The others will be determined from the rest of the parameters
    passed, as well as the default values. If the function is a method, then it
    also grabs the names of the arguments from all parents, even if they are not
    specified in the signature of the method itself (e.g. assumed to be in args,
    kwargs). The hierarchy is: passed parameters, namespace, default values.
    Note: pass only **kwargs in namespace, *args not supported.

    Example:
    ```
    parser = argparse.ArgumentParser()
    parser.add_argument("--test1", default=1, type=int)
    parser.add_argument("--test2", default=2, type=int)
    parser.add_argument("--test3", default=3, type=int)

    argparse_namespace = parser.parse_args(["--test1", "1"])
    simple_namespace = SimpleNamespace(test4=4, test5=5, test6=6)
    simple_namespace_default = SimpleNamespace(test2=2)

    a = b = 1

    @from_namespace
    def test_fun_1(a, b, test1, test2):
        assert a == b
        assert test1 == test2 - 1

    @from_namespace
    def test_fun_2(a, b, test3=0, test4=0, **kwargs):
        assert "test" in kwargs
        assert a == b
        assert test3 == test4 - 1

    @from_namespace
    def test_fun_3(a, b, test5, test6, c=3):
        assert a == b
        assert test5 == test6 - 1

    class TestParent:
        def __init__(self, test1, test2):
            self.test1, self.test2 = test1, test2

    class TestChild(TestParent):
        @from_namespace
        def __init__(self, a, b, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert a == b
            assert self.test1 == self.test2 - 1

        @classmethod
        def init(cls, a, b, *args, **kwargs):
            return cls(a, b, *args, **kwargs)

    class TestParentDefaultMissing:
        def __init__(self, test1=1, test2=1):
            self.test1, self.test2 = test1, test2

    class TestChildDefaultMissing(TestParentDefaultMissing):
        @from_namespace
        def __init__(self, a, b):
            super().__init__(a, b)
            assert a == b
            assert self.test1 == self.test2

    test_fun_1(a, b, init__namespace=argparse_namespace)
    test_fun_2(
        a,
        b=b,
        init__namespace=[argparse_namespace, simple_namespace],
        test=None,
    )
    test_fun_2(
        a=a,
        b=b,
        init__namespace=(simple_namespace, argparse_namespace),
        test=None,
    )
    test_fun_3(a, b, init__namespace=simple_namespace)

    TestChild(a=a, b=b, init__namespace=argparse_namespace)
    TestChild.init(a=a, b=b, init__namespace=argparse_namespace)
    TestChildDefaultMissing(a=a, b=b, init__namespace=simple_namespace_default)
    ```

    Raises:
        TypeError when some argument from the function is not provided. If
            argument from parent is not found, then we assume the user
            handles that internally and don't raise an error. This will
            result in a TypeError at runtime if the argument should have in
            fact been provided.
    """

    def expand(*args, **kwargs):
        namespace = kwargs.pop("init__namespace", None)
        if not hasattr(namespace, "__len__"):
            namespace = [namespace]
        ns_kwargs = {k: getattr(ns, k) for ns in namespace for k in dir(ns)}
        namespace = SimpleNamespace(**ns_kwargs)

        orig_spec = inspect.getfullargspec(func)

        varnames = (orig_spec.args or []) + (orig_spec.kwonlyargs or [])
        defaults = (orig_spec.defaults or ()) + (orig_spec.kwonlydefaults or ())

        ancestor_defaults = None
        ancestor_varnames = None

        if varnames[0] in ("cls", "self") and orig_spec.varkw is not None:
            class_mro = (
                args[0].__class__.__mro__
                if varnames[0] == "self"
                else args[0].__mro__
            )
            specs = [
                inspect.getfullargspec(getattr(mro, func.__name__))
                for mro in class_mro[1:]
                if hasattr(mro, func.__name__)
            ] + (
                [
                    inspect.getfullargspec(getattr(mro, "__init__"))
                    for mro in class_mro
                ]
                if varnames[0] == "cls"
                else []
            )

            ancestor_varnames = [
                (spec.args[1:] or []) + (spec.kwonlyargs or [])
                for spec in specs
            ]
            ancestor_defaults = [
                (spec.defaults or ()) + (spec.kwonlydefaults or ())
                for spec in specs
            ]

        kwargs = _get_kwargs_from_params_and_namespace(
            args,
            kwargs,
            varnames,
            defaults,
            namespace,
            ancestor_varnames,
            ancestor_defaults,
        )

        return func(**kwargs)

    return expand


def splitify_namespace(
    namespace: Union[SimpleNamespace, List[SimpleNamespace]], split: str
) -> SimpleNamespace:
    """Modifies attributes of namespaces to match the given split
    (train, dev, or test).

    Args:
        namespaces: Namespace or list of namespaces to modify.
        split: Split to modify the namespaces for.

    Returns:
        Modified namespace.
    """
    if not hasattr(namespace, "__len__"):
        namespace = [namespace]
    ns_kwargs = {k: getattr(ns, k) for ns in namespace for k in vars(ns)}
    split_kwargs = {}

    for k, v in ns_kwargs.items():
        if k.startswith(split):
            # also assume undescore -> +1
            split_kwargs[k[len(split) + 1 :]] = v
        else:
            split_kwargs[k] = v

    namespace = SimpleNamespace(**split_kwargs)
    return namespace
