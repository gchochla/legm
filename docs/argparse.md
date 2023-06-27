# Class Hyperparameters

tl;dr: To automate the configuration of hyperparameters for the manager, when creating a new class that has hyperparameters you are going to want to define in different runs, each class should define (or inherit) `argparse_args`, a `Dict[str, Dict[str, Any]]` that holds the hyperparameter name and its values for `argparse` in a `**kwargs` format `argparse_args` can be a class-level variable (in which case, when inheriting and in need of modification, use `deepcopy`), a static method returning such a dictionary, etc. Prefer to keep the name in `argparse_args` the same as in the function definition, because in that way you can automate the parameter-passing. This will also allow you to use `legm.from_namespace`, a decorator that automatically parses the necessary parameters of the function, as long as you pass the namespace(s) to the new argument `init__namespace`. If the hyperparameters need to be specified by split [train, dev, test, or any naming convention you want], where the split part of the name does not appear in the actual argument of the function, we also include `legm.splitify_namespace` to properly parse the namespace for your specific split. Here's an example with a class-level variable:

```python
import argparse
from legm import from_namespace, splitify_namespace

def add_arguments(
    parser, arguments
):
    for k, v in arguments.items():
        parser.add_argument(f"--{k}", **v)


class Dataset:
    argparse_args = dict(train_a=..., dev_a=..., test_a=..., b=...)

    @from_namespace
    def __init__(self, a, b):
        self.a = a
        self.b = b
        ...


parser = argparse.ArgumentParser()
add_arguments(parser, Dataset.argparse_args)

train_ds = Dataset(init__namespace=splitify_namespace(args, "train"))
dev_ds = Dataset(init__namespace=splitify_namespace(args, "dev"))
test_ds = Dataset(init__namespace=splitify_namespace(args, "test"))
```

`argparse_args` hyperparameter has the exact split at the start of the name followed by an underscore before the actual name. Note that `add_arguments` is already provided, and you should use that instead. The reason is because `argparse_args` can also hold other metadata, which you are probably going to want to use. See [here](./metadata.md).

---

## argparse_args

To automate the process of creating appropriate command-line arguments for the scripts utilizing the modules implemented herein, every class that has hyperparameters *MUST* contain a class-level variable named `argparse_args` (exactly). This variable is at the class level because we need it to initialize the class, so it cannot be a property initialized at `__init__`, and we need to be able to inherit it from parents and modify it accordingly for each subclass. (Again, an alternative is a static method, or anything you can access without having to initialize the class).

The structure of `argparse_args` is such that we can directly use it for the `argparse.ArgumentParser.add_argument`. It is a `Dict[str, Dict[str, Any]]`, where the initial key is the name of the hyperparemeter (used for `dest` for example), and the values are the rest of the arguments in the form of kwargs, like `type`, `nargs`, etc. :

```python
class Parent:
    argparse_args = dict(
        a=dict(type=int, required=True, help="hyperparameter a, controls ..."),
        b=dict(type=int, default=0, help="hyperparameter b, used for ...")
    )

    def __init__(self, a, b):
        self.a = a
        self.b = b
```

When there is a need to modify `argparse_args` in a child (such as when we need to add a hyperparameter, subtract a hyperparameter, or we inherit from two different classes) because it is a class-level variable, modifying it in a child class modifies it in the parent dynamically as well. Therefore, other subclasses using it will get the modified version too. To avoid this, `deepcopy` the `argparse_args`:

```python
from copy import deepcopy
from path.to.classes import Parent

class Child(Parent):
    argparse_args = deepcopy(Parent.argparse_args)
    argparse_args.pop("b")
    argparse_args.update(
        dict(c=dict(type=str, default="present", choices=["present", "absent", "both"]))
    )

    def __init__(self, a, c):
        self.a = a
        self.c = c
```

With these tools, we can use simple utilities to create an argument parser:

```python

import argparse
from path.to.classes import Child, OtherClass

def add_arguments(
    parser, arguments
):
    for k, v in arguments.items():
        parser.add_argument(f"--{k}", **v)

parser = argparse.ArgumentParser()
add_arguments(parser, Child.argparse_args)
add_arguments(parser, OtherClass.argparse_args)
args = parser.parse_args()

child_inst = Child(a=args.a, c=args.c)
other = OtherClass(this_param=args.this_param, that_param=args.that_param)
```

## from_namespace

We have further automated this by assuming that the name of the hyperparameter strictly matches the name of the actual argument in the initialization. We provide a utility called `from_namespace`. `from_namespace` has been designed as a decorator for initialization functions (not just `__init__`, but also `classmethod`s; in fact, you can use it for every function), and allows you to use the extra argument `init__namespace` (not the double `_`) in whichever method it wraps, so that you can directly pass the namespace (or iterable of namespaces) as argument, and have it grab the arguments for you. Note that it also takes into account the arguments defined in parent classes of the same method, so you can just use `**kwargs` when reimplementing a method, and `from_namespace` will still grab the values. For example:

```python
import argparse
from legm import from_namespace, add_arguments  # function above

class Parent:
    argparse_args = dict(
        a=dict(type=int, required=True, help="hyperparameter a, controls ..."),
        b=dict(type=int, default=0, help="hyperparameter b, used for ...")
    )

    @from_namespace
    def __init__(self, a, b):
        self.a = a
        self.b = b

parser = argparse.ArgumentParser()
add_arguments(parser, Parent.argparse_args)

parent_inst = Parent(init__namespace=args)
```

In this manner, you don't even have to remember the names of the hyperparameters, or use endless if-the-elses if you want to create a script with many alternatives! For example, if you want to create a script for a ML model but multiple datasets with different arguments, you can create a subparser for each dataset, but once the user specifies the dataset, the script will automatically grab the correct hyperparameters:

```python
import argparse
from path.to.utils import from_namespace, add_arguments


class Dataset1:
    argparse_args = dict(a=..., b=...)

    @from_namespace
    def __init__(self, a, b):
        self.a = a
        self.b = b
        ...

class Dataset2:
    argparse_args = dict(a=..., c=..., d=...)

    @from_namespace
    def __init__(self, a, c, d):
        self.a = a
        self.c = c
        self.d = d
        ...

class Dataset3:
    argparse_args = dict(a=..., c=..., e=...)

    @from_namespace
    def __init__(self, a, c, e):
        self.a = a
        self.c = c
        self.e = e
        ...


DATASET = {
    "1": Dataset1,
    "2": Dataset2,
    "3": Dataset3,
}

parser = argparse.ArgumentParser()
sp = parser.add_subparsers(dest="dataset", required=True)
for dataset in DATASET:
    sp_task = sp.add_parser(dataset)
    add_arguments(sp_task, DATASET[dataset].argparse_args)

args = parser.parse_args()

dataset = DATASET[args.dataset](init__namespace=args)
```

## splitify_namespace

The caveat here is that we may have the different values for the hyperparameter depending on the split. A very straightforward example is the split used, which will indeed be an argument for datasets for example. To resolve this issue, we have also implemented `splitify_namespace`, which takes in namespaces and the current split, and grabs split-specific arguments and passes them to the proper argument. The `argparse_args` hyperparameter name needs to start with the split followed by the an underscore, and finally the actual name, for `splitify_namespace` to work:

```python
import argparse
from path.to.utils import from_namespace, add_arguments, splitify_namespace


class Dataset:
    argparse_args = dict(train_a=..., dev_a=..., test_a=..., b=...)

    @from_namespace
    def __init__(self, a, b):
        self.a = a
        self.b = b
        ...


parser = argparse.ArgumentParser()
add_arguments(parser, Dataset.argparse_args)

train_ds = Dataset(init__namespace=splitify_namespace(args, "train"))
dev_ds = Dataset(init__namespace=splitify_namespace(args, "dev"))
test_ds = Dataset(init__namespace=splitify_namespace(args, "test"))
```

Note that `add_arguments` is already provided as well (and was only "defined" in this file for pedagogical purposes), and you should use that instead. The reason is because `argparse_args` can also hold other metadata, which you are probably gonna want to use. See [here](./metadata.md) for more details.
