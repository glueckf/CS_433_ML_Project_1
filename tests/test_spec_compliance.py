"""Mechanical checks on the rules from the project description.

These are the rules that are easy to break without noticing and expensive to
discover on the day of the deadline, so they run on every pull request:

* the six graded functions exist, with the exact required signatures
* nothing on the graded path imports a forbidden library
* no leftover placeholder marker survives into the final submission
* notebooks stay thin, so that no logic is trapped inside one

Some of these mirror checks the official grader performs. When they disagree,
the official grader wins; open an issue and we fix this file.
"""

import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# The exact signatures from Table 1 of the project description. Parameter names
# and their order are part of the contract: the graders call these positionally
# and by keyword.
REQUIRED_SIGNATURES = {
    "mean_squared_error_gd": ["y", "tx", "initial_w", "max_iters", "gamma"],
    "mean_squared_error_sgd": ["y", "tx", "initial_w", "max_iters", "gamma"],
    "least_squares": ["y", "tx"],
    "ridge_regression": ["y", "tx", "lambda_"],
    "logistic_regression": ["y", "tx", "initial_w", "max_iters", "gamma"],
    "reg_logistic_regression": [
        "y",
        "tx",
        "lambda_",
        "initial_w",
        "max_iters",
        "gamma",
    ],
}

# Modules that are ours rather than third-party dependencies.
LOCAL_MODULES = {"mlproject", "helpers", "implementations", "conftest"}

# Visualisation is permitted for visualisation only, so these are confined to
# the plotting module and the figure scripts.
VISUALISATION_MODULES = {"matplotlib", "seaborn"}


def python_files() -> list[Path]:
    """Return every Python file in the repo, skipping the virtualenv."""
    skip = {".venv", "venv", "grading_tests", "github_workdir", "__pycache__"}
    return [
        path
        for path in ROOT.rglob("*.py")
        if not any(part in skip for part in path.parts)
    ]


def imported_roots(path: Path) -> set[str]:
    """Return the top-level names of every module imported by a file."""
    tree = ast.parse(path.read_text(), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def allowed_roots(path: Path) -> set[str]:
    """Return the module names a given file is permitted to import."""
    import sys

    allowed = set(sys.stdlib_module_names) | {"numpy"}
    relative = path.relative_to(ROOT)

    # implementations.py is loaded standalone by the graders, so it must not
    # depend on anything of ours either.
    if relative == Path("implementations.py"):
        return allowed

    allowed |= LOCAL_MODULES
    if relative == Path("mlproject/plots.py") or relative.parts[0] == "scripts":
        allowed |= VISUALISATION_MODULES
    if relative.parts[0] == "tests":
        allowed |= {"pytest"}
    return allowed


def notebooks() -> list[Path]:
    """Return every committed notebook."""
    skip = {".ipynb_checkpoints", ".venv", "grading_tests"}
    return [
        path
        for path in ROOT.rglob("*.ipynb")
        if not any(part in skip for part in path.parts)
    ]


@pytest.mark.parametrize("filename", ["README.md", "implementations.py", "run.py"])
def test_required_file_exists_at_root(filename: str) -> None:
    """The grader checks for these three files at the repository root."""
    assert (ROOT / filename).exists(), f"Missing required file {filename}."


@pytest.mark.parametrize(
    ("name", "expected_params"), sorted(REQUIRED_SIGNATURES.items())
)
def test_required_signature(name: str, expected_params: list[str]) -> None:
    """Each graded function exists with the exact required parameter list."""
    import inspect

    import implementations

    assert hasattr(implementations, name), f"implementations.py is missing {name}."
    params = list(inspect.signature(getattr(implementations, name)).parameters)
    assert params == expected_params, (
        f"{name} has parameters {params}, the project description requires "
        f"{expected_params}. Names and order are both part of the contract."
    )


@pytest.mark.parametrize("name", sorted(REQUIRED_SIGNATURES))
def test_required_function_has_docstring(name: str) -> None:
    """The grader fails a graded function that has no docstring."""
    import implementations

    assert getattr(implementations, name).__doc__, f"{name} has no docstring."


@pytest.mark.parametrize("path", python_files(), ids=str)
def test_only_allowed_imports(path: Path) -> None:
    """No file imports a library the project description forbids."""
    forbidden = imported_roots(path) - allowed_roots(path)
    assert not forbidden, (
        f"{path.relative_to(ROOT)} imports {sorted(forbidden)}. Project 1 allows "
        "the standard library and numpy only; matplotlib is confined to "
        "mlproject/plots.py and scripts/."
    )


@pytest.mark.parametrize("path", python_files(), ids=str)
def test_no_leftover_task_markers(path: Path) -> None:
    """Mirrors the grader's check for leftover task markers in any .py file.

    The grader lowercases each file and fails on the four-letter marker built
    below. It is assembled from two halves so that this very file does not
    trip the check.
    """
    marker = "to" + "do"
    assert marker not in path.read_text().lower(), (
        f"{path.relative_to(ROOT)} contains a leftover '{marker}' marker. The "
        "official grader fails on this. Use '*** YOUR CODE HERE ***' instead."
    )


@pytest.mark.parametrize("path", notebooks(), ids=str)
def test_notebooks_define_no_logic(path: Path) -> None:
    """Notebooks may call our code but must not define functions or classes.

    A function that lives only in a notebook cannot be tested, cannot be
    reused, and cannot be regenerated by a teammate. Move it into mlproject/
    and import it back into the notebook. This rule is also what keeps the
    notebook folders from quietly growing into a second, untested codebase.
    """
    document = json.loads(path.read_text())
    offenders: list[str] = []
    for cell in document.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        # Drop IPython magics and shell escapes, which are not valid Python.
        source = "\n".join(
            line
            for line in source.splitlines()
            if not line.lstrip().startswith(("%", "!", "?"))
        )
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                offenders.append(node.name)

    assert not offenders, (
        f"{path.relative_to(ROOT)} defines {sorted(offenders)}. Move them into "
        "mlproject/ and import them back into the notebook."
    )
