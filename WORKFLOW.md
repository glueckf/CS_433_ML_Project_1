# How we work in this repo

Everything here exists for one reason: three people should be able to work in
parallel for five weeks and still have a repository that merges cleanly and
reproduces its own results on the last day.

You do not need to understand all of it up front. Read "Setup" and "The loop",
then come back to the rest when something blocks you.

---

## Setup (once)

You need Python 3.11 and `git`.

```bash
git clone https://github.com/glueckf/CS_433_ML_Project_1.git
cd CS_433_ML_Project_1
make setup
source .venv/bin/activate
```

`make setup` creates a virtual environment in `.venv/`, installs the exact
package versions everyone else is using, and installs the git hooks.

**Always work with the environment activated.** If `python` is not the one in
`.venv/`, you are running against different package versions than CI, and
results will not match. Run `source .venv/bin/activate` in every new terminal.

### The data

The dataset is 327 MB and is never committed — `x_train.csv` alone is more
than twice GitHub's hard limit for a single file. Download the three files
from [AIcrowd](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1)
into `data/` and keep the names exactly as they are (`x_train.csv`,
`x_test.csv`, `y_train.csv`). Then:

```bash
make check-data
```

This compares your files against the checksums in `data/MANIFEST.sha256`. If
it fails, your download is incomplete or a different version, and every number
you produce will silently disagree with everyone else's. Fix it before you
start.

---

## The loop (every change)

```bash
git switch main
git pull                          # always start from the current main
git switch -c feat/impute-missing # your own branch

# ... do the work ...

make check                        # same checks CI runs; fix anything red
git add -p                        # review your own diff before committing
git commit -m "Add median imputation for missing values"
git push -u origin feat/impute-missing
```

Then open a pull request on GitHub and ask for a review.

A few habits that prevent almost all of the pain:

- **One idea per branch, merged within a few days.** A branch that lives for
  two weeks will conflict with everything. If your work is getting big, split
  it and merge the first half.
- **Pull `main` often**, even mid-branch: `git switch main && git pull && git
  switch - && git rebase main`. Conflicts you resolve today are small;
  conflicts you resolve in three weeks are not.
- **Run `make check` before you push.** It is the same thing CI runs, so a
  green local run means a green pull request.
- **Never commit to `main` directly.** It is protected, so the push will be
  rejected anyway.

### Branch names

`feat/` for new work, `fix/` for corrections, `exp/` for an experiment you may
well throw away. Short and descriptive after that: `feat/one-hot-encoding`,
`fix/sigmoid-overflow`, `exp/polynomial-degree-3`.

### Reviews

Every pull request needs one approval before it can merge. A review is not a
formality — the point is that at least two of us understand every piece of the
pipeline, because we all have to defend it in the report.

We merge with **squash merge**, so each pull request becomes a single commit on
`main` and the history stays readable.

---

## The rules, and why they exist

These are checked automatically. Each one exists because breaking it costs
real marks or real days.

| Rule | Why |
| --- | --- |
| Only the standard library and NumPy on the graded path | The project description forbids everything else. matplotlib is allowed for plotting only, so it may be imported in `mlproject/plots.py` and `scripts/` and nowhere else. |
| The six functions in `implementations.py` keep their exact signatures | The graders import that file and call them by name. A renamed parameter fails the grading. |
| `implementations.py` imports nothing of ours | The graders load it as a standalone file. An import of `mlproject` would break it. |
| No file may contain the word "t-o-d-o" | The official grading test greps every `.py` file for it and fails. Write `*** YOUR CODE HERE ***` instead. |
| Notebooks contain no `def` and no `class` | Logic trapped in a notebook cannot be tested, reviewed, or reused. See `notebooks/README.md`. |
| No file over 5 MB may be committed | Stops a dataset csv or a submission file from entering the history, which is painful to undo. |
| Dependency pins change in one place, with review | Three people bumping NumPy independently is exactly how branches become unmergeable. |

### Where code goes

- **`implementations.py`** — the six graded methods, and nothing else.
- **`mlproject/`** — everything reusable: loading, cleaning, features,
  metrics, cross-validation, plotting. If two people would want it, it goes
  here, with a docstring.
- **`scripts/`** — things you run to produce an artefact, such as the report
  figures. Not imported by anything.
- **`notebooks/`** — exploration, in your own folder. Nothing else may depend
  on a notebook; if we deleted the whole folder, `run.py` would still work.
- **`tests/`** — anything worth not breaking twice.

The rough division of labour follows the module boundaries, so that two people
rarely edit the same file on the same day.

---

## When CI goes red

The pull request page shows which check failed. In order of likelihood:

**`ruff format --check` failed.** Formatting only. Run `make format` and commit
the result.

**`ruff check` failed.** A lint rule: an unused import, a missing docstring, a
line over 88 characters. `make format` fixes most of them automatically; the
rest are named in the log with a file and line number.

**`pytest` failed.** Read the assertion message — the checks in
`tests/test_spec_compliance.py` are written to say what to do about it.

You can always reproduce CI exactly with `make check`.

### The official grading tests

EPFL publishes public tests that check our signatures. They ask not to copy
them into the repo, because they update them, so we pull a fresh copy instead:

```bash
make grading-tests
```

Run this before every submission, not daily.

---

## Submitting to AIcrowd

We get **five submissions per day across the whole team**, so coordinate in the
group chat before uploading. The leaderboard is a sanity check, never our
evaluation: tuning against it overfits the public split and burns the budget.
Decide with local cross-validation, then confirm on AIcrowd.

Note that about 8.8% of labels are positive, so accuracy alone is misleading —
predicting "no" for everyone already scores about 0.91 with an F1 of zero.
Always report accuracy and F1 together.

---

## Repository settings

For reference, `main` is protected with: pull request required, one approving
review, review dismissed on new pushes, code owner review required, CI checks
required to pass, linear history, no force pushes.

`.github/CODEOWNERS` controls who is asked to review what. It gets updated as
the team joins.
