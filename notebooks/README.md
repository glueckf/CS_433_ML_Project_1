# Notebooks

Exploration lives here: looking at distributions, sanity-checking an idea,
trying a model out before it is worth keeping.

Two rules, both checked automatically:

1. **Work in your own folder.** Create `notebooks/<your-name>/` and stay in it.
   Two people editing one notebook produces a merge conflict that is
   effectively impossible to resolve, because notebooks are JSON.
2. **No `def` and no `class` in a committed notebook.** The moment you write a
   function, it moves into `mlproject/` and the notebook imports it back.

Outputs are stripped automatically when you commit, so notebooks stay small
and reviewable. That also means figures for the report are not produced here:
those come from `scripts/make_figures.py`, so anyone can regenerate them.

See [WORKFLOW.md](../WORKFLOW.md) for the reasoning.
