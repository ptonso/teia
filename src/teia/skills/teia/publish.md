# Get a plugin ready to publish

A published plugin is a promise: anyone who installs Teia and the plugin reruns the project's results. That holds only when everything the plugin needs is released, pinned and recorded. This is the last step of a project's life before it goes public, and it changes little code. Mostly it moves what the project borrowed into a permanent home and writes down versions.

## Steps

1. **Run `teia plugin check --publish`.** It turns two warnings into errors: a missing `LICENSE`, and every `teia.*` module the plugin uses that does not come from the `pyteia` distribution (`provider`). Each `provider` finding is a component borrowed from an unpublished package. Show the list to the user: each one must move to the plugin, into a published package, or into Teia, and that decision belongs to the user and the owner of that package.
2. **Pin every dependency.** In `pyproject.toml`, depend on released versions with a lower bound (`pyteia>=0.4`) and on unreleased repositories by tag or commit (`name @ git+<repository>@<tag>`). Remove local paths and editable references.
3. **Record the baselines.** For each baseline: its upstream commit, the way it ran, and the check that shows it runs correctly ([baselines](baselines.md#show-the-baseline-runs-correctly)).
4. **Keep the run scripts.** The exact `teia train` command line of every result in the paper, in a script in the project.
5. **Rerun from a clean environment.** Create a new virtual environment, install only the plugin (`pip install .`, which pulls its pinned dependencies), and rerun one result end to end. A result that only runs in the development environment is not yet publishable.

## Report

The final `teia plugin check --publish` output, the dependencies pinned, any `provider` finding still open and the decision it needs, and the clean-environment rerun with its result.
