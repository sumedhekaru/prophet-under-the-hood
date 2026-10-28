# prophet-under-the-hood

Experiments on how [Prophet](https://github.com/facebook/prophet) actually fits its model, and what that means for
forecasts. Every number in the companion article comes from a script and a results page here, and each can be rerun
on stock Prophet.

The tool behind the experiments is [prophet-batch](https://github.com/sumedhekaru/prophet-batch), which computes the
exact optimum (MAP) of Prophet's model. That lets us compare Prophet's own fit with the optimum it is aiming for.

**Status:** work in progress. The questions below are being tested; none is a finding yet.

1. Does Prophet's default fit reach the optimum of its own model, and how far is it when it doesn't?
2. When does stopping short of the optimum help the forecast, and when does it hurt?
3. Do holiday windows given in days carry into the forecast on weekly data?

## Running

```sh
uv sync
uv run python <study>/<script>.py
```

Python 3.12, prophet 1.5.0. Results pages record the machine, package versions and commit for every run.
