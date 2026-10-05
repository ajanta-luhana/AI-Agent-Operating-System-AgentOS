from typing import Annotated, TypedDict


def merge_results(old: list[dict], new: list[dict]) -> list[dict]:
    """Parallel agents append; planner resets with [{'_reset': True}]."""
    new = new or []
    if any(isinstance(x, dict) and x.get("_reset") for x in new):
        return [x for x in new if not x.get("_reset")]
    return (old or []) + new


class State(TypedDict, total=False):
    run_id: str
    goal: str
    has_dataset: bool
    plan: list[dict]
    approved: bool
    results: Annotated[list[dict], merge_results]
    critique: str | None
    retries: int
    verified: bool
    report: str | None
