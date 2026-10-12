"""Deterministic fixture context and tools. No network, no side effects, no real user data."""
from __future__ import annotations

FIXTURE_CITY = "Fixture City"

DEFAULT_CONTEXT = {
    "city": FIXTURE_CITY,
    "forecast_tomorrow": "light rain, 14C, 70% chance of showers in the morning",
    "calendar": [
        {"time": "09:30", "title": "Design review", "location": "Office"},
        {"time": "14:00", "title": "Dentist", "location": "Clinic"},
    ],
    "notes": ["Send the routing-config patch", "Book the train for Friday"],
    "modules": ["weather", "calendar", "commute"],
    "commute_minutes": {"Office": 35, "Clinic": 20},
}


def get_user_context(ctx: dict) -> dict:
    return {k: ctx[k] for k in ("city", "forecast_tomorrow", "calendar", "notes", "modules") if k in ctx}


def get_commute_time(ctx: dict) -> dict:
    dest = ctx.get("destination")
    if not dest:
        raise ValueError("destination required")
    minutes = ctx.get("commute_minutes", {}).get(dest)
    if minutes is None:
        return {"destination": dest, "minutes": None, "note": "unknown destination"}
    return {"destination": dest, "minutes": minutes}


def build_module(ctx: dict) -> dict:
    name = ctx.get("module_name", "new_module")
    return {"module": name, "built": True, "files": [f"modules/{name}/__init__.py"]}


def validate_module(ctx: dict) -> dict:
    name = ctx.get("module_name", "new_module")
    return {"module": name, "valid": True, "checks": ["imports", "schema"]}


def list_modules(ctx: dict) -> dict:
    return {"modules": list(ctx.get("modules", []))}


TOOLS = {
    "get_user_context": get_user_context,
    "get_commute_time": get_commute_time,
    "build_module": build_module,
    "validate_module": validate_module,
    "list_modules": list_modules,
}


def run_tools(names: list[str], ctx: dict) -> dict:
    """Run the required tool sequence in order. Missing tool is a programming error, not a
    runtime disposition: policy must have checked availability first."""
    merged = dict(DEFAULT_CONTEXT)
    merged.update(ctx)
    return {name: TOOLS[name](merged) for name in names}
