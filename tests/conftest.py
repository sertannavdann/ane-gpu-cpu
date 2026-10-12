import pytest

from prototype.contracts import Availability, Request, RequestClass
from prototype.generator_fake import ScriptedGenerator
from prototype.router_fake import KeywordRouter

ALL_TOOLS = frozenset({"get_user_context", "get_commute_time", "build_module", "validate_module", "list_modules"})


@pytest.fixture
def availability():
    return Availability(ALL_TOOLS, frozenset({"heavy", "standard"}))


@pytest.fixture
def generator():
    return ScriptedGenerator(token_delay_s=0.001)


@pytest.fixture
def router():
    return KeywordRouter()


def fg(id_, text, **kw):
    return Request(id_, kw.pop("session", id_), text, RequestClass.FOREGROUND, **kw)


def bg(id_, text, **kw):
    kw.setdefault("max_output_tokens", 60)
    return Request(id_, kw.pop("session", id_), text, RequestClass.BACKGROUND, **kw)
