import pytest


@pytest.fixture(autouse=True)
def isolated_configuration(monkeypatch, tmp_path):
    for name in (
        "TYPESAFE_API_KEY",
        "OPENROUTER_API_KEY",
        "JEV_API",
        "JEV_URL",
        "JEV_MODEL",
        "JEV_GATEWAY_URL",
        "JEV_GATEWAY_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
