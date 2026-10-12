import tomllib

from scripts.repair_codex_paseo_hooks import disable_edits, paseo_hooks


def hook(home, **overrides):
    value = dict(key=str(home / "hooks.json") + ":stop:0:0", eventName="stop",
                 source="user", sourcePath=str(home / "hooks.json"),
                 isManaged=False, pluginId=None, enabled=True,
                 command='powershell.exe -File "paseo-codex-windows.ps1" -HookEvent Stop')
    return value | overrides


def test_selection_preserves_other_sources_and_deduplicates(tmp_path):
    target = hook(tmp_path)
    unrelated = [hook(tmp_path, source="project"), hook(tmp_path, isManaged=True),
                 hook(tmp_path, pluginId="chrome"), hook(tmp_path, command="check-policy"),
                 hook(tmp_path, eventName="sessionStart"),
                 hook(tmp_path, sourcePath=str(tmp_path / "other.json"))]
    assert paseo_hooks([{"hooks": [target] + unrelated}, {"hooks": [target]}], tmp_path) == [target]


def test_accepts_legacy_paseo_command(tmp_path):
    target = hook(tmp_path, command='if defined PASEO_TERMINAL_ID (paseo hooks codex Stop)')
    assert paseo_hooks([{"hooks": [target]}], tmp_path) == [target]


def test_disable_preserves_trust_and_quotes_windows_key():
    key = r"C:\Users\a.b\.codex\hooks.json:stop:0:0"
    state = {key: {"trusted_hash": "sha256:original", "enabled": True}, "other": {"enabled": True}}
    edits = disable_edits([{"key": key, "enabled": True}], state)
    assert edits[0]["value"] == {"trusted_hash": "sha256:original", "enabled": False}
    assert edits[0]["mergeStrategy"] == "replace"
    quoted = edits[0]["keyPath"].removeprefix("hooks.state.")
    assert tomllib.loads(f"[{quoted}]\nenabled=false")[key] == {"enabled": False}
    assert state[key]["enabled"] is True
    assert state["other"]["enabled"] is True


def test_second_apply_has_no_edits(tmp_path):
    target = hook(tmp_path, enabled=False)
    assert disable_edits([target], {}) == []
