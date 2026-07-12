import naming as N


def test_clean_strips_system_reminder():
    t = "hello <system-reminder>ignore me</system-reminder> world"
    assert N.clean(t) == "hello world"


def test_clean_strips_command_and_session_wrappers():
    t = "<command-name>/x</command-name> real <session>y</session> text"
    assert "command-name" not in N.clean(t)
    assert "session" not in N.clean(t)
    assert "real" in N.clean(t) and "text" in N.clean(t)


def test_clean_strips_write_title_artifact():
    t = "do the thing Write the title in the language the user wrote in, regardless of the language of the examples above. now"
    out = N.clean(t)
    assert "Write the title in the language" not in out


def test_clean_signal_accumulates_until_threshold():
    turns = ["short one", "b" * 500, "c" * 500]
    sig = N.clean_signal(turns, min_chars=400, max_turns=3)
    # stops after crossing 400 chars; third turn not needed
    assert "c" * 500 not in sig
    assert len(sig) >= 400


def test_clean_signal_caps_send_length():
    sig = N.clean_signal(["z" * 9000], min_chars=400, max_turns=3, max_send=3000)
    assert len(sig) == 3000


def test_clean_signal_skips_tiny_turns():
    sig = N.clean_signal(["ok", "/model opus"], min_chars=400, max_turns=3)
    assert sig == ""  # nothing substantive → defer
