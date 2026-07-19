def test_load_template_injects_tokens_and_theme_script(tmp_path, monkeypatch):
    import templates

    monkeypatch.setattr(templates, "TEMPLATES_DIR", tmp_path)
    (tmp_path / "sample.html").write_text(
        "<head><!--TOKENS--><!--THEME_SCRIPT--></head>"
    )

    html = templates._load_template("sample.html")
    assert "<!--TOKENS-->" not in html
    assert "<!--THEME_SCRIPT-->" not in html
    assert "--accent-cache: #3fc9b0" in html
    assert "window.toggleTheme" in html
