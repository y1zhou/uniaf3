"""Tests for the write_config utility function."""

import io
import json
from pathlib import Path

import yaml

FIXTURES = Path(__file__).parent / "fixtures"


class TestWriteConfig:
    """Test write_config with files and streams."""

    def test_write_yaml_file(self, tmp_path):
        from uniaf3.schema import UniAF3Config, write_config

        conf = UniAF3Config.from_file(FIXTURES / "uniaf3_example.yaml")
        out = tmp_path / "out.yaml"
        write_config(conf, out)
        assert out.exists()
        parsed = yaml.safe_load(out.read_text())
        assert "sequences" in parsed

    def test_write_json_file(self, tmp_path):
        from uniaf3.schema import UniAF3Config, write_config

        conf = UniAF3Config.from_file(FIXTURES / "uniaf3_example.yaml")
        out = tmp_path / "out.json"
        write_config(conf, out)
        assert out.exists()
        parsed = json.loads(out.read_text())
        assert "sequences" in parsed

    def test_write_text_stream(self):
        from uniaf3.schema import UniAF3Config, write_config

        conf = UniAF3Config.from_file(FIXTURES / "uniaf3_example.yaml")
        buf = io.StringIO()
        write_config(conf, stream=buf, format="yaml")
        text = buf.getvalue()
        assert "sequences" in text

    def test_write_binary_stream(self):
        from uniaf3.schema import UniAF3Config, write_config

        conf = UniAF3Config.from_file(FIXTURES / "uniaf3_example.yaml")
        buf = io.BytesIO()
        write_config(conf, stream=buf, format="json")
        data = buf.getvalue()
        assert b"sequences" in data

    def test_write_json_stream(self):
        from uniaf3.schema import UniAF3Config, write_config

        conf = UniAF3Config.from_file(FIXTURES / "uniaf3_example.yaml")
        buf = io.StringIO()
        write_config(conf, stream=buf, format="json")
        text = buf.getvalue()
        parsed = json.loads(text)
        assert "sequences" in parsed

    def test_write_model_config_to_stream(self):
        from uniaf3.adapters import to_boltz
        from uniaf3.schema import UniAF3Config, write_config

        conf = UniAF3Config.from_file(FIXTURES / "uniaf3_example.yaml")
        boltz = to_boltz(conf)
        buf = io.StringIO()
        write_config(boltz, stream=buf, format="yaml")
        text = buf.getvalue()
        parsed = yaml.safe_load(text)
        assert "sequences" in parsed
