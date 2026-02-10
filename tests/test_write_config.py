"""Tests for the write_config and dump_config utility functions."""

import json
from pathlib import Path

import yaml

FIXTURES = Path(__file__).parent / "fixtures"


class TestWriteConfig:
    """Test write_config with files."""

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


class TestDumpConfig:
    """Test dump_config serialization."""

    def test_dump_yaml(self):
        from uniaf3.schema import UniAF3Config, dump_config

        conf = UniAF3Config.from_file(FIXTURES / "uniaf3_example.yaml")
        text = dump_config(conf, fmt="yaml")
        parsed = yaml.safe_load(text)
        assert "sequences" in parsed

    def test_dump_json(self):
        from uniaf3.schema import UniAF3Config, dump_config

        conf = UniAF3Config.from_file(FIXTURES / "uniaf3_example.yaml")
        text = dump_config(conf, fmt="json")
        parsed = json.loads(text)
        assert "sequences" in parsed

    def test_dump_model_config_yaml(self, tmp_path):
        from uniaf3.adapters import to_boltz
        from uniaf3.schema import UniAF3Config, dump_config

        conf = UniAF3Config.from_file(FIXTURES / "uniaf3_example.yaml")
        boltz = to_boltz(conf, msa_dir=tmp_path, strict=False)
        text = dump_config(boltz, fmt="yaml")
        parsed = yaml.safe_load(text)
        assert "sequences" in parsed
        assert len(parsed["sequences"]) > 0

    def test_dump_model_config_json(self, tmp_path):
        from uniaf3.adapters import to_boltz
        from uniaf3.schema import UniAF3Config, dump_config

        conf = UniAF3Config.from_file(FIXTURES / "uniaf3_example.yaml")
        boltz = to_boltz(conf, msa_dir=tmp_path, strict=False)
        text = dump_config(boltz, fmt="json")
        parsed = json.loads(text)
        assert "sequences" in parsed
        assert len(parsed["sequences"]) > 0
