import pytest

from kamiru.core.device import arch_status


class TestArchStatus:
    def test_native_exact(self):
        # 5070 Ti (sm_120) con wheels cu128 que traen sm_120
        assert arch_status((12, 0), ["sm_80", "sm_90", "sm_100", "sm_120"]) == "native"

    def test_native_same_major_lower_minor(self):
        # GPU 8.9 (Ada) corre SASS sm_86? no — pero sm_89 sí; misma familia
        assert arch_status((8, 9), ["sm_80", "sm_89"]) == "native"

    def test_jit_only_ptx(self):
        # wheels cu126 sin sm_120: solo PTX viejo → JIT eterno (el "cuelgue")
        assert arch_status((12, 0), ["sm_80", "sm_90", "compute_90"]) == "jit"

    def test_unsupported(self):
        # ni SASS ni PTX compatibles
        assert arch_status((12, 0), []) == "unsupported"

    def test_old_gpu_native(self):
        assert arch_status((8, 6), ["sm_80", "sm_86", "compute_90"]) == "native"

    def test_malformed_entries_ignored(self):
        assert arch_status((12, 0), ["garbage", "sm_120"]) == "native"


class TestEnsureModelOffline:
    def test_offline_cache_hit_no_network(self, monkeypatch, tmp_path):
        """Con cache completo no debe haber NINGUNA llamada de red."""
        import huggingface_hub

        from kamiru.core import model_fetch

        monkeypatch.setenv("KAMIRU_HOME", str(tmp_path))
        calls = {"local_only": 0}

        def fake_snapshot(repo, local_files_only=False, **kw):
            assert local_files_only, "el primer intento debe ser offline"
            calls["local_only"] += 1
            return "/cache/snapshot"

        def boom(*a, **kw):
            raise AssertionError("no debe tocarse la red con cache completo")

        monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
        monkeypatch.setattr(huggingface_hub, "HfApi", boom)
        msgs = []
        path = model_fetch.ensure_model("fake/repo", progress=msgs.append)
        assert path == "/cache/snapshot"
        assert calls["local_only"] == 1
        assert any("sin descarga ni red" in m for m in msgs)

    def test_incomplete_cache_falls_to_online(self, monkeypatch, tmp_path):
        """Cache incompleto → camino online (con timeout)."""
        import huggingface_hub

        from kamiru.core import model_fetch

        monkeypatch.setenv("KAMIRU_HOME", str(tmp_path))

        def fake_snapshot(repo, local_files_only=False, **kw):
            raise FileNotFoundError("cache incompleto")

        class FakeSibling:
            def __init__(self, name, size):
                self.rfilename, self.size = name, size

        class FakeInfo:
            sha = "abc123"
            siblings = [FakeSibling("config.json", 100),
                        FakeSibling("model.safetensors", 1000)]

        downloaded = []

        class FakeApi:
            def __init__(self, token=None):
                pass

            def model_info(self, repo, files_metadata=False, timeout=None):
                assert timeout is not None, "la consulta a HF debe llevar timeout"
                return FakeInfo()

        def fake_hub_download(repo, name, revision=None, token=None, etag_timeout=None):
            downloaded.append(name)
            return str(tmp_path / "snap" / name)

        monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
        monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hub_download)
        monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache",
                            lambda repo, name, revision=None: None)
        msgs = []
        path = model_fetch.ensure_model("fake/repo", progress=msgs.append)
        assert path == str(tmp_path / "snap")
        assert "config.json" in downloaded and "model.safetensors" in downloaded
        assert any("2 archivo(s)" in m for m in msgs)
