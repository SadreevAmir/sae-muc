from __future__ import annotations

from sae_muc.artifacts import StageManifest


def test_manifest_absent_means_rerun(tmp_path):
    m = StageManifest(tmp_path, "generate")
    assert not m.exists()
    assert not m.should_skip()


def test_manifest_with_existing_outputs_skips(tmp_path):
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "file.parquet").write_text("x")
    m = StageManifest(tmp_path, "generate")
    m.write(outputs=["out/file.parquet"])
    assert m.should_skip()


def test_manifest_with_missing_output_reruns(tmp_path):
    m = StageManifest(tmp_path, "generate")
    m.write(outputs=["out/missing.parquet"])
    assert m.exists()
    assert not m.should_skip()


def test_manifest_extra_fields_preserved(tmp_path):
    m = StageManifest(tmp_path, "vuf")
    m.write(outputs=[], extra={"layers": [15, 16], "n_certain": 250})
    data = m.read()
    assert data["layers"] == [15, 16]
    assert data["stage"] == "vuf"
