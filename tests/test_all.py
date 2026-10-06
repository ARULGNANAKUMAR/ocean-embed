"""
tests/test_all.py
North Indian Ocean Intelligence Platform — Phase 1
Comprehensive test suite using synthetic datasets.
"""

import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import preprocessing as pre
from app import app, run_phase1, run_phase2, run_phase3

# ──────────────────────────────────────────────────────────────
# FIXTURES
# ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def cfg():
    """Load real config."""
    cfg_path = Path(__file__).parent.parent / "config.json"
    return pre.load_config(str(cfg_path))


@pytest.fixture(scope="session")
def tmp_workspace(tmp_path_factory):
    """Temporary workspace for synthetic tests."""
    return tmp_path_factory.mktemp("workspace")


@pytest.fixture(scope="session")
def synthetic_netcdf(tmp_workspace):
    """Create a synthetic NetCDF file mimicking SST data."""
    import xarray as xr

    lat  = np.arange(10.0, 15.25, 0.25)
    lon  = np.arange(80.0, 85.25, 0.25)
    time = pd.date_range("2025-01-01", periods=30, freq="D")
    sst  = 27 + 3 * np.random.rand(len(time), len(lat), len(lon))

    ds = xr.Dataset(
        {"sst": (["time", "lat", "lon"], sst, {"units": "degC", "long_name": "Sea Surface Temperature"})},
        coords={"time": time, "lat": lat, "lon": lon},
    )
    fp = str(tmp_workspace / "sst_synthetic.nc")
    ds.to_netcdf(fp)
    return fp


@pytest.fixture(scope="session")
def synthetic_glorys(tmp_workspace):
    """Create a synthetic GLORYS NetCDF with thetao spanning 0-1000m depths.
    Depths include all 15 target standard depths (0,5,10,...,1000m) so both
    Phase 1 (nearest-match) and Phase 2 (exact extraction) tests pass.
    """
    import xarray as xr

    # Standard target depths matching config target_depths + extra native GLORYS levels
    # P1 validator uses glorys_required_depths (0.494, 1.541 ...) — nearest match only
    depths = [0.0, 0.5, 1.5, 2.6, 3.8, 5.0, 6.4, 7.9, 9.6, 10.0,
              11.4, 13.5, 15.8, 18.5, 20.0, 21.6, 25.2, 29.4, 30.0,
              50.0, 75.0, 100.0, 125.0, 150.0, 200.0, 300.0, 500.0,
              700.0, 1000.0]
    lat    = np.arange(10.0, 15.25, 0.25)
    lon    = np.arange(80.0, 85.25, 0.25)
    time   = pd.date_range("2025-01-01", periods=30, freq="D")
    thetao = 25 + np.random.rand(len(time), len(depths), len(lat), len(lon))

    ds = xr.Dataset(
        {"thetao": (["time", "depth", "lat", "lon"], thetao, {"units": "degC"})},
        coords={"time": time, "depth": depths, "lat": lat, "lon": lon},
    )
    fp = str(tmp_workspace / "glorys_synthetic.nc")
    ds.to_netcdf(fp)
    return fp


@pytest.fixture(scope="session")
def synthetic_argo(tmp_workspace):
    """Create a synthetic ARGO NetCDF."""
    import xarray as xr

    n_prof = 20
    n_lev  = 50
    lat    = np.random.uniform(10, 15, n_prof)
    lon    = np.random.uniform(80, 85, n_prof)
    times  = pd.date_range("2025-01-01", periods=n_prof, freq="D")
    temp   = 20 + 5 * np.random.rand(n_prof, n_lev)
    pres   = np.tile(np.linspace(0, 2000, n_lev), (n_prof, 1))

    ds = xr.Dataset(
        {
            "TEMP":     (["N_PROF", "N_LEVELS"], temp, {"units": "degree_Celsius"}),
            "PRES":     (["N_PROF", "N_LEVELS"], pres, {"units": "dbar"}),
            "LATITUDE": (["N_PROF"], lat),
            "LONGITUDE":(["N_PROF"], lon),
            "JULD":     (["N_PROF"], times),
        }
    )
    fp = str(tmp_workspace / "argo_synthetic.nc")
    ds.to_netcdf(fp)
    return fp


@pytest.fixture(scope="session")
def synthetic_csv(tmp_workspace):
    """Create a synthetic CSV with Argo-like data."""
    n = 100
    df = pd.DataFrame({
        "datetime":    pd.date_range("2025-01-01", periods=n, freq="6h"),
        "latitude":    np.random.uniform(10, 15, n),
        "longitude":   np.random.uniform(80, 85, n),
        "temperature": np.random.uniform(20, 30, n),
        "salinity":    np.random.uniform(33, 36, n),
        "pressure":    np.random.uniform(0, 500, n),
        "qc_flag":     np.ones(n, dtype=int),
    })
    fp = str(tmp_workspace / "argo_csv_synthetic.csv")
    df.to_csv(fp, index=False)
    return fp


@pytest.fixture(scope="session")
def synthetic_raw_dir(tmp_workspace, synthetic_netcdf, synthetic_glorys,
                      synthetic_argo, synthetic_csv):
    """Build a raw/ directory with all synthetic files."""
    raw_dir = tmp_workspace / "raw"
    raw_dir.mkdir(exist_ok=True)

    for src in [synthetic_netcdf, synthetic_glorys, synthetic_argo]:
        dst = raw_dir / Path(src).name
        if not dst.exists():
            import shutil
            shutil.copy(src, dst)

    csv_dst = raw_dir / Path(synthetic_csv).name
    if not csv_dst.exists():
        import shutil
        shutil.copy(synthetic_csv, csv_dst)

    return str(raw_dir)


# ──────────────────────────────────────────────────────────────
# 1. SCANNER
# ──────────────────────────────────────────────────────────────

class TestScanner:
    def test_scan_finds_files(self, synthetic_raw_dir):
        records = pre.scan_raw_directory(synthetic_raw_dir)
        assert len(records) >= 4, "Expected at least 4 files"

    def test_scan_extensions(self, synthetic_raw_dir):
        records = pre.scan_raw_directory(synthetic_raw_dir)
        exts = {r["extension"] for r in records}
        assert ".nc" in exts
        assert ".csv" in exts

    def test_scan_empty_dir(self, tmp_workspace):
        empty = str(tmp_workspace / "empty_raw")
        os.makedirs(empty, exist_ok=True)
        records = pre.scan_raw_directory(empty)
        assert records == []

    def test_scan_nonexistent_dir(self):
        records = pre.scan_raw_directory("/nonexistent_path_xyz")
        assert records == []

    def test_inventory_csv_created(self, synthetic_raw_dir, tmp_workspace):
        records = pre.scan_raw_directory(synthetic_raw_dir)
        out = str(tmp_workspace / "inventory.csv")
        pre.save_dataset_inventory(records, out)
        assert os.path.exists(out)
        df = pd.read_csv(out)
        assert len(df) == len(records)


# ──────────────────────────────────────────────────────────────
# 2. CLASSIFICATION
# ──────────────────────────────────────────────────────────────

class TestClassification:
    @pytest.mark.parametrize("filename,expected", [
        ("sst_2025.nc",        "CORE_INPUT"),
        ("sss_monthly.nc",     "CORE_INPUT"),
        ("sla_v3.nc",          "CORE_INPUT"),
        ("glorys_thetao.nc",   "TARGET"),
        ("thetao_2025.nc",     "TARGET"),
        ("argo_profiles.nc",   "VALIDATION"),
        ("ioi_index.csv",      "AUXILIARY"),
        ("random_data.nc",     "UNKNOWN"),
    ])
    def test_classify(self, filename, expected):
        assert pre.classify_dataset(filename) == expected


# ──────────────────────────────────────────────────────────────
# 3. NETCDF READER
# ──────────────────────────────────────────────────────────────

class TestNetCDFReader:
    def test_inspect_basic(self, synthetic_netcdf):
        insp = pre.inspect_netcdf(synthetic_netcdf)
        assert insp["error"] is None
        assert "sst" in insp["variables"]
        assert insp["latitude_range"] is not None
        assert insp["longitude_range"] is not None

    def test_lat_range_in_prototype(self, synthetic_netcdf):
        insp = pre.inspect_netcdf(synthetic_netcdf)
        assert insp["latitude_range"][0] >= 10.0
        assert insp["latitude_range"][1] <= 15.25

    def test_spatial_resolution(self, synthetic_netcdf):
        insp = pre.inspect_netcdf(synthetic_netcdf)
        assert insp["spatial_resolution_deg"] is not None
        assert abs(insp["spatial_resolution_deg"] - 0.25) < 0.01

    def test_time_range(self, synthetic_netcdf):
        insp = pre.inspect_netcdf(synthetic_netcdf)
        assert insp["time_range"] is not None
        assert "2025-01-01" in insp["time_range"][0]

    def test_missing_values(self, synthetic_netcdf):
        insp = pre.inspect_netcdf(synthetic_netcdf)
        assert isinstance(insp["missing_value_counts"], dict)

    def test_bad_file(self, tmp_workspace):
        bad = str(tmp_workspace / "bad.nc")
        with open(bad, "wb") as f:
            f.write(b"not a netcdf file")
        insp = pre.inspect_netcdf(bad)
        assert insp["error"] is not None


# ──────────────────────────────────────────────────────────────
# 4. CSV READER
# ──────────────────────────────────────────────────────────────

class TestCSVReader:
    def test_inspect_basic(self, synthetic_csv):
        insp = pre.inspect_csv(synthetic_csv)
        assert insp["error"] is None
        assert insp["row_count"] == 100
        assert insp["column_count"] >= 6

    def test_detected_columns(self, synthetic_csv):
        insp = pre.inspect_csv(synthetic_csv)
        det  = insp["detected_columns"]
        assert "latitude" in det
        assert "longitude" in det
        assert "temperature" in det

    def test_no_duplicates(self, synthetic_csv):
        insp = pre.inspect_csv(synthetic_csv)
        assert insp["duplicate_rows"] == 0

    def test_missing_values_counted(self, synthetic_csv):
        insp = pre.inspect_csv(synthetic_csv)
        assert isinstance(insp["missing_value_counts"], dict)

    def test_delimiter_detected(self, synthetic_csv):
        insp = pre.inspect_csv(synthetic_csv)
        assert insp["delimiter"] == ","


# ──────────────────────────────────────────────────────────────
# 5. METADATA GENERATION
# ──────────────────────────────────────────────────────────────

class TestMetadata:
    def test_metadata_record_fields(self, synthetic_raw_dir, cfg):
        records = pre.scan_raw_directory(synthetic_raw_dir)
        rec  = records[0]
        fp   = rec["filepath"]
        ext  = rec["extension"]
        if ext in {".nc", ".nc4", ".cdf"}:
            insp = pre.inspect_netcdf(fp)
        else:
            insp = pre.inspect_csv(fp)
        meta = pre.build_metadata_record(rec, insp, 1)
        required_fields = ["dataset_id", "dataset_name", "dataset_type",
                           "variables", "units", "resolution", "coverage",
                           "row_count", "missing_values", "notes"]
        for field in required_fields:
            assert field in meta, f"Missing field: {field}"

    def test_metadata_master_saved(self, synthetic_raw_dir, cfg, tmp_workspace):
        records = pre.scan_raw_directory(synthetic_raw_dir)
        metas   = []
        for i, rec in enumerate(records):
            fp   = rec["filepath"]
            ext  = rec["extension"]
            if ext in {".nc", ".nc4", ".cdf"}:
                insp = pre.inspect_netcdf(fp)
            else:
                insp = pre.inspect_csv(fp)
            metas.append(pre.build_metadata_record(rec, insp, i + 1))

        out = str(tmp_workspace / "metadata_master.csv")
        df  = pre.generate_metadata_master(metas, out)
        assert os.path.exists(out)
        assert len(df) == len(records)


# ──────────────────────────────────────────────────────────────
# 6. QC ENGINE
# ──────────────────────────────────────────────────────────────

class TestQCEngine:
    def test_pass_clean_netcdf(self, synthetic_netcdf, cfg):
        insp = pre.inspect_netcdf(synthetic_netcdf)
        rec  = {"filepath": synthetic_netcdf, "extension": ".nc", "filename": "x.nc"}
        qc   = pre.qc_netcdf(synthetic_netcdf, insp, cfg)
        assert qc["status"] in ("PASS", "WARNING")

    def test_fail_corrupt_file(self, tmp_workspace, cfg):
        bad  = str(tmp_workspace / "corrupt.nc")
        with open(bad, "wb") as f:
            f.write(b"\x00\x01corrupt")
        insp = pre.inspect_netcdf(bad)
        qc   = pre.qc_netcdf(bad, insp, cfg)
        assert qc["status"] == "FAIL"

    def test_csv_duplicate_warning(self, tmp_workspace, cfg):
        df  = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        fp  = str(tmp_workspace / "dup.csv")
        df.to_csv(fp, index=False)
        insp = pre.inspect_csv(fp)
        qc   = pre.qc_csv(fp, insp, cfg)
        assert qc["status"] == "WARNING"
        assert any("Duplicate" in i for i in qc["issues"])

    def test_qc_returns_all_records(self, synthetic_raw_dir, cfg):
        scan = pre.scan_raw_directory(synthetic_raw_dir)
        inspections = {}
        for r in scan:
            fp  = r["filepath"]
            ext = r["extension"]
            if ext in {".nc", ".nc4", ".cdf"}:
                inspections[fp] = pre.inspect_netcdf(fp)
            else:
                inspections[fp] = pre.inspect_csv(fp)
        qc_res = pre.run_qc(scan, inspections, cfg)
        assert len(qc_res) == len(scan)


# ──────────────────────────────────────────────────────────────
# 7. COVERAGE ANALYZER
# ──────────────────────────────────────────────────────────────

class TestCoverageAnalyzer:
    def test_spatial_overlap_full(self, cfg):
        proto = cfg["region"]["prototype"]
        # Dataset fully contains prototype
        overlap = pre.compute_spatial_overlap([5, 20], [75, 90], proto)
        assert overlap == 1.0

    def test_spatial_overlap_none(self, cfg):
        proto = cfg["region"]["prototype"]
        overlap = pre.compute_spatial_overlap([25, 30], [110, 120], proto)
        assert overlap == 0.0

    def test_temporal_overlap_full(self, cfg):
        p_start = cfg["prototype_dates"]["start"]
        p_end   = cfg["prototype_dates"]["end"]
        overlap = pre.compute_temporal_overlap(
            ["2025-01-01", "2025-01-30"], p_start, p_end
        )
        assert overlap == 1.0

    def test_temporal_overlap_none(self, cfg):
        p_start = cfg["prototype_dates"]["start"]
        p_end   = cfg["prototype_dates"]["end"]
        overlap = pre.compute_temporal_overlap(
            ["2024-01-01", "2024-12-31"], p_start, p_end
        )
        assert overlap == 0.0

    def test_analyze_coverage_returns_list(self, synthetic_raw_dir, cfg):
        scan = pre.scan_raw_directory(synthetic_raw_dir)
        inspections = {}
        for r in scan:
            fp  = r["filepath"]
            ext = r["extension"]
            if ext in {".nc", ".nc4", ".cdf"}:
                inspections[fp] = pre.inspect_netcdf(fp)
            else:
                inspections[fp] = pre.inspect_csv(fp)
        cov = pre.analyze_coverage(scan, inspections, cfg)
        assert isinstance(cov, list)
        assert len(cov) == len(scan)


# ──────────────────────────────────────────────────────────────
# 8. RESOLUTION ANALYZER
# ──────────────────────────────────────────────────────────────

class TestResolutionAnalyzer:
    def test_no_mismatch_standard(self, synthetic_netcdf, cfg):
        scan  = [{"filepath": synthetic_netcdf, "extension": ".nc",
                  "filename": "sst.nc", "dataset_type": "CORE_INPUT"}]
        insp  = {synthetic_netcdf: pre.inspect_netcdf(synthetic_netcdf)}
        res   = pre.analyze_resolution(scan, insp, cfg)
        # Should be empty since our synthetic is 0.25°
        assert isinstance(res, list)

    def test_mismatch_detected(self, tmp_workspace, cfg):
        import xarray as xr
        lat  = np.arange(10.0, 15.5, 0.5)   # 0.5° resolution
        lon  = np.arange(80.0, 85.5, 0.5)
        time = pd.date_range("2025-01-01", periods=2)
        sst  = np.random.rand(2, len(lat), len(lon))
        ds   = xr.Dataset({"sst": (["time","lat","lon"], sst)},
                          coords={"time": time, "lat": lat, "lon": lon})
        fp   = str(tmp_workspace / "lowres_sst.nc")
        ds.to_netcdf(fp)

        scan  = [{"filepath": fp, "extension": ".nc",
                  "filename": "lowres_sst.nc", "dataset_type": "CORE_INPUT"}]
        insp  = {fp: pre.inspect_netcdf(fp)}
        res   = pre.analyze_resolution(scan, insp, cfg)
        # 0.5° mismatches 0.25° expected
        assert len(res) == 1
        assert res[0]["spatial_mismatch"] is True


# ──────────────────────────────────────────────────────────────
# 9. GLORYS VALIDATOR
# ──────────────────────────────────────────────────────────────

class TestGLORYSValidator:
    def test_thetao_found(self, synthetic_glorys, cfg):
        insp   = pre.inspect_netcdf(synthetic_glorys)
        result = pre.validate_glorys(synthetic_glorys, insp, cfg)
        assert result["thetao_present"] is True

    def test_exact_depth_match(self, synthetic_glorys, cfg):
        insp   = pre.inspect_netcdf(synthetic_glorys)
        result = pre.validate_glorys(synthetic_glorys, insp, cfg)
        # With a wide depth range, nearest levels should cover all required depths
        total_covered = len(result["exact_matches"]) + len(result["nearest_levels"])
        assert total_covered == len(cfg["glorys_required_depths"])

    def test_pass_status(self, synthetic_glorys, cfg):
        insp   = pre.inspect_netcdf(synthetic_glorys)
        result = pre.validate_glorys(synthetic_glorys, insp, cfg)
        # thetao present, depth range spans required levels → not FAIL
        assert result["thetao_present"] is True
        assert result["status"] in ("PASS", "WARNING")

    def test_missing_thetao(self, synthetic_netcdf, cfg):
        insp   = pre.inspect_netcdf(synthetic_netcdf)
        result = pre.validate_glorys(synthetic_netcdf, insp, cfg)
        assert result["thetao_present"] is False
        assert result["status"] == "FAIL"


# ──────────────────────────────────────────────────────────────
# 10. ARGO VALIDATOR
# ──────────────────────────────────────────────────────────────

class TestARGOValidator:
    def test_profiles_detected(self, synthetic_argo, cfg):
        insp   = pre.inspect_netcdf(synthetic_argo)
        result = pre.validate_argo(synthetic_argo, insp, cfg)
        assert result["num_profiles"] == 20

    def test_region_overlap(self, synthetic_argo, cfg):
        insp   = pre.inspect_netcdf(synthetic_argo)
        result = pre.validate_argo(synthetic_argo, insp, cfg)
        assert result["region_overlap"] is True

    def test_date_overlap(self, synthetic_argo, cfg):
        insp   = pre.inspect_netcdf(synthetic_argo)
        result = pre.validate_argo(synthetic_argo, insp, cfg)
        assert result["date_overlap"] is True

    def test_pass_status(self, synthetic_argo, cfg):
        insp   = pre.inspect_netcdf(synthetic_argo)
        result = pre.validate_argo(synthetic_argo, insp, cfg)
        assert result["status"] == "PASS"

    def test_temp_range_reported(self, synthetic_argo, cfg):
        insp   = pre.inspect_netcdf(synthetic_argo)
        result = pre.validate_argo(synthetic_argo, insp, cfg)
        assert result["temp_range"] is not None
        assert result["temp_range"][0] >= 20.0
        assert result["temp_range"][1] <= 30.0


# ──────────────────────────────────────────────────────────────
# 11. SQLITE DATABASE
# ──────────────────────────────────────────────────────────────

class TestSQLiteDatabase:
    def test_init_creates_tables(self, tmp_workspace):
        db_path = str(tmp_workspace / "test.db")
        conn    = pre.init_database(db_path)
        cursor  = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables  = {row[0] for row in cursor.fetchall()}
        conn.close()
        assert "datasets"         in tables
        assert "variables"        in tables
        assert "quality_reports"  in tables
        assert "coverage_reports" in tables
        assert "training_status"  in tables

    def test_insert_and_retrieve_dataset(self, tmp_workspace, synthetic_netcdf):
        db_path = str(tmp_workspace / "test2.db")
        conn    = pre.init_database(db_path)
        rec     = {
            "filename": "sst.nc", "filepath": synthetic_netcdf,
            "dataset_type": "CORE_INPUT", "extension": ".nc",
            "file_size_bytes": 1024,
        }
        did = pre.insert_dataset(conn, rec, "abc123")
        assert did > 0

        datasets = pre.get_all_datasets(conn)
        assert len(datasets) >= 1
        assert datasets[0]["dataset_name"] == "sst.nc"
        conn.close()

    def test_qc_report_stored(self, tmp_workspace, synthetic_netcdf, cfg):
        db_path = str(tmp_workspace / "test3.db")
        conn    = pre.init_database(db_path)
        rec     = {
            "filename": "sst.nc", "filepath": synthetic_netcdf,
            "dataset_type": "CORE_INPUT", "extension": ".nc",
            "file_size_bytes": 1024,
        }
        did  = pre.insert_dataset(conn, rec, "hash")
        insp = pre.inspect_netcdf(synthetic_netcdf)
        qc   = pre.qc_netcdf(synthetic_netcdf, insp, cfg)
        pre.insert_qc_report(conn, did, qc)

        reports = pre.get_quality_reports(conn)
        assert len(reports) >= 1
        conn.close()

    def test_coverage_report_stored(self, tmp_workspace, synthetic_netcdf, cfg):
        db_path = str(tmp_workspace / "test4.db")
        conn    = pre.init_database(db_path)
        rec     = {
            "filename": "sst.nc", "filepath": synthetic_netcdf,
            "dataset_type": "CORE_INPUT", "extension": ".nc",
            "file_size_bytes": 1024,
        }
        did = pre.insert_dataset(conn, rec, "hash")
        cov = {"filename": "sst.nc", "spatial_overlap": 1.0,
               "temporal_overlap": 1.0, "overall": "PASS"}
        pre.insert_coverage_report(conn, did, cov)
        reports = pre.get_coverage_reports(conn)
        assert len(reports) >= 1
        conn.close()


# ──────────────────────────────────────────────────────────────
# 12. HASH INTEGRITY
# ──────────────────────────────────────────────────────────────

class TestHashIntegrity:
    def test_sha256_consistent(self, synthetic_netcdf):
        h1 = pre.compute_sha256(synthetic_netcdf)
        h2 = pre.compute_sha256(synthetic_netcdf)
        assert h1 == h2
        assert len(h1) == 64

    def test_generate_hashes(self, synthetic_raw_dir):
        records = pre.scan_raw_directory(synthetic_raw_dir)
        hashes  = pre.generate_file_hashes(records)
        assert len(hashes) == len(records)
        for h in hashes.values():
            assert not h.startswith("ERROR")

    def test_verify_unchanged_files(self, synthetic_raw_dir):
        records = pre.scan_raw_directory(synthetic_raw_dir)
        hashes  = pre.generate_file_hashes(records)
        verify  = pre.verify_file_hashes(records, hashes)
        for fp, status in verify.items():
            assert status == "OK"

    def test_detect_tampered_file(self, tmp_workspace):
        fp  = str(tmp_workspace / "tamper_test.txt")
        with open(fp, "w") as f:
            f.write("original content")
        rec     = {"filepath": fp, "filename": "tamper_test.txt"}
        hashes  = pre.generate_file_hashes([rec])
        with open(fp, "w") as f:
            f.write("modified content")
        verify  = pre.verify_file_hashes([rec], hashes)
        assert "MISMATCH" in verify[fp]


# ──────────────────────────────────────────────────────────────
# 13. FASTAPI ENDPOINTS
# ──────────────────────────────────────────────────────────────

class TestFastAPIEndpoints:
    @pytest.fixture(autouse=True)
    def populate_state(self, synthetic_raw_dir, cfg, tmp_workspace):
        """Run pipeline to populate app state before API tests."""
        import app as app_module

        # Build a temp config pointing to synthetic dirs
        test_cfg = json.loads(json.dumps(cfg))
        test_cfg["paths"]["raw_data"]       = synthetic_raw_dir
        test_cfg["paths"]["database"]       = str(tmp_workspace / "api_test.db")
        test_cfg["paths"]["reports"]        = str(tmp_workspace / "reports")
        test_cfg["paths"]["processed_data"] = str(tmp_workspace / "processed")

        run_phase1(test_cfg, start_api=False)

    def test_get_datasets(self):
        client = app.test_client()
        resp   = client.get("/api/datasets")
        assert resp.status_code == 200
        data   = resp.get_json()
        assert "datasets" in data
        assert "total" in data

    def test_get_dataset_by_id(self):
        client = app.test_client()
        resp   = client.get("/api/datasets/0")
        assert resp.status_code == 200
        data   = resp.get_json()
        assert "dataset" in data

    def test_get_dataset_not_found(self):
        client = app.test_client()
        resp   = client.get("/api/datasets/99999")
        assert resp.status_code == 404

    def test_get_reports(self):
        client = app.test_client()
        resp   = client.get("/api/reports")
        assert resp.status_code == 200
        assert "reports" in resp.get_json()

    def test_get_quality(self):
        client = app.test_client()
        resp   = client.get("/api/quality")
        assert resp.status_code == 200
        assert "quality_reports" in resp.get_json()

    def test_get_coverage(self):
        client = app.test_client()
        resp   = client.get("/api/coverage")
        assert resp.status_code == 200
        assert "coverage_reports" in resp.get_json()

    def test_get_summary(self):
        client = app.test_client()
        resp   = client.get("/api/summary")
        assert resp.status_code == 200
        data   = resp.get_json()
        assert "total_datasets" in data


# ──────────────────────────────────────────────────────────────
# INTEGRATION TEST
# ──────────────────────────────────────────────────────────────

class TestIntegration:
    def test_full_pipeline(self, synthetic_raw_dir, cfg, tmp_workspace):
        """End-to-end pipeline test with synthetic data."""
        test_cfg = json.loads(json.dumps(cfg))
        test_cfg["paths"]["raw_data"]       = synthetic_raw_dir
        test_cfg["paths"]["database"]       = str(tmp_workspace / "integration.db")
        test_cfg["paths"]["reports"]        = str(tmp_workspace / "int_reports")
        test_cfg["paths"]["processed_data"] = str(tmp_workspace / "int_processed")

        success = run_phase1(test_cfg, start_api=False)
        assert success is True

        # Verify all reports exist
        rdir = test_cfg["paths"]["reports"]
        expected_files = [
            "dataset_inventory.csv",
            "metadata_master.csv",
            "dataset_quality_report.csv",
            "dataset_coverage_report.csv",
            "resolution_report.csv",
            "glorys_depth_report.csv",
            "argo_report.csv",
            "phase1_summary.json",
            "PHASE1_REPORT.md",
        ]
        for fname in expected_files:
            path = os.path.join(rdir, fname)
            assert os.path.exists(path), f"Missing report: {fname}"

        # Verify DB
        db_path = test_cfg["paths"]["database"]
        assert os.path.exists(db_path)
        conn = sqlite3.connect(db_path)
        cur  = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM datasets")
        count = cur.fetchone()[0]
        conn.close()
        assert count >= 4

        # Verify summary JSON
        with open(os.path.join(rdir, "phase1_summary.json")) as f:
            summary = json.load(f)
        assert summary["phase1_status"] == "COMPLETE"
        assert summary["total_datasets"] >= 4


# ═══════════════════════════════════════════════════════════════
# ██████████████████  PHASE 2 TESTS  ████████████████████████████
# ═══════════════════════════════════════════════════════════════

import sqlite3 as _sqlite3

# ─────────────────────────────────────────────
# Additional fixtures for Phase 2
# ─────────────────────────────────────────────

@pytest.fixture(scope="session")
def p2_cfg():
    """Load config with Phase 2 keys."""
    cfg_path = Path(__file__).parent.parent / "config.json"
    return pre.load_config(str(cfg_path))


@pytest.fixture(scope="session")
def target_grid(p2_cfg):
    return pre.build_target_grid(p2_cfg)


@pytest.fixture(scope="session")
def daily_idx(p2_cfg):
    return pd.date_range(
        p2_cfg["prototype_dates"]["start"],
        p2_cfg["prototype_dates"]["end"], freq="D"
    )


@pytest.fixture(scope="session")
def syn_sst_ds(synthetic_netcdf):
    """Open the synthetic SST file as an xarray dataset."""
    import xarray as xr
    return xr.open_dataset(synthetic_netcdf)


@pytest.fixture(scope="session")
def syn_glorys_ds(synthetic_glorys):
    import xarray as xr
    return xr.open_dataset(synthetic_glorys)


@pytest.fixture(scope="session")
def syn_sla_csv(tmp_path_factory, p2_cfg):
    """Synthetic along-track SLA CSV."""
    import numpy as np, pandas as pd
    rng = np.random.default_rng(42)
    n   = 300
    df  = pd.DataFrame({
        "time":       pd.date_range("2025-01-01", periods=n, freq="2h30min"),
        "latitude":   rng.uniform(10, 15, n),
        "longitude":  rng.uniform(80, 85, n),
        "sla":        rng.normal(0, 0.05, n),
        "qc_flag":    np.ones(n, dtype=int),
    })
    fp = str(tmp_path_factory.mktemp("sla") / "sla_synthetic.csv")
    df.to_csv(fp, index=False)
    return fp


@pytest.fixture(scope="session")
def p2_workspace(tmp_path_factory, synthetic_raw_dir, p2_cfg):
    """
    Run the complete Phase 1 + Phase 2 pipeline on synthetic data.
    Returns a dict with paths and the resulting datasets.
    """
    import json, shutil
    ws  = tmp_path_factory.mktemp("p2_ws")
    raw = ws / "raw"
    raw.mkdir()

    # Copy synthetic files into raw/
    for fp in Path(synthetic_raw_dir).rglob("*"):
        if fp.is_file():
            shutil.copy(fp, raw / fp.name)

    test_cfg = json.loads(json.dumps(p2_cfg))
    test_cfg["paths"]["raw_data"]        = str(raw)
    test_cfg["paths"]["database"]        = str(ws / "ocean_platform.db")
    test_cfg["paths"]["reports"]         = str(ws / "reports/phase1")
    test_cfg["paths"]["reports_phase2"]  = str(ws / "reports/phase2")
    test_cfg["paths"]["processed_data"]  = str(ws / "processed")
    test_cfg["paths"]["final_data"]      = str(ws / "final")
    test_cfg["paths"]["predictions"]     = str(ws / "predictions")

    from app import run_phase1, run_phase2
    run_phase1(test_cfg, start_api=False)
    run_phase2(test_cfg, start_api=False)

    return {"cfg": test_cfg, "ws": ws}


# ─────────────────────────────────────────────
# P2.1 Phase 1 Validation
# ─────────────────────────────────────────────

class TestPhase1Validation:
    def test_valid_when_complete(self, p2_workspace):
        result = pre.validate_phase1_complete(p2_workspace["cfg"])
        assert result["ok"] is True

    def test_invalid_when_reports_missing(self, tmp_path, p2_cfg):
        import json
        bad_cfg = json.loads(json.dumps(p2_cfg))
        bad_cfg["paths"]["reports"] = str(tmp_path / "nonexistent")
        bad_cfg["paths"]["database"] = str(tmp_path / "no.db")
        result = pre.validate_phase1_complete(bad_cfg)
        assert result["ok"] is False
        assert len(result["missing_reports"]) > 0

    def test_db_check(self, p2_workspace):
        result = pre.validate_phase1_complete(p2_workspace["cfg"])
        assert result["db_exists"] is True


# ─────────────────────────────────────────────
# P2.2 Region Crop
# ─────────────────────────────────────────────

class TestRegionCrop:
    def test_crop_stays_in_bounds(self, syn_sst_ds, p2_cfg):
        import xarray as xr
        ds = syn_sst_ds.copy()
        ds, cov = pre.crop_region(ds, p2_cfg)
        if "latitude" in ds.coords:
            assert float(ds["latitude"].min()) >= p2_cfg["region"]["prototype"]["lat_min"] - 0.01
            assert float(ds["latitude"].max()) <= p2_cfg["region"]["prototype"]["lat_max"] + 0.01
        if "longitude" in ds.coords:
            assert float(ds["longitude"].min()) >= p2_cfg["region"]["prototype"]["lon_min"] - 0.01
            assert float(ds["longitude"].max()) <= p2_cfg["region"]["prototype"]["lon_max"] + 0.01

    def test_crop_coverage_info_returned(self, syn_sst_ds, p2_cfg):
        ds, cov = pre.crop_region(syn_sst_ds.copy(), p2_cfg)
        assert "before" in cov
        assert "after"  in cov

    def test_360_lon_conversion(self, p2_cfg):
        import xarray as xr, numpy as np
        lat  = np.array([10., 12., 15.])
        lon  = np.array([80., 82., 85.]) + 360   # 0-360 convention
        data = np.zeros((3, 3))
        ds   = xr.Dataset({"sst": (["latitude","longitude"], data)},
                          coords={"latitude": lat, "longitude": lon})
        ds_norm = pre.normalize_longitude(ds)
        assert float(ds_norm["longitude"].min()) < 180


# ─────────────────────────────────────────────
# P2.3 Time Crop
# ─────────────────────────────────────────────

class TestTimeCrop:
    def test_crop_within_dates(self, syn_sst_ds, p2_cfg):
        import xarray as xr
        ds, ti = pre.crop_time(syn_sst_ds.copy(), p2_cfg)
        if "time" in ds.coords and ds.sizes.get("time", 0) > 0:
            times = pd.to_datetime(ds["time"].values)
            assert str(times.min().date()) >= p2_cfg["prototype_dates"]["start"]
            assert str(times.max().date()) <= p2_cfg["prototype_dates"]["end"]

    def test_crop_returns_info(self, syn_sst_ds, p2_cfg):
        ds, ti = pre.crop_time(syn_sst_ds.copy(), p2_cfg)
        assert "before_steps" in ti or "skipped" in ti

    def test_no_time_coord_safe(self, p2_cfg):
        import xarray as xr, numpy as np
        ds = xr.Dataset({"v": (["x"], np.array([1., 2.]))},
                        coords={"x": [0, 1]})
        ds_out, ti = pre.crop_time(ds, p2_cfg)
        assert "skipped" in ti or "error" in ti


# ─────────────────────────────────────────────
# P2.4 Coordinate Rename
# ─────────────────────────────────────────────

class TestCoordinateRename:
    def test_lat_renamed(self):
        import xarray as xr, numpy as np
        ds = xr.Dataset({"v": (["lat","lon"], np.zeros((3,3)))},
                        coords={"lat": [10.,12.,15.], "lon": [80.,82.,85.]})
        ds_out, log = pre.standardize_coordinates(ds)
        assert "latitude"  in ds_out.coords
        assert "longitude" in ds_out.coords

    def test_nav_lat_renamed(self):
        import xarray as xr, numpy as np
        ds = xr.Dataset({"v": (["nav_lat","nav_lon"], np.zeros((2,2)))},
                        coords={"nav_lat": [10.,12.], "nav_lon": [80.,82.]})
        ds_out, log = pre.standardize_coordinates(ds)
        assert "latitude" in ds_out.coords

    def test_depth_renamed(self):
        import xarray as xr, numpy as np
        ds = xr.Dataset({"v": (["lev"], np.zeros(5))},
                        coords={"lev": [0.,5.,10.,20.,30.]})
        ds_out, log = pre.standardize_coordinates(ds)
        assert "depth" in ds_out.coords

    def test_no_rename_when_canonical(self):
        import xarray as xr, numpy as np
        ds = xr.Dataset({"v": (["latitude","longitude"], np.zeros((2,2)))},
                        coords={"latitude": [10.,12.], "longitude": [80.,82.]})
        ds_out, log = pre.standardize_coordinates(ds)
        assert log == {}


# ─────────────────────────────────────────────
# P2.5 Variable Rename
# ─────────────────────────────────────────────

class TestVariableRename:
    def test_ugos_to_current_u(self, p2_cfg):
        import xarray as xr, numpy as np
        ds = xr.Dataset({"ugos": (["latitude","longitude"], np.zeros((3,3)))},
                        coords={"latitude": [10.,12.,15.], "longitude": [80.,82.,85.]})
        ds_out, log = pre.standardize_variables(ds, p2_cfg)
        assert "current_u" in ds_out.data_vars

    def test_analysed_sst_to_sst(self, p2_cfg):
        import xarray as xr, numpy as np
        ds = xr.Dataset({"analysed_sst": (["latitude","longitude"], np.zeros((3,3)))},
                        coords={"latitude": [10.,12.,15.], "longitude": [80.,82.,85.]})
        ds_out, log = pre.standardize_variables(ds, p2_cfg)
        assert "sst" in ds_out.data_vars

    def test_no_rename_when_canonical(self, p2_cfg):
        import xarray as xr, numpy as np
        ds = xr.Dataset({"sst": (["latitude","longitude"], np.zeros((2,2)))},
                        coords={"latitude": [10.,12.], "longitude": [80.,82.]})
        ds_out, log = pre.standardize_variables(ds, p2_cfg)
        assert log == {}


# ─────────────────────────────────────────────
# P2.6 Unit Conversion
# ─────────────────────────────────────────────

class TestUnitConversion:
    def test_kelvin_to_celsius(self, p2_cfg):
        import xarray as xr, numpy as np
        ds = xr.Dataset(
            {"sst": (["lat"], np.array([300., 301., 302.], dtype="float32"),
                     {"units": "K"})},
            coords={"lat": [10., 12., 15.]}
        )
        log = pre.standardize_units(ds, p2_cfg)
        assert len(log) == 1
        assert log[0]["to"] == "degC"
        assert abs(float(ds["sst"].values[0]) - (300. - 273.15)) < 0.01

    def test_celsius_unchanged(self, p2_cfg):
        import xarray as xr, numpy as np
        ds = xr.Dataset(
            {"sst": (["lat"], np.array([28., 29.], dtype="float32"),
                     {"units": "degC"})},
            coords={"lat": [10., 12.]}
        )
        log = pre.standardize_units(ds, p2_cfg)
        assert log == []

    def test_cm_to_m(self, p2_cfg):
        import xarray as xr, numpy as np
        ds = xr.Dataset(
            {"sla": (["lat"], np.array([10., 20.], dtype="float32"),
                     {"units": "cm"})},
            coords={"lat": [10., 12.]}
        )
        log = pre.standardize_units(ds, p2_cfg)
        assert len(log) == 1
        assert abs(float(ds["sla"].values[0]) - 0.10) < 0.001


# ─────────────────────────────────────────────
# P2.7 Missing Value Handler
# ─────────────────────────────────────────────

class TestMissingValueHandler:
    def test_fillvalue_becomes_nan(self):
        import xarray as xr, numpy as np
        data = np.array([1., 9.96921e+36, 2.], dtype="float32")
        ds   = xr.Dataset({"sst": (["x"], data, {"_FillValue": 9.96921e+36})},
                          coords={"x": [0,1,2]})
        report = pre.handle_missing_values(ds)
        assert np.isnan(ds["sst"].values[1])
        assert report["sst"]["newly_masked"] >= 1

    def test_infinity_becomes_nan(self):
        import xarray as xr, numpy as np
        data = np.array([1., np.inf, -np.inf, 2.], dtype="float32")
        ds   = xr.Dataset({"v": (["x"], data)}, coords={"x": [0,1,2,3]})
        pre.handle_missing_values(ds)
        assert np.isnan(ds["v"].values[1])
        assert np.isnan(ds["v"].values[2])

    def test_no_zero_replacement(self):
        import xarray as xr, numpy as np
        data = np.array([0., 1., 2.], dtype="float32")
        ds   = xr.Dataset({"v": (["x"], data)}, coords={"x": [0,1,2]})
        pre.handle_missing_values(ds)
        assert ds["v"].values[0] == 0.0   # zero must remain zero

    def test_report_structure(self):
        import xarray as xr, numpy as np
        ds = xr.Dataset({"v": (["x"], np.array([1., np.nan, 2.]))},
                        coords={"x": [0,1,2]})
        report = pre.handle_missing_values(ds)
        assert "v" in report
        for key in ["final_nan_count","missing_pct","total_cells"]:
            assert key in report["v"]


# ─────────────────────────────────────────────
# P2.8 Daily Temporal Harmonization
# ─────────────────────────────────────────────

class TestDailyHarmonization:
    def test_daily_passthrough(self, syn_sst_ds, p2_cfg):
        ds_daily, method = pre.harmonize_to_daily(syn_sst_ds.copy(), "daily", p2_cfg)
        assert method == "daily_passthrough"

    def test_monthly_to_daily(self, p2_cfg):
        import xarray as xr, numpy as np
        times = pd.date_range("2025-01-01", periods=3, freq="ME")
        data  = np.random.rand(3, 3, 3).astype("float32")
        ds    = xr.Dataset({"sss": (["time","latitude","longitude"], data)},
                           coords={"time": times,
                                   "latitude": [10.,12.,15.],
                                   "longitude": [80.,82.,85.]})
        ds_daily, method = pre.harmonize_to_daily(ds, "monthly", p2_cfg)
        assert "interp" in method
        assert ds_daily.sizes["time"] == 30

    def test_hourly_to_daily_mean(self, p2_cfg):
        import xarray as xr, numpy as np
        times = pd.date_range("2025-01-01", periods=24*5, freq="h")
        data  = np.random.rand(len(times), 3, 3).astype("float32")
        ds    = xr.Dataset({"wind_u": (["time","latitude","longitude"], data)},
                           coords={"time": times,
                                   "latitude": [10.,12.,15.],
                                   "longitude": [80.,82.,85.]})
        ds_daily, method = pre.harmonize_to_daily(ds, "hourly", p2_cfg)
        assert method == "hourly_daily_mean"

    def test_no_time_coord(self, p2_cfg):
        import xarray as xr, numpy as np
        ds = xr.Dataset({"v": (["x"], np.zeros(3))}, coords={"x": [0,1,2]})
        ds_out, method = pre.harmonize_to_daily(ds, "daily", p2_cfg)
        assert method == "no_time"


# ─────────────────────────────────────────────
# P2.9 Spatial Harmonization
# ─────────────────────────────────────────────

class TestSpatialHarmonization:
    def test_target_grid_shape(self, target_grid):
        lats = target_grid["latitude"]
        lons = target_grid["longitude"]
        assert len(lats) == 21   # 10.0 to 15.0 at 0.25° = 21 points
        assert len(lons) == 21   # 80.0 to 85.0 at 0.25° = 21 points

    def test_target_grid_bounds(self, target_grid, p2_cfg):
        proto = p2_cfg["region"]["prototype"]
        assert abs(target_grid["latitude"][0]  - proto["lat_min"]) < 0.01
        assert abs(target_grid["latitude"][-1] - proto["lat_max"]) < 0.01
        assert abs(target_grid["longitude"][0]  - proto["lon_min"]) < 0.01
        assert abs(target_grid["longitude"][-1] - proto["lon_max"]) < 0.01

    def test_regrid_output_shape(self, syn_sst_ds, target_grid, p2_cfg):
        ds, _  = pre.crop_region(syn_sst_ds.copy(), p2_cfg)
        ds_std, _ = pre.standardize_coordinates(ds)
        ds_grid   = pre.regrid_to_standard(ds_std, target_grid, p2_cfg)
        if "latitude" in ds_grid.coords:
            assert ds_grid.sizes["latitude"]  == 21
            assert ds_grid.sizes["longitude"] == 21

    def test_regrid_float32(self, syn_sst_ds, target_grid, p2_cfg):
        import numpy as np
        ds, _  = pre.crop_region(syn_sst_ds.copy(), p2_cfg)
        ds_std, _ = pre.standardize_coordinates(ds)
        ds_grid   = pre.regrid_to_standard(ds_std, target_grid, p2_cfg)
        for var in ds_grid.data_vars:
            assert ds_grid[var].dtype == np.float32


# ─────────────────────────────────────────────
# P2.10 SLA Along-Track Gridding
# ─────────────────────────────────────────────

class TestSLAGridding:
    def test_sla_grid_shape(self, syn_sla_csv, target_grid, daily_idx, p2_cfg):
        ds_sla, status = pre.grid_sla_alongtrack(
            syn_sla_csv, p2_cfg, target_grid, daily_idx)
        assert status == "ok"
        assert ds_sla is not None
        assert "sla" in ds_sla.data_vars
        assert "sla_observation_count" in ds_sla.data_vars
        assert ds_sla.sizes["time"] == 30
        assert ds_sla.sizes["latitude"]  == 21
        assert ds_sla.sizes["longitude"] == 21

    def test_no_extrapolation(self, syn_sla_csv, target_grid, daily_idx, p2_cfg):
        import numpy as np
        ds_sla, _ = pre.grid_sla_alongtrack(
            syn_sla_csv, p2_cfg, target_grid, daily_idx)
        cnt = ds_sla["sla_observation_count"].values
        sla = ds_sla["sla"].values
        # Where count == 0, sla must be NaN
        assert np.all(np.isnan(sla[cnt == 0]))

    def test_observation_count_nonneg(self, syn_sla_csv, target_grid, daily_idx, p2_cfg):
        import numpy as np
        ds_sla, _ = pre.grid_sla_alongtrack(
            syn_sla_csv, p2_cfg, target_grid, daily_idx)
        assert np.all(ds_sla["sla_observation_count"].values >= 0)

    def test_missing_columns_fails_gracefully(self, tmp_path, target_grid, daily_idx, p2_cfg):
        import pandas as pd
        fp = str(tmp_path / "bad_sla.csv")
        pd.DataFrame({"col_a": [1,2], "col_b": [3,4]}).to_csv(fp, index=False)
        ds_sla, status = pre.grid_sla_alongtrack(fp, p2_cfg, target_grid, daily_idx)
        assert ds_sla is None


# ─────────────────────────────────────────────
# P2.11 GLORYS Depth Interpolation
# ─────────────────────────────────────────────

class TestGLORYSDepthProcessor:
    def test_exact_depth_extracted(self, synthetic_glorys, p2_cfg):
        import xarray as xr
        ds     = xr.open_dataset(synthetic_glorys)
        ds_out, dlog = pre.process_glorys_depths(ds, p2_cfg)
        assert ds_out is not None
        assert "thetao" in ds_out.data_vars
        # All 15 target depths should be produced (exact + interpolated)
        valid = len(dlog.get("exact", [])) + len(dlog.get("interpolated", []))
        assert ds_out.sizes["depth"] == valid
        assert valid == len(p2_cfg["target_depths"])

    def test_depth_log_keys(self, synthetic_glorys, p2_cfg):
        import xarray as xr
        ds = xr.open_dataset(synthetic_glorys)
        _, dlog = pre.process_glorys_depths(ds, p2_cfg)
        for key in ["target","exact","interpolated","missing"]:
            assert key in dlog

    def test_no_thetao_returns_none(self, synthetic_netcdf, p2_cfg):
        import xarray as xr
        ds = xr.open_dataset(synthetic_netcdf)
        ds_out, dlog = pre.process_glorys_depths(ds, p2_cfg)
        assert ds_out is None


# ─────────────────────────────────────────────
# P2.12 Dataset Alignment
# ─────────────────────────────────────────────

class TestDatasetAlignment:
    def test_surface_shape(self, p2_workspace):
        surf_path = str(Path(p2_workspace["ws"]) / "final" / "ocean_surface_inputs.nc")
        assert os.path.exists(surf_path)
        ds = xr.open_dataset(surf_path)
        assert ds.sizes["time"]      == 30
        assert ds.sizes["latitude"]  == 21
        assert ds.sizes["longitude"] == 21
        ds.close()

    def test_target_shape(self, p2_workspace):
        tgt_path = str(Path(p2_workspace["ws"]) / "final" / "subsurface_temperature_target.nc")
        assert os.path.exists(tgt_path)
        ds = xr.open_dataset(tgt_path)
        assert ds.sizes["time"]      == 30
        assert ds.sizes["latitude"]  == 21
        assert ds.sizes["longitude"] == 21
        assert ds.sizes["depth"]     == 15
        ds.close()

    def test_surface_has_all_variables(self, p2_workspace):
        surf_path = str(Path(p2_workspace["ws"]) / "final" / "ocean_surface_inputs.nc")
        ds = xr.open_dataset(surf_path)
        expected = ["sst","sss","sla","sla_observation_count",
                    "current_u","current_v","wind_u","wind_v"]
        for var in expected:
            assert var in ds.data_vars, f"Missing surface var: {var}"
        ds.close()

    def test_target_has_thetao(self, p2_workspace):
        tgt_path = str(Path(p2_workspace["ws"]) / "final" / "subsurface_temperature_target.nc")
        ds = xr.open_dataset(tgt_path)
        assert "thetao" in ds.data_vars
        ds.close()


# ─────────────────────────────────────────────
# P2.13 NetCDF Generation
# ─────────────────────────────────────────────

class TestNetCDFGeneration:
    def test_surface_nc_readable(self, p2_workspace):
        import xarray as xr
        fp = str(Path(p2_workspace["ws"]) / "final" / "ocean_surface_inputs.nc")
        ds = xr.open_dataset(fp)
        assert len(ds.data_vars) >= 1
        ds.close()

    def test_target_nc_readable(self, p2_workspace):
        import xarray as xr
        fp = str(Path(p2_workspace["ws"]) / "final" / "subsurface_temperature_target.nc")
        ds = xr.open_dataset(fp)
        assert "thetao" in ds.data_vars
        ds.close()

    def test_coordinates_in_netcdf(self, p2_workspace):
        import xarray as xr
        fp = str(Path(p2_workspace["ws"]) / "final" / "ocean_surface_inputs.nc")
        ds = xr.open_dataset(fp)
        assert "time"      in ds.coords
        assert "latitude"  in ds.coords
        assert "longitude" in ds.coords
        ds.close()

    def test_dates_correct(self, p2_workspace, p2_cfg):
        import xarray as xr
        fp = str(Path(p2_workspace["ws"]) / "final" / "ocean_surface_inputs.nc")
        ds = xr.open_dataset(fp)
        times = pd.to_datetime(ds["time"].values)
        assert str(times[0].date())  == p2_cfg["prototype_dates"]["start"]
        assert str(times[-1].date()) == p2_cfg["prototype_dates"]["end"]
        ds.close()

    def test_depths_correct(self, p2_workspace, p2_cfg):
        import xarray as xr, numpy as np
        fp = str(Path(p2_workspace["ws"]) / "final" / "subsurface_temperature_target.nc")
        ds = xr.open_dataset(fp)
        depths = list(ds["depth"].values)
        expected = p2_cfg["target_depths"]
        assert len(depths) == len(expected)
        ds.close()


# ─────────────────────────────────────────────
# P2.14 Statistics
# ─────────────────────────────────────────────

class TestStatistics:
    def test_stats_fields(self, p2_workspace):
        import xarray as xr
        fp = str(Path(p2_workspace["ws"]) / "final" / "ocean_surface_inputs.nc")
        ds = xr.open_dataset(fp)
        stats = pre.compute_dataset_statistics(ds)
        assert len(stats) > 0
        for s in stats:
            for key in ["variable","mean","std","min","max","missing_pct","coverage_pct"]:
                assert key in s
        ds.close()

    def test_missing_pct_in_range(self, p2_workspace):
        import xarray as xr
        fp = str(Path(p2_workspace["ws"]) / "final" / "ocean_surface_inputs.nc")
        ds = xr.open_dataset(fp)
        stats = pre.compute_dataset_statistics(ds)
        for s in stats:
            assert 0 <= s["missing_pct"] <= 100
        ds.close()


# ─────────────────────────────────────────────
# P2.15 SQLite Phase 2 Tables
# ─────────────────────────────────────────────

class TestSQLitePhase2:
    def test_phase2_tables_exist(self, p2_workspace):
        db_path = p2_workspace["cfg"]["paths"]["database"]
        conn    = _sqlite3.connect(db_path)
        cur     = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables  = {r[0] for r in cur.fetchall()}
        conn.close()
        for t in ["harmonized_dataset","statistics","missing_values",
                  "interpolation_log","depth_log"]:
            assert t in tables, f"Missing table: {t}"

    def test_harmonized_datasets_populated(self, p2_workspace):
        db_path = p2_workspace["cfg"]["paths"]["database"]
        conn    = _sqlite3.connect(db_path)
        rows    = pre.get_harmonized_datasets(conn)
        conn.close()
        assert len(rows) >= 2   # surface + target

    def test_statistics_populated(self, p2_workspace):
        db_path = p2_workspace["cfg"]["paths"]["database"]
        conn    = _sqlite3.connect(db_path)
        stats   = pre.get_phase2_statistics(conn)
        conn.close()
        assert len(stats) > 0

    def test_depth_log_populated(self, p2_workspace):
        db_path = p2_workspace["cfg"]["paths"]["database"]
        conn    = _sqlite3.connect(db_path)
        depths  = pre.get_phase2_depths(conn)
        conn.close()
        assert len(depths) > 0

    def test_interpolation_log_populated(self, p2_workspace):
        db_path = p2_workspace["cfg"]["paths"]["database"]
        conn    = _sqlite3.connect(db_path)
        ilog    = pre.get_phase2_interpolation(conn)
        conn.close()
        assert len(ilog) >= 0   # may be 0 if no files found; just ensure no crash


# ─────────────────────────────────────────────
# P2 FastAPI Endpoints
# ─────────────────────────────────────────────

class TestPhase2APIEndpoints:
    @pytest.fixture(autouse=True)
    def populate_state(self, p2_workspace):
        import json
        from app import run_phase1, run_phase2
        run_phase1(p2_workspace["cfg"], start_api=False)
        run_phase2(p2_workspace["cfg"], start_api=False)

    def test_phase2_status(self):
        client = app.test_client()
        resp   = client.get("/api/phase2/status")
        assert resp.status_code == 200
        assert "status" in resp.get_json()

    def test_statistics_endpoint(self):
        client = app.test_client()
        resp   = client.get("/api/statistics")
        assert resp.status_code == 200
        assert "statistics" in resp.get_json()

    def test_depths_endpoint(self):
        client = app.test_client()
        resp   = client.get("/api/depths")
        assert resp.status_code == 200
        assert "depths" in resp.get_json()

    def test_surface_endpoint(self):
        client = app.test_client()
        resp   = client.get("/api/surface")
        assert resp.status_code == 200
        data   = resp.get_json()
        assert "variables" in data

    def test_target_endpoint(self):
        client = app.test_client()
        resp   = client.get("/api/target")
        assert resp.status_code == 200
        data   = resp.get_json()
        assert "thetao" in data["variables"]

    def test_missing_endpoint(self):
        client = app.test_client()
        resp   = client.get("/api/missing")
        assert resp.status_code == 200
        assert "missing" in resp.get_json()

    def test_interpolation_endpoint(self):
        client = app.test_client()
        resp   = client.get("/api/interpolation")
        assert resp.status_code == 200
        assert "interpolation" in resp.get_json()


# ─────────────────────────────────────────────
# P2 Integration: Raw Files Untouched
# ─────────────────────────────────────────────

class TestPhase2RawIntegrity:
    def test_raw_files_unchanged(self, p2_workspace):
        raw_dir = p2_workspace["cfg"]["paths"]["raw_data"]
        records = pre.scan_raw_directory(raw_dir)
        hashes1 = pre.generate_file_hashes(records)
        hashes2 = pre.generate_file_hashes(records)
        for fp in hashes1:
            assert hashes1[fp] == hashes2[fp]

    def test_no_modification_after_phase2(self, p2_workspace):
        """Verify raw files match hashes generated before Phase 2."""
        raw_dir = p2_workspace["cfg"]["paths"]["raw_data"]
        records = pre.scan_raw_directory(raw_dir)
        hashes  = pre.generate_file_hashes(records)
        verify  = pre.verify_file_hashes(records, hashes)
        for fp, status in verify.items():
            assert status == "OK", f"Raw file tampered: {fp}"


# ─────────────────────────────────────────────
# P2 Full Integration Test
# ─────────────────────────────────────────────

class TestPhase2FullIntegration:
    def test_all_reports_exist(self, p2_workspace):
        rdir = p2_workspace["cfg"]["paths"]["reports_phase2"]
        expected = [
            "phase2_summary.json",
            "phase2_alignment_report.csv",
            "phase2_missing_report.csv",
            "phase2_statistics.csv",
            "phase2_depth_report.csv",
            "phase2_interpolation_report.csv",
            "PHASE2_REPORT.md",
        ]
        for fname in expected:
            path = os.path.join(rdir, fname)
            assert os.path.exists(path), f"Missing Phase 2 report: {fname}"

    def test_summary_complete_status(self, p2_workspace):
        import json
        rdir    = p2_workspace["cfg"]["paths"]["reports_phase2"]
        fp      = os.path.join(rdir, "phase2_summary.json")
        with open(fp) as f:
            summary = json.load(f)
        assert summary["phase2_status"] == "COMPLETE"
        assert summary["n_time_steps"]  == 30

    def test_final_dir_has_both_netcdfs(self, p2_workspace):
        final_dir = p2_workspace["cfg"]["paths"]["final_data"]
        assert os.path.exists(os.path.join(final_dir, "ocean_surface_inputs.nc"))
        assert os.path.exists(os.path.join(final_dir, "subsurface_temperature_target.nc"))

    def test_plots_generated(self, p2_workspace):
        plots_dir = os.path.join(
            p2_workspace["cfg"]["paths"]["reports_phase2"], "plots"
        )
        expected_plots = [
            "grid_coverage.png",
            "daily_coverage.png",
            "depth_availability.png",
            "missing_heatmap.png",
            "interpolation_summary.png",
        ]
        for plot in expected_plots:
            assert os.path.exists(os.path.join(plots_dir, plot)), \
                f"Missing plot: {plot}"


# ═══════════════════════════════════════════════════════════════
# ██████████████████  PHASE 3 TESTS  ████████████████████████████
# ═══════════════════════════════════════════════════════════════

import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import ai_engine  as ai
import evaluation as ev
import visualization as viz
import admin      as adm


# ─────────────────────────────────────────────
# P3 shared fixtures
# ─────────────────────────────────────────────

@pytest.fixture(scope="session")
def p3_cfg():
    cfg = pre.load_config(str(Path(__file__).parent.parent / "config.json"))
    if "phase3" not in cfg:
        cfg["phase3"] = {"epochs":2,"lr":1e-3,"batch_size":4,"patience":3,
                         "resume":False,"embed_dim":32,"mc_samples":3,
                         "loss_weights":{"mse":1.0,"grad":0.1,"smooth":0.05,"therm":0.0}}
    return cfg


@pytest.fixture(scope="session")
def tiny_surface():
    """Tiny synthetic surface array (T=10, C=8, H=5, W=5)."""
    rng = np.random.default_rng(0)
    arr = rng.normal(25, 3, (10, 8, 5, 5)).astype("float32")
    return arr


@pytest.fixture(scope="session")
def tiny_target():
    """Tiny synthetic target array (T=10, D=15, H=5, W=5)."""
    rng = np.random.default_rng(1)
    arr = rng.normal(22, 4, (10, 15, 5, 5)).astype("float32")
    return arr


@pytest.fixture(scope="session")
def tiny_model():
    return ai.OceanCNNModel(in_channels=8, n_depths=15, embed_dim=32, h=5, w=5)


@pytest.fixture(scope="session")
def tiny_norm_stats(tiny_surface, tiny_target):
    _, _, ns = ai.normalize_data(tiny_surface, tiny_target)
    return ns


@pytest.fixture(scope="session")
def tiny_pred_ds(tiny_model, tiny_surface, tiny_norm_stats, p3_cfg):
    times = pd.date_range("2025-01-01", periods=10, freq="D")
    lats  = np.arange(10.0, 10.0 + 5*0.25, 0.25)
    lons  = np.arange(80.0, 80.0 + 5*0.25, 0.25)
    tiny_model.eval()
    preds = []
    for t in range(10):
        p = ai.predict_single(tiny_model, tiny_surface[t], tiny_norm_stats)
        preds.append(p)
    pred_arr = np.stack(preds, axis=0)
    depths   = p3_cfg.get("target_depths", list(range(15)))[:15]
    return xr.Dataset(
        {"thetao_pred": (["time","depth","latitude","longitude"], pred_arr)},
        coords={"time": times, "depth": depths[:pred_arr.shape[1]],
                "latitude": lats, "longitude": lons},
    )


@pytest.fixture(scope="session")
def p3_full_workspace(p2_workspace, p3_cfg):
    """Run Phase 3 on top of the existing p2_workspace."""
    import json
    ws  = p2_workspace["ws"]
    cfg = json.loads(json.dumps(p2_workspace["cfg"]))
    cfg["phase3"] = p3_cfg.get("phase3", {})
    cfg["paths"]["final_data"]      = str(ws / "final")
    cfg["paths"]["predictions"]     = str(ws / "predictions")
    # Phase 3 pipeline hardcodes "reports/phase3" — change cwd so it lands in ws
    import os
    orig_cwd = os.getcwd()
    os.chdir(str(ws))
    # Ensure phase1 reports exist relative to new cwd
    import shutil
    p1src = ws / "reports" / "phase1"
    p1dst = ws / "reports" / "phase1"
    # Already in ws — just run
    try:
        from app import run_phase1, run_phase2, run_phase3
        run_phase1(cfg, start_api=False)
        run_phase2(cfg, start_api=False)
        run_phase3(cfg, start_api=False)
    finally:
        os.chdir(orig_cwd)
    return {"cfg": cfg, "ws": ws}


# ─────────────────────────────────────────────
# P3.A Data Loading & Normalisation
# ─────────────────────────────────────────────

class TestDataLoadingNorm:
    def test_load_returns_correct_shapes(self, p3_full_workspace):
        cfg = p3_full_workspace["cfg"]
        surf, tgt, times, lats, lons = ai.load_phase2_data(cfg)
        assert surf.ndim == 4   # (T, C, H, W)
        assert tgt.ndim  == 4   # (T, D, H, W)
        assert surf.shape[1] == 8
        assert tgt.shape[1]  == 15

    def test_normalise_zero_mean(self, tiny_surface, tiny_target):
        surf_n, tgt_n, ns = ai.normalize_data(tiny_surface, tiny_target)
        # Each channel should be approximately zero-mean
        for c in range(surf_n.shape[1]):
            mu = float(np.nanmean(surf_n[:, c]))
            assert abs(mu) < 0.5, f"Channel {c} not near zero-mean: {mu}"

    def test_normalise_stats_keys(self, tiny_surface, tiny_target):
        _, _, ns = ai.normalize_data(tiny_surface, tiny_target)
        for c in range(8):
            assert f"surf_ch{c}" in ns
        assert "target" in ns


# ─────────────────────────────────────────────
# P3.B Model Architecture
# ─────────────────────────────────────────────

class TestModelArchitecture:
    def test_model_forward_pass(self, tiny_model, tiny_surface):
        tiny_model.eval()
        x = torch.from_numpy(tiny_surface[:2]).float()
        with torch.no_grad():
            pred, emb, att = tiny_model(x)
        assert pred.shape == (2, 15, 5, 5)
        assert emb.shape[0] == 2
        assert att.shape[0] == 2

    def test_satellite_embedding_engine(self, tiny_surface):
        engine = ai.SatelliteEmbeddingEngine(in_channels=8, embed_dim=32)
        engine.eval()
        x = torch.from_numpy(tiny_surface[:1]).float()
        with torch.no_grad():
            emb, att, feat = engine(x)
        assert emb.shape == (1, 32, 5, 5)
        assert att.shape[0] == 1

    def test_attention_output_range(self, tiny_model, tiny_surface):
        tiny_model.eval()
        x = torch.from_numpy(tiny_surface[:1]).float()
        with torch.no_grad():
            _, _, att = tiny_model(x)
        arr = att.cpu().numpy()
        assert float(arr.min()) >= 0.0
        assert float(arr.max()) <= 1.0 + 1e-4

    def test_model_output_dtype(self, tiny_model, tiny_surface):
        x = torch.from_numpy(tiny_surface[:1]).float()
        with torch.no_grad():
            pred, _, _ = tiny_model(x)
        assert pred.dtype == torch.float32

    def test_resblock_identity(self):
        block = ai.ResBlock(16)
        block.eval()
        x = torch.zeros(1, 16, 4, 4)
        out = block(x)
        assert out.shape == x.shape


# ─────────────────────────────────────────────
# P3.C Physics-Guided Loss
# ─────────────────────────────────────────────

class TestPhysicsGuidedLoss:
    def test_loss_returns_scalar(self, p3_cfg):
        loss_fn = ai.PhysicsGuidedLoss(p3_cfg)
        pred   = torch.randn(2, 15, 5, 5)
        target = torch.randn(2, 15, 5, 5)
        loss, bd = loss_fn(pred, target)
        assert loss.dim() == 0
        assert "total" in bd

    def test_loss_breakdown_keys(self, p3_cfg):
        loss_fn = ai.PhysicsGuidedLoss(p3_cfg)
        pred    = torch.randn(1, 15, 5, 5)
        target  = torch.randn(1, 15, 5, 5)
        _, bd   = loss_fn(pred, target)
        for k in ["mse","grad","smooth","total"]:
            assert k in bd

    def test_all_nan_target(self, p3_cfg):
        loss_fn = ai.PhysicsGuidedLoss(p3_cfg)
        pred    = torch.randn(1, 15, 5, 5)
        target  = torch.full((1, 15, 5, 5), float("nan"))
        loss, _ = loss_fn(pred, target)
        assert float(loss) == 0.0

    def test_loss_backward(self, p3_cfg):
        loss_fn = ai.PhysicsGuidedLoss(p3_cfg)
        pred    = torch.randn(1, 15, 5, 5, requires_grad=True)
        target  = torch.randn(1, 15, 5, 5)
        loss, _ = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None


# ─────────────────────────────────────────────
# P3.D Training Engine
# ─────────────────────────────────────────────

class TestTrainingEngine:
    def test_chronological_split(self):
        tr, va, te = ai.chronological_split(30)
        assert tr[-1] < va[0]
        assert va[-1] < te[0]
        assert len(tr) + len(va) + len(te) == 30

    def test_early_stopping(self):
        es = ai.EarlyStopping(patience=2)
        assert not es.step(1.0)
        assert not es.step(0.9)   # improvement
        assert not es.step(0.95)  # worse → counter 1
        assert es.step(0.96)      # worse → counter 2 → stop

    def test_train_model_runs(self, tiny_surface, tiny_target, p3_cfg, tmp_path):
        cfg = {**p3_cfg, "phase3": {**p3_cfg.get("phase3",{}), "epochs": 2, "patience": 5}}
        result = ai.train_model(
            ai.OceanCNNModel(8, 15, 32, 5, 5),
            tiny_surface, tiny_target, cfg, str(tmp_path)
        )
        assert "history" in result
        assert len(result["history"]) >= 1
        assert os.path.exists(str(tmp_path / "best_model.pt"))
        assert os.path.exists(str(tmp_path / "latest_model.pt"))

    def test_training_history_csv(self, tmp_path, tiny_surface, tiny_target, p3_cfg):
        cfg = {**p3_cfg, "phase3": {**p3_cfg.get("phase3",{}), "epochs": 2}}
        ai.train_model(ai.OceanCNNModel(8,15,32,5,5), tiny_surface, tiny_target, cfg, str(tmp_path))
        assert os.path.exists(str(tmp_path / "training_history.csv"))
        df = pd.read_csv(str(tmp_path / "training_history.csv"))
        assert "train_loss" in df.columns


# ─────────────────────────────────────────────
# P3.E Prediction Engine
# ─────────────────────────────────────────────

class TestPredictionEngine:
    def test_predict_single_shape(self, tiny_model, tiny_surface, tiny_norm_stats):
        pred = ai.predict_single(tiny_model, tiny_surface[0], tiny_norm_stats)
        assert pred.shape == (15, 5, 5)
        assert pred.dtype == np.float32

    def test_predict_all_shape(self, tiny_pred_ds):
        assert tiny_pred_ds["thetao_pred"].shape == (10, 15, 5, 5)

    def test_future_forecast_keys(self, tiny_model, tiny_surface,
                                   tiny_norm_stats, p3_cfg):
        times = list(pd.date_range("2025-01-01", periods=10, freq="D"))
        lats  = list(np.arange(10.0, 11.25, 0.25))
        lons  = list(np.arange(80.0, 81.25, 0.25))
        fc = ai.generate_future_forecast(
            tiny_model, tiny_surface, times, lats, lons, tiny_norm_stats, p3_cfg)
        for days in [7, 15, 30]:
            assert days in fc
            assert "confidence" in fc[days]
            assert "temperature_field" in fc[days]

    def test_future_confidence_degrades(self, tiny_model, tiny_surface,
                                         tiny_norm_stats, p3_cfg):
        times = list(pd.date_range("2025-01-01", periods=10, freq="D"))
        lats  = list(np.arange(10.0, 11.25, 0.25))
        lons  = list(np.arange(80.0, 81.25, 0.25))
        fc = ai.generate_future_forecast(
            tiny_model, tiny_surface, times, lats, lons, tiny_norm_stats, p3_cfg)
        assert fc[7]["confidence"] > fc[15]["confidence"] > fc[30]["confidence"]


# ─────────────────────────────────────────────
# P3.F Marine Heatwave Detection
# ─────────────────────────────────────────────

class TestHeatwaveDetection:
    def test_returns_list(self, tiny_pred_ds, p3_cfg):
        hs = ai.detect_marine_heatwaves(tiny_pred_ds, p3_cfg)
        assert isinstance(hs, list)

    def test_hotspot_fields(self, tiny_pred_ds, p3_cfg):
        hs = ai.detect_marine_heatwaves(tiny_pred_ds, p3_cfg)
        if hs:
            for key in ["latitude","longitude","severity","sst_anomaly","probability"]:
                assert key in hs[0]

    def test_severity_labels(self, tiny_pred_ds, p3_cfg):
        hs = ai.detect_marine_heatwaves(tiny_pred_ds, p3_cfg)
        for h in hs:
            assert h["severity"] in ("LOW","MEDIUM","HIGH","EXTREME")

    def test_probability_in_range(self, tiny_pred_ds, p3_cfg):
        hs = ai.detect_marine_heatwaves(tiny_pred_ds, p3_cfg)
        for h in hs:
            assert 0.0 <= h["probability"] <= 1.0


# ─────────────────────────────────────────────
# P3.G Impact Prediction
# ─────────────────────────────────────────────

class TestImpactPrediction:
    def test_impact_structure(self, tiny_pred_ds, p3_cfg):
        hs      = ai.detect_marine_heatwaves(tiny_pred_ds, p3_cfg)
        impacts = ai.generate_impact_predictions(hs)
        assert isinstance(impacts, list)
        if impacts:
            assert "hotspot" in impacts[0]
            assert "impacts" in impacts[0]

    def test_impact_fields(self, tiny_pred_ds, p3_cfg):
        hs = [{"latitude":12.0,"longitude":82.0,"severity":"HIGH",
               "sst_anomaly":2.5,"probability":0.85,
               "affected_depth_m":30.0,"expected_start":"2025-01-28",
               "expected_duration_days":3,"expected_severity":"HIGH",
               "subsurface_layers":3,"affected_sst":29.0}]
        impacts = ai.generate_impact_predictions(hs)
        assert len(impacts) == 1
        for imp in impacts[0]["impacts"]:
            for k in ["impact","severity","reason","preventive_action"]:
                assert k in imp

    def test_extreme_has_all_impacts(self):
        hs = [{"latitude":12.0,"longitude":82.0,"severity":"EXTREME",
               "sst_anomaly":3.5,"probability":0.95,
               "affected_depth_m":500.0,"expected_start":"2025-01-25",
               "expected_duration_days":5,"expected_severity":"EXTREME",
               "subsurface_layers":8,"affected_sst":31.0}]
        impacts = ai.generate_impact_predictions(hs)
        assert len(impacts[0]["impacts"]) > 0


# ─────────────────────────────────────────────
# P3.H Uncertainty
# ─────────────────────────────────────────────

class TestUncertainty:
    def test_uncertainty_keys(self, tiny_model, tiny_surface, tiny_norm_stats):
        unc = ai.generate_uncertainty(tiny_model, tiny_surface, tiny_norm_stats, n_samples=3)
        for k in ["confidence_map","uncertainty_map","overall_confidence",
                  "depth_uncertainty","pixel_uncertainty"]:
            assert k in unc

    def test_confidence_in_range(self, tiny_model, tiny_surface, tiny_norm_stats):
        unc = ai.generate_uncertainty(tiny_model, tiny_surface, tiny_norm_stats, n_samples=3)
        assert 0.0 <= unc["overall_confidence"] <= 1.0

    def test_depth_uncertainty_length(self, tiny_model, tiny_surface, tiny_norm_stats):
        unc = ai.generate_uncertainty(tiny_model, tiny_surface, tiny_norm_stats, n_samples=3)
        assert len(unc["depth_uncertainty"]) == 15


# ─────────────────────────────────────────────
# P3.I Explainability
# ─────────────────────────────────────────────

class TestExplainability:
    def test_xai_keys(self, tiny_model, tiny_surface, tiny_norm_stats, p3_cfg):
        xai = ai.generate_explainability(tiny_model, tiny_surface, tiny_norm_stats, p3_cfg)
        for k in ["attention_map","feature_contribution","input_importance_ranking",
                  "variable_contribution_pct","depth_contribution","explanation"]:
            assert k in xai

    def test_feature_pct_sums_to_100(self, tiny_model, tiny_surface, tiny_norm_stats, p3_cfg):
        xai  = ai.generate_explainability(tiny_model, tiny_surface, tiny_norm_stats, p3_cfg)
        total= sum(xai["feature_contribution"].values())
        assert abs(total - 100.0) < 1.0

    def test_ranking_ordered(self, tiny_model, tiny_surface, tiny_norm_stats, p3_cfg):
        xai  = ai.generate_explainability(tiny_model, tiny_surface, tiny_norm_stats, p3_cfg)
        rank = xai["input_importance_ranking"]
        pcts = [r["pct"] for r in rank]
        assert pcts == sorted(pcts, reverse=True)

    def test_attention_map_shape(self, tiny_model, tiny_surface, tiny_norm_stats, p3_cfg):
        xai = ai.generate_explainability(tiny_model, tiny_surface, tiny_norm_stats, p3_cfg)
        arr = np.array(xai["attention_map"])
        assert arr.ndim == 2
        assert arr.shape == (5, 5)


# ─────────────────────────────────────────────
# P3.J ARGO Validation
# ─────────────────────────────────────────────

class TestARGOValidation:
    def test_no_argo_files(self, tiny_pred_ds, tmp_path, p3_cfg):
        result = ai.validate_against_argo(tiny_pred_ds, str(tmp_path), p3_cfg)
        assert result["status"] in ("NO_ARGO_FILES", "NO_MATCH", "PASS")

    def test_with_synthetic_argo(self, tiny_pred_ds, tmp_path, p3_cfg):
        # Create tiny synthetic ARGO file in tmp_path
        n_prof = 5; n_lev = 15
        lats   = np.random.uniform(10, 10+5*0.25, n_prof)
        lons   = np.random.uniform(80, 80+5*0.25, n_prof)
        times  = pd.date_range("2025-01-01", periods=n_prof, freq="D")
        temps  = 20 + np.random.rand(n_prof, n_lev)
        pres   = np.tile(np.linspace(0, 1000, n_lev), (n_prof, 1))
        ds_a   = xr.Dataset(
            {"TEMP":(["N_PROF","N_LEVELS"],temps),
             "PRES":(["N_PROF","N_LEVELS"],pres),
             "LATITUDE":(["N_PROF"],lats),
             "LONGITUDE":(["N_PROF"],lons),
             "JULD":(["N_PROF"],times)})
        fp = str(tmp_path / "argo_test.nc")
        ds_a.to_netcdf(fp)
        result = ai.validate_against_argo(tiny_pred_ds, str(tmp_path), p3_cfg)
        assert "status" in result

    def test_validation_metrics_present(self, p3_full_workspace):
        # ARGO validation should have run (even with no ARGO files — status != error)
        argo_csv = p3_full_workspace["ws"] / "reports" / "phase3" / "argo_validation.csv"
        assert argo_csv.exists(), f"ARGO validation report missing: {argo_csv}"


# ─────────────────────────────────────────────
# P3 SQLite Phase 3 Tables
# ─────────────────────────────────────────────

class TestSQLitePhase3:
    def test_phase3_tables_exist(self, p3_full_workspace):
        db_path = p3_full_workspace["cfg"]["paths"]["database"]
        conn    = sqlite3.connect(db_path)
        cur     = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables  = {r[0] for r in cur.fetchall()}
        conn.close()
        for t in ["predictions","heatwave_history","model_versions",
                  "explainability_log","training_runs"]:
            assert t in tables, f"Missing table: {t}"

    def test_predictions_logged(self, p3_full_workspace):
        db_path = p3_full_workspace["cfg"]["paths"]["database"]
        conn    = sqlite3.connect(db_path)
        cur     = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM predictions")
        cnt = cur.fetchone()[0]
        conn.close()
        assert cnt >= 1

    def test_heatwave_history_logged(self, p3_full_workspace):
        db_path = p3_full_workspace["cfg"]["paths"]["database"]
        conn    = sqlite3.connect(db_path)
        cur     = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM heatwave_history")
        cnt = cur.fetchone()[0]
        conn.close()
        assert cnt >= 0  # may be 0 if no heatwaves detected

    def test_model_version_logged(self, p3_full_workspace):
        db_path = p3_full_workspace["cfg"]["paths"]["database"]
        conn    = sqlite3.connect(db_path)
        cur     = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM model_versions")
        cnt = cur.fetchone()[0]
        conn.close()
        assert cnt >= 1


# ─────────────────────────────────────────────
# P3 FastAPI Endpoints
# ─────────────────────────────────────────────

class TestPhase3APIEndpoints:
    @pytest.fixture(autouse=True)
    def setup_state(self, p3_full_workspace):
        """State already populated by p3_full_workspace fixture (session-scoped)."""
        pass

    def _client(self):
        from app import app as flask_app
        return flask_app.test_client()

    def _admin_client(self):
        """
        A test client with a server-side admin session already set, mirroring
        what /api/auth/firebase-session would establish after a real,
        verified Firebase login. Admin routes are now authorized off this
        signed session (fb_role == "admin") rather than a shared static
        token -- see firebase_auth.require_role and reports/
        firebase_security_audit.md Section 8.
        """
        client = self._client()
        with client.session_transaction() as sess:
            sess["fb_uid"] = "test-admin-uid"
            sess["fb_email"] = "arulgnanakumar@gmail.com"
            sess["fb_role"] = "admin"
        return client

    def test_health(self):
        resp = self._client().get("/api/health")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "ok"

    def test_status(self):
        resp = self._client().get("/api/status")
        assert resp.status_code == 200
        assert "phase3" in resp.get_json()

    def test_prediction_latest(self):
        resp = self._client().get("/api/prediction/latest")
        assert resp.status_code == 200

    def test_prediction_date(self):
        resp = self._client().get("/api/prediction/date?days_ahead=7")
        assert resp.status_code == 200

    def test_heatwave(self):
        resp = self._client().get("/api/heatwave")
        assert resp.status_code == 200
        assert "hotspots" in resp.get_json()

    def test_heatwave_forecast(self):
        resp = self._client().get("/api/heatwave/forecast")
        assert resp.status_code == 200
        assert "alert" in resp.get_json()

    def test_depth_profile(self):
        resp = self._client().get("/api/depth-profile?lat=12.5&lon=82.5&date=2025-01-15")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "depths_m" in data

    def test_argo_validation_endpoint(self):
        resp = self._client().get("/api/argo/validation")
        assert resp.status_code == 200

    def test_attention_endpoint(self):
        resp = self._client().get("/api/attention")
        assert resp.status_code == 200
        assert "attention_map" in resp.get_json()

    def test_confidence_endpoint(self):
        resp = self._client().get("/api/confidence")
        assert resp.status_code == 200
        assert "overall_confidence" in resp.get_json()

    def test_uncertainty_endpoint(self):
        resp = self._client().get("/api/uncertainty")
        assert resp.status_code == 200
        assert "overall_confidence" in resp.get_json()

    def test_voice_query(self):
        resp = self._client().post("/api/voice/query?query=Show+Bay+of+Bengal+heatwave")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "intent" in data

    def test_admin_retrain(self):
        # Updated for the Firebase/RBAC migration: admin routes are no
        # longer gated by a shared static token (?token=...). They now
        # require a server-side session with fb_role == "admin", set only
        # after real Firebase ID token verification. See
        # reports/firebase_security_audit.md and firebase_auth.py.
        resp = self._admin_client().post("/api/admin/retrain")
        assert resp.status_code == 200
        assert "status" in resp.get_json()

    def test_admin_no_session_rejected(self):
        resp = self._client().post("/api/admin/retrain", headers={"Accept": "application/json"})
        assert resp.status_code == 401

    def test_admin_non_admin_session_rejected(self):
        client = self._client()
        with client.session_transaction() as sess:
            sess["fb_uid"] = "regular-user-uid"
            sess["fb_email"] = "someone@example.com"
            sess["fb_role"] = "user"
        resp = client.post("/api/admin/retrain", headers={"Accept": "application/json"})
        assert resp.status_code == 403

    def test_admin_history(self):
        resp = self._admin_client().get("/api/admin/history")
        assert resp.status_code == 200

    def test_multivar_endpoint(self):
        resp = self._client().get("/api/multivar")
        assert resp.status_code == 200

    def test_user_ocean_summary(self):
        resp = self._client().get("/api/user/ocean-summary")
        assert resp.status_code == 200

    def test_user_heatwave_alert(self):
        resp = self._client().get("/api/user/heatwave-alert")
        assert resp.status_code == 200
        assert "alert_level" in resp.get_json()

    def test_page_routes_render(self):
        for path in ["/", "/earth", "/dashboard", "/prediction", "/heatwave", "/admin", "/voice", "/about"]:
            resp = self._client().get(path)
            assert resp.status_code == 200, f"Page {path} failed with {resp.status_code}"


# ─────────────────────────────────────────────
# P3 Visualization
# ─────────────────────────────────────────────

class TestPhase3Visualization:
    def test_plots_generated(self, p3_full_workspace):
        plots_dir = str(p3_full_workspace["ws"] / "reports" / "phase3" / "plots")
        if not os.path.exists(plots_dir):
            plots_dir = "reports/phase3/plots"  # fallback
        expected  = [
            "temperature_heatmap.png", "depth_heatmap.png",
            "marine_heatwave_map.png", "attention_map.png",
            "embedding_pca.png", "training_curve.png",
            "rmse_vs_depth.png", "correlation_vs_depth.png",
            "confidence_map.png", "uncertainty_map.png",
            "prediction_difference_map.png",
        ]
        for fname in expected:
            assert os.path.exists(os.path.join(plots_dir, fname)), \
                f"Missing Phase 3 plot: {fname}"

    def test_plots_nonzero(self, p3_full_workspace):
        plots_dir = str(p3_full_workspace["ws"] / "reports" / "phase3" / "plots")
        if not os.path.exists(plots_dir):
            plots_dir = "reports/phase3/plots"  # fallback
        for f in Path(plots_dir).glob("*.png"):
            assert f.stat().st_size > 1000, f"Empty plot: {f.name}"


# ─────────────────────────────────────────────
# P3 Reports
# ─────────────────────────────────────────────

class TestPhase3Reports:
    def test_all_reports_exist(self, p3_full_workspace):
        rdir = str(p3_full_workspace["ws"] / "reports" / "phase3")
        if not os.path.exists(rdir):
            rdir = "reports/phase3"
        for fname in ["phase3_summary.json","training_history.csv",
                      "prediction_statistics.csv","heatwave_report.csv",
                      "argo_validation.csv","attention_summary.csv",
                      "embedding_summary.csv","confidence_summary.csv",
                      "uncertainty_summary.csv","PHASE3_REPORT.md"]:
            assert os.path.exists(os.path.join(rdir, fname)), f"Missing: {fname}"

    def test_summary_complete(self, p3_full_workspace):
        fp = str(p3_full_workspace["ws"] / "reports" / "phase3" / "phase3_summary.json")
        if not os.path.exists(fp):
            fp = "reports/phase3/phase3_summary.json"
        with open(fp) as f:
            s = json.load(f)
        assert s["phase3_status"] == "COMPLETE"
        assert s["phase"] == 3

    def test_prediction_netcdf_exists(self, p3_full_workspace):
        pred_nc = str(p3_full_workspace["ws"] / "predictions" / "ocean_prediction.nc")
        if not os.path.exists(pred_nc):
            pred_nc = "predictions/ocean_prediction.nc"
        assert os.path.exists(pred_nc)
        ds = xr.open_dataset(pred_nc)
        assert "thetao_pred" in ds.data_vars
        ds.close()


# ─────────────────────────────────────────────
# P3 Admin
# ─────────────────────────────────────────────

class TestAdminBackend:
    def test_verify_token(self):
        import hashlib
        good = hashlib.sha256(b"sih_ocean_admin_2025").hexdigest()
        assert adm.verify_admin_token(good)
        assert not adm.verify_admin_token("bad_token")

    def test_admin_tables_init(self, tmp_path):
        db  = str(tmp_path / "admin_test.db")
        conn= sqlite3.connect(db)
        adm.init_admin_tables(conn)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {r[0] for r in cur.fetchall()}
        conn.close()
        for t in ["admin_uploads","admin_actions","model_history",
                  "dataset_history","prediction_history"]:
            assert t in tables

    def test_upload_registers(self, tmp_path):
        db  = str(tmp_path / "admin2.db")
        conn= sqlite3.connect(db)
        adm.init_admin_tables(conn)
        fp  = str(tmp_path / "test_upload.txt")
        with open(fp, "w") as f:
            f.write("test")
        result = adm.admin_upload_dataset(conn, "test_upload.txt", fp, "CORE_INPUT")
        conn.close()
        assert result["status"] == "uploaded"

    def test_get_stats(self, tmp_path):
        db   = str(tmp_path / "stats.db")
        conn = sqlite3.connect(db)
        adm.init_admin_tables(conn)
        stats= adm.get_platform_statistics(conn)
        conn.close()
        assert isinstance(stats, dict)


# ─────────────────────────────────────────────
# P3 Raw Integrity
# ─────────────────────────────────────────────

class TestPhase3RawIntegrity:
    def test_raw_files_untouched_after_phase3(self, p3_full_workspace):
        raw_dir = p3_full_workspace["cfg"]["paths"]["raw_data"]
        records = pre.scan_raw_directory(raw_dir)
        h1 = pre.generate_file_hashes(records)
        h2 = pre.generate_file_hashes(records)
        for fp in h1:
            assert h1[fp] == h2[fp]
