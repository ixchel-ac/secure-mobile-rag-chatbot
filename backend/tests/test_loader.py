"""Tests for the Synthea data loader."""

import pytest
from pathlib import Path

from app.ingestion.loader import (
    extract_uuid_from_filename,
    parse_patient_name_from_header,
    load_patients_csv,
    load_text_files,
    load_all,
    PatientRecord,
)


class TestExtractUuidFromFilename:
    """Tests for extract_uuid_from_filename function."""

    def test_standard_synthea_filename(self):
        filename = "Adah626_Flo729_Klein929_f7adc5f3-fc3c-f928-7cd1-d8d3a49f3c38.txt"
        assert extract_uuid_from_filename(filename) == "f7adc5f3-fc3c-f928-7cd1-d8d3a49f3c38"

    def test_two_part_name(self):
        filename = "John123_Doe456_a1b2c3d4-e5f6-7890-abcd-ef1234567890.txt"
        assert extract_uuid_from_filename(filename) == "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

    def test_uppercase_uuid(self):
        filename = "Name_A1B2C3D4-E5F6-7890-ABCD-EF1234567890.txt"
        assert extract_uuid_from_filename(filename) == "A1B2C3D4-E5F6-7890-ABCD-EF1234567890"

    def test_no_uuid_returns_empty(self):
        filename = "invalid_filename.txt"
        assert extract_uuid_from_filename(filename) == ""

    def test_partial_uuid_returns_empty(self):
        filename = "Name_f7adc5f3-fc3c.txt"
        assert extract_uuid_from_filename(filename) == ""


class TestParsePatientNameFromHeader:
    """Tests for parse_patient_name_from_header function."""

    def test_standard_header(self):
        text = "Adah626 Flo729 Klein929\n=======================\nRace: White"
        assert parse_patient_name_from_header(text) == "Adah626 Flo729 Klein929"

    def test_two_part_name(self):
        text = "John Doe\n========\nSome content"
        assert parse_patient_name_from_header(text) == "John Doe"

    def test_empty_text(self):
        assert parse_patient_name_from_header("") == ""

    def test_whitespace_only(self):
        assert parse_patient_name_from_header("   \n\n  ") == ""

    def test_strips_whitespace(self):
        text = "  John Doe  \n========"
        assert parse_patient_name_from_header(text) == "John Doe"


class TestLoadPatientsCsv:
    """Tests for load_patients_csv function."""

    def test_load_csv(self, tmp_path):
        csv_content = """Id,BIRTHDATE,SSN,FIRST,MIDDLE,LAST,ADDRESS,CITY,STATE,ZIP
abc-123,1990-01-15,123-45-6789,John,William,Doe,123 Main St,Boston,MA,02101
def-456,1985-05-20,987-65-4321,Jane,,Smith,456 Oak Ave,Cambridge,MA,02139"""

        csv_file = tmp_path / "patients.csv"
        csv_file.write_text(csv_content)

        lookup = load_patients_csv(csv_file)

        assert "abc-123" in lookup
        assert "def-456" in lookup

        john = lookup["abc-123"]
        assert john["patient_id"] == "abc-123"
        assert john["ssn"] == "123-45-6789"
        assert john["dob"] == "1990-01-15"
        assert john["name"] == "John Doe"
        assert john["full_name"] == "John William Doe"
        assert "123 Main St" in john["address"]

        jane = lookup["def-456"]
        assert jane["name"] == "Jane Smith"
        assert jane["middle"] == ""


class TestLoadTextFiles:
    """Tests for load_text_files function."""

    def test_load_multiple_files(self, tmp_path):
        (tmp_path / "patient1.txt").write_text("Patient One\n=====\nContent 1")
        (tmp_path / "patient2.txt").write_text("Patient Two\n=====\nContent 2")

        results = load_text_files(tmp_path)

        assert len(results) == 2
        filenames = [r[0] for r in results]
        assert "patient1.txt" in filenames
        assert "patient2.txt" in filenames

    def test_sorted_by_filename(self, tmp_path):
        (tmp_path / "z_patient.txt").write_text("Z Patient")
        (tmp_path / "a_patient.txt").write_text("A Patient")

        results = load_text_files(tmp_path)

        assert results[0][0] == "a_patient.txt"
        assert results[1][0] == "z_patient.txt"

    def test_empty_directory(self, tmp_path):
        results = load_text_files(tmp_path)
        assert results == []

    def test_ignores_non_txt_files(self, tmp_path):
        (tmp_path / "patient.txt").write_text("Valid")
        (tmp_path / "patient.csv").write_text("Ignored")
        (tmp_path / "patient.json").write_text("Ignored")

        results = load_text_files(tmp_path)

        assert len(results) == 1
        assert results[0][0] == "patient.txt"


class TestLoadAll:
    """Integration tests for load_all function."""

    @pytest.fixture
    def synthea_data(self, tmp_path):
        """Create mock Synthea data structure."""
        text_dir = tmp_path / "text"
        csv_dir = tmp_path / "csv"
        text_dir.mkdir()
        csv_dir.mkdir()

        # Create patients.csv
        csv_content = """Id,BIRTHDATE,SSN,FIRST,MIDDLE,LAST,ADDRESS,CITY,STATE,ZIP
f7adc5f3-fc3c-f928-7cd1-d8d3a49f3c38,1990-01-15,123-45-6789,Adah626,Flo729,Klein929,123 Main St,Boston,MA,02101
11bfe293-447c-ee5d-9567-7f998fc4709b,1985-05-20,987-65-4321,Ahmad985,,Stracke611,456 Oak Ave,Cambridge,MA,02139"""

        (csv_dir / "patients.csv").write_text(csv_content)

        # Create text files matching the UUIDs
        text1 = """Adah626 Flo729 Klein929
========================
Race: White
Ethnicity: Non-Hispanic
"""
        (text_dir / "Adah626_Flo729_Klein929_f7adc5f3-fc3c-f928-7cd1-d8d3a49f3c38.txt").write_text(text1)

        text2 = """Ahmad985 Stracke611
====================
Race: Asian
"""
        (text_dir / "Ahmad985_Stracke611_11bfe293-447c-ee5d-9567-7f998fc4709b.txt").write_text(text2)

        return text_dir, csv_dir

    def test_load_all_matches_by_uuid(self, synthea_data):
        text_dir, csv_dir = synthea_data

        records = load_all(text_dir, csv_dir)

        assert len(records) == 2

        # Find record by UUID
        adah = next(r for r in records if r.patient_id == "f7adc5f3-fc3c-f928-7cd1-d8d3a49f3c38")
        assert adah.patient_name == "Adah626 Flo729 Klein929"
        assert adah.phi_entities["ssn"] == "123-45-6789"
        assert adah.phi_entities["dob"] == "1990-01-15"

    def test_load_all_handles_unmatched(self, tmp_path):
        text_dir = tmp_path / "text"
        csv_dir = tmp_path / "csv"
        text_dir.mkdir()
        csv_dir.mkdir()

        # Empty CSV (just header)
        (csv_dir / "patients.csv").write_text("Id,BIRTHDATE,SSN,FIRST,MIDDLE,LAST,ADDRESS,CITY,STATE,ZIP\n")

        # Text file with UUID not in CSV
        (text_dir / "Unknown_Person_aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee.txt").write_text("Unknown Person\n====")

        records = load_all(text_dir, csv_dir)

        assert len(records) == 1
        assert records[0].patient_id == "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        assert records[0].phi_entities == {}

    def test_load_all_returns_patient_records(self, synthea_data):
        text_dir, csv_dir = synthea_data

        records = load_all(text_dir, csv_dir)

        assert all(isinstance(r, PatientRecord) for r in records)
        assert all(r.raw_text for r in records)
        assert all(r.source_file.endswith(".txt") for r in records)


class TestWithRealData:
    """Tests using actual Synthea data (skipped if not available)."""

    @pytest.fixture
    def real_data_paths(self):
        project_root = Path(__file__).parent.parent.parent
        text_dir = project_root / "data" / "synthea" / "text"
        csv_dir = project_root / "data" / "synthea" / "csv"

        if not text_dir.exists() or not csv_dir.exists():
            pytest.skip("Real Synthea data not available")

        return text_dir, csv_dir

    def test_real_data_loads(self, real_data_paths):
        text_dir, csv_dir = real_data_paths

        records = load_all(text_dir, csv_dir)

        assert len(records) > 0
        # Most records should match
        matched = sum(1 for r in records if r.phi_entities)
        assert matched / len(records) > 0.9, "Expected >90% match rate with UUID-based lookup"

    def test_real_uuid_extraction(self, real_data_paths):
        text_dir, _ = real_data_paths

        txt_files = list(text_dir.glob("*.txt"))
        assert len(txt_files) > 0

        # Check that UUIDs can be extracted from all files
        for txt_file in txt_files[:10]:  # Check first 10
            uuid = extract_uuid_from_filename(txt_file.name)
            assert uuid, f"Failed to extract UUID from {txt_file.name}"
            assert len(uuid) == 36, f"Invalid UUID length: {uuid}"
