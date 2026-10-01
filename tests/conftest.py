"""pytest fixture for the standalone-runnable test suite in this directory
(the tests take a ``tmp`` path; run plain with
``python tests/test_credential_logic.py`` as well)."""
import pytest


@pytest.fixture
def tmp(tmp_path):
    return tmp_path
