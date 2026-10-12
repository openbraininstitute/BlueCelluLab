# Copyright 2023-2024 Blue Brain Project / EPFL

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for test-only operating-system helpers."""

import os

import pytest

from tests.helpers.os_utils import cwd


def test_cwd_restores_directory_when_body_raises(tmp_path):
    original = os.getcwd()

    with pytest.raises(RuntimeError, match="expected failure"):
        with cwd(tmp_path):
            assert os.getcwd() == str(tmp_path)
            raise RuntimeError("expected failure")

    assert os.getcwd() == original
