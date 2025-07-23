# Copyright 2025 Open Brain Institute

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import MagicMock
import pytest
from bluecellulab.reports.manager import ReportManager


@pytest.fixture
def mock_config_base():
    cfg = MagicMock()
    cfg.tstart = 0.0
    cfg.get_node_sets.return_value = {"some": {}}
    cfg.get_compartment_sets.return_value = {"some": {}}
    cfg.report_file_path.return_value = "fake/path.h5"
    return cfg


def make_report_cfg(base: dict):
    return {
        "type": base.get("type", "compartment"),
        "cells": base.get("cells"),
        "sections": base.get("sections"),
        "compartments": base.get("compartments"),
        "compartment_set": base.get("compartment_set"),
    }


@pytest.mark.parametrize("key", ["cells", "sections", "compartments"])
def test_invalid_keys_with_compartment_set(mock_config_base, key):
    report_cfg = make_report_cfg({
        "type": "compartment_set",
        key: "should_not_be_here",
        "compartment_set": "some"
    })
    mock_config_base.get_report_entries.return_value = {"r": report_cfg}

    manager = ReportManager(mock_config_base, sim_dt=0.1)

    with pytest.raises(ValueError, match=f"'{key}' may not be set with 'compartment_set'"):
        manager.write_all({})


def test_missing_cells_with_compartment(mock_config_base):
    report_cfg = make_report_cfg({
        "type": "compartment",
        "cells": None
    })
    mock_config_base.get_report_entries.return_value = {"r": report_cfg}

    manager = ReportManager(mock_config_base, sim_dt=0.1)

    with pytest.raises(ValueError, match="'cells' must be specified when using compartment reports"):
        manager.write_all({})


def test_invalid_compartments_value(mock_config_base):
    report_cfg = make_report_cfg({
        "type": "compartment",
        "cells": "some",
        "compartments": "invalid"
    })
    mock_config_base.get_report_entries.return_value = {"r": report_cfg}

    manager = ReportManager(mock_config_base, sim_dt=0.1)

    with pytest.raises(ValueError, match="invalid 'compartments' value"):
        manager.write_all({})
