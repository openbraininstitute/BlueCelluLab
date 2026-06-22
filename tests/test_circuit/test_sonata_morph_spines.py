"""Tests for SonataCircuitAccess morph_filepath with morph-spines format."""

from unittest.mock import MagicMock


class TestSonataMorphSpinesPath:
    """Tests for morph_filepath() morph-spines support."""

    def test_morph_spines_alternate_morphologies(self):
        """morph_filepath should return nested path when alternate_morphologies has morph-spines."""
        from bluecellulab.circuit.circuit_access.sonata_circuit_access import (
            SonataCircuitAccess,
        )
        from bluecellulab.circuit import CellId

        # Create a mock SonataCircuitAccess without full init
        access = SonataCircuitAccess.__new__(SonataCircuitAccess)

        mock_pop = MagicMock()
        mock_pop.config.get = lambda key, default="": {
            "alternate_morphologies": {"morph-spines": "/data/cells_with_spines.h5"},
        }.get(key, default)
        mock_pop.get = lambda cell_id, properties=None: {"morphology": "cell_42"}

        access._circuit = MagicMock()
        access._circuit.nodes = {"NodeA": mock_pop}

        result = access.morph_filepath(CellId("NodeA", 1))
        assert result == "/data/cells_with_spines.h5/morphology/cell_42.h5"

    def test_h5v1_still_works(self):
        """morph_filepath should still return h5v1 paths when morph-spines is not present."""
        from bluecellulab.circuit.circuit_access.sonata_circuit_access import (
            SonataCircuitAccess,
        )
        from bluecellulab.circuit import CellId

        access = SonataCircuitAccess.__new__(SonataCircuitAccess)

        mock_pop = MagicMock()
        mock_pop.config.get = lambda key, default="": {
            "alternate_morphologies": {"h5v1": "/data/merged.h5"},
        }.get(key, default)
        mock_pop.get = lambda cell_id, properties=None: {"morphology": "cell_99"}

        access._circuit = MagicMock()
        access._circuit.nodes = {"NodeA": mock_pop}

        result = access.morph_filepath(CellId("NodeA", 1))
        assert result == "/data/merged.h5/cell_99.h5"

    def test_morph_spines_takes_priority_over_h5v1(self):
        """morph-spines should be checked before h5v1."""
        from bluecellulab.circuit.circuit_access.sonata_circuit_access import (
            SonataCircuitAccess,
        )
        from bluecellulab.circuit import CellId

        access = SonataCircuitAccess.__new__(SonataCircuitAccess)

        mock_pop = MagicMock()
        mock_pop.config.get = lambda key, default="": {
            "alternate_morphologies": {
                "h5v1": "/data/merged.h5",
                "morph-spines": "/data/spines.h5",
            },
        }.get(key, default)
        mock_pop.get = lambda cell_id, properties=None: {"morphology": "cell_A"}

        access._circuit = MagicMock()
        access._circuit.nodes = {"NodeA": mock_pop}

        result = access.morph_filepath(CellId("NodeA", 1))
        assert result == "/data/spines.h5/morphology/cell_A.h5"

    def test_auto_detection_morphologies_dir_is_h5(self, synthetic_morph_spines_h5):
        """Auto-detect morph-spines when morphologies_dir points to an .h5 file."""
        from bluecellulab.circuit.circuit_access.sonata_circuit_access import (
            SonataCircuitAccess,
        )
        from bluecellulab.circuit import CellId

        access = SonataCircuitAccess.__new__(SonataCircuitAccess)

        mock_pop = MagicMock()

        def config_get(key, default=""):
            if key == "alternate_morphologies":
                return None
            if key == "morphologies_dir":
                return synthetic_morph_spines_h5
            return default

        mock_pop.config.get = config_get
        mock_pop.get = lambda cell_id, properties=None: {"morphology": "test_cell"}

        access._circuit = MagicMock()
        access._circuit.nodes = {"NodeA": mock_pop}

        result = access.morph_filepath(CellId("NodeA", 1))
        assert "morphology/test_cell.h5" in result
        assert synthetic_morph_spines_h5 in result
