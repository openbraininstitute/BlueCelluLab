from .compartment import CompartmentReportWriter
from .spikes import SpikeReportWriter

REGISTRY = {
    "compartment": CompartmentReportWriter,
    "spikes": SpikeReportWriter,
}


def get_writer(report_type):
    return REGISTRY[report_type]
