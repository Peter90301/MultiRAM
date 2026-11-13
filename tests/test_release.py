import importlib.util
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def test_encoding_batches_and_energy():
    oms = load("release_oms", ROOT / "proteomic_full_pipeline/pim_hyperoms_estimator.py")
    value = oms._estimate_feram_encoding(33, 32, 100, 500e6, 50)
    assert value["num_query_batches"] == 2
    assert math.isclose(value["latency_sec"], 4e-7)
    assert math.isclose(value["energy_mj"], 33 * 50 * 1e-9)
    assert oms._estimate_feram_encoding(0, 32, 100, 500e6, 50)["latency_sec"] == 0

def test_driver_import_has_no_data_side_effects():
    driver = load("release_driver", ROOT / "run_pim_full_pipeline_timing.py")
    assert driver.DEFAULT_REF_FASTA.is_file()
    assert driver.DEFAULT_QUERY_FASTQ.is_file()
