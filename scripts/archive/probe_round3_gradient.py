"""Real capability probe; outputs stay separate from the frozen round-two run."""
from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, os.environ.get("ACP_SOURCE", "/mnt/e/Calculations/Common_Script/Auto_Calc_Platform/ACP_V1_20260811/src"))
os.environ["PATH"] = "/opt/openmpi418/bin:/opt/orca_6_1_1:"+os.environ.get("PATH", "")
os.environ.setdefault("OMPI_ALLOW_RUN_AS_ROOT", "1")
os.environ.setdefault("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "1")
from gradient_backend_cccp import ORCAGradientBackend

rid = sys.argv[1] if len(sys.argv)>1 else "RXN_0000047010"
source = ROOT/"outputs/PES2TS_Demo24_continuation_round2_20261002"/rid/"endpoint"
plan = json.loads((source/"PathPlan.json").read_text())
case = json.loads((source/"result.json").read_text())
backend = ORCAGradientBackend(charge=plan["charge"], multiplicity=plan["multiplicity"],
    elements=plan["elements"], folder=ROOT/"outputs/PES2TS_Demo24_round3_20261002"/"capability_probe"/rid)
result = backend(case["frames"][0]["geometry"], "single_point")
print(json.dumps({k:v for k,v in result.items() if k not in {"gradient_evidence", "gradient_hartree_per_angstrom"}}, indent=2), flush=True)
