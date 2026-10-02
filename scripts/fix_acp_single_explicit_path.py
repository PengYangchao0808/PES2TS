"""Keep explicit one-driver full-boundary plans through the ACP adapter."""
from pathlib import Path
import shutil
root=Path('/mnt/e/Calculations/Common_Script/Auto_Calc_Platform/ACP_V1_20260811')
p=root/'src/acp/backends/orca.py';old=p.read_text()
a='        if len(drive_coordinates) > 1:'
b='        if len(drive_coordinates) > 1 or plan.fixed_endpoints or any(c.values for c in plan.coordinates):'
if b not in old:
    if a not in old:raise ValueError('adapter anchor missing')
    backup=Path('/mnt/e/Calculations/AI4S_ML_Studys/PES2TS/outputs/acp_synchronized_patch_20261002/single_adapter_orca.py')
    if not backup.exists():shutil.copyfile(p,backup)
    p.write_text(old.replace(a,b,1))
print('Explicit one-driver plans retain full path contract')
