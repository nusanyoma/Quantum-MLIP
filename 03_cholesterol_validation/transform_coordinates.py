import re
import argparse
import sys
import pathlib

_here = pathlib.Path(__file__).resolve()
for _p in _here.parents:
    for _cand in (_p / "lib", _p / "code" / "code" / "lib"):
        if (_cand / "ani_transfer").is_dir():
            sys.path.insert(0, str(_cand))
            break
    else:
        continue
    break
from ani_transfer.paths import DATA_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="basename (without extension) of the *.gjf.log file")
    parser.add_argument("--data-dir", default=str(DATA_DIR),
                         help="directory containing <file>.gjf.log (default: QNNP/data)")
    args = parser.parse_args()
    file = args.file
    data_dir = pathlib.Path(args.data_dir)

    # ファイルから最終のStandard orientationを抽出
    with open(data_dir / f"{file}.gjf.log", encoding="utf-8") as f:
        lines = f.readlines()

    # 検索
    std_indices = [i for i, line in enumerate(lines) if "Standard orientation:" in line]
    start = std_indices[-1]
    table_start = start + 5
    table = []
    for line in lines[table_start:]:
        if "-----" in line or line.strip() == "":
            break
        table.append(line)

    # 原子番号を変換
    anum2sym = {1: "H", 6: "C", 8: "O"}
    xyz_lines = []
    for row in table:
        parts = row.split()
        anum = int(parts[1])
        sym = anum2sym.get(anum, "?")
        x, y, z = map(float, parts[3:6])
        xyz_lines.append(f"{sym} {x:.6f} {y:.6f} {z:.6f}")

    # xyzファイル
    with open(data_dir / f"{file}.xyz", "w") as f:
        f.write(f"{len(xyz_lines)}\n{file.split('_')[0]} optimized geometry\n")
        f.write("\n".join(xyz_lines))


if __name__ == "__main__":
    main()
