"""Export a snapshot for ParaView using cell-centred VTK ImageData."""
import argparse
from pathlib import Path
import numpy as np
from results import field


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--index", type=int, default=-1)
    a = p.parse_args()
    c, t, f = field(a.run, a.index)
    nx, ny, nz = (int(c[f"n{x}"]) for x in "xyz")
    extent = f"0 {nx} 0 {ny} 0 {nz}"
    origin = " ".join(c[f"origin_{x}"] for x in "xyz")
    spacing = " ".join(c[f"d{x}"] for x in "xyz")
    with a.output.open("wb") as out:
        header = f'''<?xml version="1.0"?>
<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" header_type="UInt64">
<ImageData WholeExtent="{extent}" Origin="{origin}" Spacing="{spacing}">
<Piece Extent="{extent}"><CellData Scalars="concentration">
<DataArray type="Float64" Name="concentration" format="appended" offset="0"/>
</CellData></Piece></ImageData><AppendedData encoding="raw">_'''
        out.write(header.encode())
        data = np.asarray(f, dtype="<f8")
        out.write(np.asarray([data.nbytes], dtype="<u8").tobytes())
        out.write(data.tobytes())
        out.write(b"</AppendedData></VTKFile>\n")
    print(f"Exported t={t:g} s to {a.output}")


if __name__ == "__main__":
    main()
