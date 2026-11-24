"""One and two MPI ranks must agree for both public plasma prescriptions."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest


def write_spectra(path, expected_ranks):
    from mpi4py import MPI
    from SpyDust.SpyDust import SpyDust

    comm = MPI.COMM_WORLD
    assert comm.size == expected_ranks
    env = dict(nh=30, T=100., Chi=1., xh=1.2e-3, xC=3e-4, y=0, gamma=0, dipole=9.3, line=7)
    spectra = []
    for legacy in (False, True):
        spectrum = SpyDust(env, n_freq=24, Ndipole=2, N_angular_Omega=64,
                           single_beta=True, spdust_plasma=legacy)
        assert np.isfinite(spectrum).all() and spectrum[1].max() > 0
        np.testing.assert_array_equal(spectrum, comm.bcast(spectrum if comm.rank == 0 else None))
        spectra.append(spectrum)
    if comm.size > 1:
        from SpyDust.SED import SED
        from SpyDust.mpiutil import parallel_map, parallel_jobs_no_gather_no_return

        def fail_on_second_item(i):
            if i == 1:
                raise ValueError("failed item")
            return i

        with pytest.raises(RuntimeError, match="failed item"):
            parallel_map(fail_on_second_item, [0, 1])
        with pytest.raises(RuntimeError, match="failed item"):
            parallel_jobs_no_gather_no_return(lambda paired: fail_on_second_item(next(paired)[0]),
                                               [[(0,)], [(1,)]])
        np.random.seed(78731 + comm.rank)
        items = list(range(18))
        random_order = parallel_map(int, items, method="rand", root=comm.size-1)
        if comm.rank == comm.size-1:
            assert random_order == items
        else:
            assert random_order is None
        # Some callers pass the file name only to the output rank.
        output = Path(path).parent if comm.rank == 0 else None
        with pytest.raises(OSError, match="rank 0"):
            SpyDust(env, n_freq=4, Ndipole=1, N_angular_Omega=64,
                    single_beta=True, output_file=output)
        output = Path(path).with_suffix(".txt")
        valid = SpyDust(env, n_freq=4, Ndipole=1, N_angular_Omega=64,
                        single_beta=True, output_file=output if comm.rank == 0 else None)
        np.testing.assert_allclose(np.loadtxt(output).T, valid, rtol=1e-6, atol=0)
        with pytest.raises(ValueError, match="weights"):
            SED([1.], np.ones((2, 2, 4)), [-0.2, -0.1], [1., 2., 3., 4.],
                [0., 1.], [[1., 1.], [-1., 2.]])
    if comm.rank == 0:
        np.save(path, spectra)


def test_mpi_spectra(tmp_path):
    import SpyDust

    launcher = shutil.which("mpiexec")
    if launcher is None:
        pytest.skip("mpiexec is required for the two-rank comparison")
    # Run the child interpreters against the same source tree or installed wheel.
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [
        str(Path(SpyDust.__file__).resolve().parents[1]), env.get("PYTHONPATH"),
    ]))
    for ranks in (1, 2):
        target = tmp_path / f"spectra-{ranks}.npy"
        cmd = [sys.executable, __file__, str(target), str(ranks)]
        if ranks > 1:
            cmd = [launcher, "-n", str(ranks)] + cmd
        subprocess.run(cmd, cwd=tmp_path, env=env, timeout=120, check=True)
    serial, parallel = (np.load(tmp_path / f"spectra-{n}.npy") for n in (1, 2))
    np.testing.assert_allclose(parallel, serial, rtol=1e-12, atol=0)


if __name__ == "__main__":
    write_spectra(sys.argv[1], int(sys.argv[2]))
