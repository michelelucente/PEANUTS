# PEANUTS cluster tuning (Numba + parallel jobs)

Quick checklist to keep startup fast and avoid repeated JIT on clusters, including parallel job launchers.

## To-do (setup)
- Set a job-local cache: export `NUMBA_CACHE_DIR=${TMPDIR:-/tmp}/numba_cache_$JOB_ID` (or `$SLURM_JOBID`/`$PBS_JOBID`) before running.
- Warm once per job: call a tiny PEANUTS run (any short input) to populate the cache, then run the real workload.
- Avoid cleaning the cache mid-job: don’t delete `NUMBA_CACHE_DIR` between task invocations inside the same job.
- Use consistent env: same Python/Numba version and CPU ISA across nodes; otherwise a recompile is expected.

## If you still see recompilation
- Verify cache location: `echo $NUMBA_CACHE_DIR` and check it exists and is writable; per-job directories prevent write contention.
- Inspect cache logs: run with `NUMBA_DEBUG_CACHE=1` to see “data loaded/saved” messages; “saved” on every call signals cache misses.
- Check CPU/ABI drift: mixing AVX2-only nodes with AVX512 nodes or differing Python/Numba versions triggers rebuilds.
- Confirm single process owns the cache dir: avoid sharing a writable cache across parallel jobs; prefer read-only prewarmed cache or per-job cache.
- Ensure no read-only filesystem: caches on strict read-only NFS will force recompilation to a temp dir every run.

## Parallel usage tips
- Each process can read the same prewarmed cache; only writes cause contention. Use per-job caches for writers, or distribute a prebuilt read-only cache.
- For embarrassingly parallel sweeps in one job, warm once, then launch child processes that inherit `NUMBA_CACHE_DIR`.
- Avoid `multiprocessing` spawn storms right at cold start; warm in the parent, then fork/spawn children.

## Prewarm example (bash)
```bash
export NUMBA_CACHE_DIR=${TMPDIR:-/tmp}/numba_cache_$SLURM_JOBID
mkdir -p "$NUMBA_CACHE_DIR"
# cheap warm-up
python run_peanuts.py -f examples/solar_earth_test.yaml >/dev/null
# now run your heavier sweep
python run_peanuts.py -f your_config.yaml
```
