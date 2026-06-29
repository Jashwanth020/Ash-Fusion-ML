"""
run_pipelines.py — Sequentially runs the ash fusion temperature prediction pipelines.

This script executes:
- pipeline_DT.py
- pipeline_FT.py
- pipeline_HT.py
- pipeline_ST.py

It runs them sequentially to prevent locks/deadlocks in the joblib/loky multiprocessing backend.
"""

import os
import sys
import subprocess
import time

def main():
    quick = '--quick' in sys.argv
    mode_str = "QUICK" if quick else "FULL"

    scripts = [
        "pipeline_DT.py",
        "pipeline_FT.py",
        "pipeline_HT.py",
        "pipeline_ST.py"
    ]

    print("=" * 80)
    print(f"  STARTING RUN OF ALL ASH FUSION TEMPERATURE PIPELINES ({mode_str} MODE)")
    print("=" * 80)

    start_all = time.time()
    success_count = 0

    for script in scripts:
        if not os.path.exists(script):
            print(f"[ERROR] Script not found: {script}")
            continue

        print("\n" + "#" * 80)
        print(f"  RUNNING: {script}")
        print("#" * 80 + "\n")

        cmd = [sys.executable, script]
        if quick:
            cmd.append("--quick")

        start_script = time.time()
        try:
            # Run the script and let it print stdout/stderr in real time
            res = subprocess.run(cmd, check=True)
            elapsed = time.time() - start_script
            print(f"\n[SUCCESS] {script} completed in {elapsed:.2f} seconds.")
            success_count += 1
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start_script
            print(f"\n[FAILED] {script} failed after {elapsed:.2f} seconds (exit code: {e.returncode}).")
            # We continue to the next script even if one fails
        except KeyboardInterrupt:
            print("\n[ABORTED] Interrupted by user.")
            sys.exit(1)

    total_elapsed = time.time() - start_all
    print("\n" + "=" * 80)
    print(f"  RUN ALL COMPLETED")
    print(f"  Successful: {success_count}/{len(scripts)}")
    print(f"  Total Time: {total_elapsed:.2f} seconds ({total_elapsed/60.0:.2f} minutes)")
    print("=" * 80)

if __name__ == '__main__':
    main()
