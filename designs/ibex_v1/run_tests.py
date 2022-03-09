import os
import subprocess
import shutil
import argparse
import tempfile
from glob import glob


def run_test(test_path, output_dir, verification_dir):
  """Run a single test and retrieve coverage report"""
  print(f"Simulating '{test_path}' for coverage")
  SEED = 0
  test_path = os.path.abspath(test_path)
  output_dir = os.path.abspath(output_dir)
  output_dir = os.path.join(output_dir, f"seed-{SEED}")
  verification_dir = os.path.abspath(verification_dir)

  test_id = os.path.basename(test_path).replace(".yaml", "")
  test_output_dir = os.path.join(output_dir, test_id)
  urg_report_dir = os.path.join(test_output_dir, "urg_report")
  if os.path.exists(urg_report_dir):
    print(f"Skipping '{test_path}' as it has already been simulated, "
          f"see '{urg_report_dir}'\n")
    return

  os.makedirs(test_output_dir, exist_ok=True)

  testlist_dir = os.path.join(verification_dir, "riscv_dv_extension")
  testlist_path = os.path.join(testlist_dir, "testlist.yaml")
  testlist_backup_path = os.path.join(testlist_dir, ".testlist.yaml")

  # Backup original testlist.yaml
  if os.path.exists(testlist_path):
    shutil.move(testlist_path, testlist_backup_path)
  # Overwrite testlist.yaml with test to simulate
  shutil.copy(test_path, testlist_path)
  prev_dir = os.getcwd()
  os.chdir(verification_dir)  # Move to verification directory

  with tempfile.TemporaryDirectory() as temp_dir:
    temp_urg_dir = os.path.join(temp_dir, "rtl_sim", "urgReport")
    sim_log_path = os.path.join(temp_dir, f"seed-{SEED}/rtl_sim",
                                f"generated_test_{test_id}.{SEED}/sim.log")
    cmd = (f"make SEED={SEED} SIMULATOR=vcs ISS=spike ITERATIONS=1 COV=1 "
           f"OUT={temp_dir}")
    with open(os.path.join(test_output_dir, "sim.stdout"), "w") as stdout, \
            open(os.path.join(test_output_dir, "sim.stderr"), "w") as stderr:
      try:
        print(f"- Command: '{cmd}'")
        subprocess.run(
            cmd.split(" "), stdout=stdout, stderr=stderr, check=True)
        shutil.move(temp_urg_dir,  # Move urg report to test output directory
                    urg_report_dir)
        shutil.move(sim_log_path, test_output_dir)
        print(f"Simulation finished for '{test_path}'")
        print(f"- Test and coverage reports stored in '{test_output_dir}'\n")
      except subprocess.CalledProcessError as e:
        print(f"ERROR: {e}")
        reserve_dir = os.path.join(test_output_dir, "full_sim_dir")
        shutil.copytree(temp_dir, reserve_dir)
        print(f"Simulation FAILED for '{test_path}'")
        print(f"- Find failed simulation output in '{reserve_dir}'\n")

  # Keep a backup of test in the output directory
  shutil.copy(test_path, test_output_dir)
  if os.path.exists(testlist_backup_path):  # Restore original testlist.yaml
    shutil.move(testlist_backup_path, testlist_path)
  os.chdir(prev_dir)  # Come back to original directory


def run(tests_dir: str, output_dir: str, verification_dir: str):
  tests = list(glob(os.path.join(tests_dir, "*.yaml")))
  tests.sort()
  os.makedirs(output_dir, exist_ok=True)
  for tp in tests:
    run_test(tp, output_dir, verification_dir)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-td", "--tests_dir", type=str, required=True)
  parser.add_argument("-od", "--output_dir", type=str, required=True)
  parser.add_argument("-vd", "--verification_dir", type=str, required=True)
  args = parser.parse_args()
  run(args.tests_dir, args.output_dir, args.verification_dir)


if __name__ == "__main__":
  main()
