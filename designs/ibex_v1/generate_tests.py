import os
import argparse
import yaml
import copy

from glob import glob

import numpy as np

class DvTestTemplate:
  def __init__(self, yaml_path: str):
    with open(yaml_path, "r") as f:
      self.content = yaml.load(f, Loader=yaml.FullLoader)
    self.template_name = os.path.basename(yaml_path)
  def generate(self, id_str):
    """Generate a test case from the template"""
    ret = copy.deepcopy(self.content)
    ret["test"] = ret["test"].replace("<id>", id_str)
    ret["description"] = ret["description"].replace("<id>", id_str)
    knobs = {}
    for k, v in ret["gen_opts"].items():
      assert "type" in v, f"Missing type for {k}"
      if "forced_default" in v:
        knobs[k] = v["forced_default"]
        continue

      if v["type"] == "int":
        assert "min_val" in v and "max_val" in v, (
            f"Missing min_val or max_val for {k}")
        min_val = v["min_val"]
        max_val = v["max_val"]
        knobs[k] = int(np.random.randint(min_val, max_val + 1))
      elif v["type"] == "bool":
        knobs[k] = int(np.random.choice([True, False]))
      elif v["type"] == "choice":
        assert "values" in v, f"Missing values for {k}"
        knobs[k] = str(np.random.choice(v["values"]))
      else:
        assert False, f"Unknown type {v['type']}"
    ret["gen_opts"] = "\n".join(f"+{k}={v}" for k, v in knobs.items())
    ret["template_name"] = self.template_name

    return [ret]

def generate_tests(template_dir: str, output_dir: str, num_tests: int):
  """Generate tests given test template"""
  def str_presenter(dumper, data):
    if len(data.splitlines()) > 1:  # check for multiline string
      return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)

  yaml.add_representer(str, str_presenter)

  # Load test templates
  templates = []
  for yaml_path in glob(os.path.join(template_dir, "*.yaml")):
    templates.append(DvTestTemplate(yaml_path))

  os.makedirs(output_dir, exist_ok=True)

  # Generate tests
  for i in range(num_tests):
    # Sample a template from loaded templates
    template = np.random.choice(templates)
    id_str = f"{i:05d}"
    with open(os.path.join(output_dir, f"{id_str}.yaml"), "w") as f:
      yaml.dump(template.generate(id_str), f)

  print(f"Generated {num_tests} test cases in '{output_dir}'.")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-td", "--template_dir", required=True,
                      help="Directory containing the test template files.")
  parser.add_argument("-od", "--output_dir",
                      default="generated/tests",
                      help=("Path to the directory where the generatd test "
                             "YAML files will be written"))
  parser.add_argument("-nt", "--num_tests", type=int,
                      default=10, help="Number of tests to generate")
  args = parser.parse_args()
  generate_tests(args.template_dir, args.output_dir, args.num_tests)


if __name__ == "__main__":
  main()
