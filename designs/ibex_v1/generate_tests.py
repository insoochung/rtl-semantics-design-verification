import os
import argparse
import yaml
import copy

from glob import glob

import numpy as np


class DvTestTemplate:
    def __init__(self, content):
        self.content = content
        self.template_name = content["rtl_test"]

    def apply_restrictions(self, content, knobs):
        # Apply restrictions
        restrictions = self.content["gen_opts_restrictions"]
        for k, v in restrictions["forced_defaults"].items():
            if content["gen_opts"][k]["type"] in ["int", "bool"]:
                knobs[k] = int(v)
            else:
                assert content["gen_opts"][k]["type"] == "choice"
                knobs[k] = str(v)
        if "at_most_one" not in restrictions:
            return knobs
        for item in restrictions["at_most_one"]:
            assert item["type"] in ["bool", "int"]
            opts = item["opts"]
            vals = [0] * len(opts)
            true_idx = np.random.randint(0, len(opts) + 1)
            if true_idx < len(vals):
                vals[true_idx] = 1
            for opt, val in zip(opts, vals):
                knobs[opt] = val
        return knobs

    def generate(self, id_str):
        """Generate a test case from the template"""
        content = copy.deepcopy(self.content)
        content["test"] = content["test"].replace("<id>", id_str)
        content["description"] = content["description"].replace("<id>", id_str)
        knobs = {}
        for k, v in content["gen_opts"].items():
            assert "type" in v, f"Missing type for {k}"
            if v["type"] == "int":
                assert (
                    "min_val" in v and "max_val" in v
                ), f"Missing min_val or max_val for {k}"
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

        knobs = self.apply_restrictions(content, knobs)

        content["gen_opts"] = "\n".join(f"+{k}={v}" for k, v in knobs.items())

        return [content]

    @classmethod
    def from_yaml(cls, yaml_path):
        with open(yaml_path, "r") as f:
            content = yaml.load(f, Loader=yaml.FullLoader)
        if content["base_template"] == "self":
            return None  # Do not create from a base template
        yaml_dir = os.path.dirname(yaml_path)
        base_template_path = os.path.join(yaml_dir, content["base_template"])
        with open(base_template_path, "r") as f:
            base_content = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in content.items():
            base_content[k] = v
        return cls(base_content)

    def dump_to_yaml(self, yaml_path, id_str="<undefined>"):
        generated = self.generate(id_str)
        assert len(generated) == 1
        generated = generated[0]
        gen_opts = generated["gen_opts"]
        gen_opts_str = [" " * 4 + l for l in gen_opts.split("\n")]
        gen_opts_str = "\n".join(["  gen_opts: >"] + gen_opts_str)

        del generated["gen_opts"]
        with open(yaml_path, "w") as f:
            yaml.dump([generated], f)
        with open(yaml_path, "a") as f:
            f.write(gen_opts_str)


def generate_tests(template_dir: str, output_dir: str, num_tests: int):
    """Generate tests given test template"""
    # Load test templates
    templates = []
    for yaml_path in glob(os.path.join(template_dir, "*.yaml")):
        template = DvTestTemplate.from_yaml(yaml_path)
        if template is not None:  # Base template is not converted to a template
            templates.append(template)

    os.makedirs(output_dir, exist_ok=True)

    # Generate tests
    for i in range(num_tests):
        # Sample a template from loaded templates
        template = np.random.choice(templates)
        id_str = f"{i:05d}"
        template.dump_to_yaml(os.path.join(output_dir, f"{id_str}.yaml"), id_str)

    print(f"Generated {num_tests} test cases in '{output_dir}'.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-td",
        "--template_dir",
        required=True,
        help="Directory containing the test template files.",
    )
    parser.add_argument(
        "-od",
        "--output_dir",
        default="generated/tests",
        help=(
            "Path to the directory where the generatd test "
            "YAML files will be written"
        ),
    )
    parser.add_argument(
        "-nt", "--num_tests", type=int, default=10, help="Number of tests to generate"
    )
    args = parser.parse_args()
    generate_tests(args.template_dir, args.output_dir, args.num_tests)


if __name__ == "__main__":
    main()
