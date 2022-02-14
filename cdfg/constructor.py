from graph import RtlFile
from parser import get_verible_parsed_rtl

# def get_module_graphs(filepath, verible_tree):
#   module_graphs = []


def get_rtl_file_obj(filepath, verible_tree):
  return RtlFile(filepath, verible_tree)


if __name__ == "__main__":
  parsed_rtl = get_verible_parsed_rtl()
  for filepath, verible_tree in parsed_rtl.items():
    print(filepath)
    get_rtl_file_obj(verible_tree, filepath)
