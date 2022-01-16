import os

BASE_DIR = os.path.abspath(
  os.path.dirname(
    os.path.dirname(__file__)))
IBEX_DIR = os.path.join(BASE_DIR, "ibex")
RTL_DIR = os.path.join(IBEX_DIR, "rtl")
LARK_RULES = os.path.join(BASE_DIR, "lark_rules.lark")