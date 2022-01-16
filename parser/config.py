import os

BASE_DIR = os.path.abspath(
    os.path.dirname(
        os.path.dirname(__file__)))

PARSER_DIR = os.path.join(BASE_DIR, "parser")
IBEX_DIR = os.path.join(BASE_DIR, "ibex")

RTL_DIR = os.path.join(IBEX_DIR, "rtl")
LARK_RULES = os.path.join(PARSER_DIR, "lark_rules.lark")