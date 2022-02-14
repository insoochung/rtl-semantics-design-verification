import os

BASE_DIR = os.path.abspath(
    os.path.dirname(
        os.path.dirname(__file__)))

PARSER_DIR = os.path.join(BASE_DIR, "cdfg")
IBEX_DIR = os.path.join(BASE_DIR, "ibex")
IBEX_RTL_DIR = os.path.join(IBEX_DIR, "rtl")

REFORMATTED_DIR = os.path.join(BASE_DIR, "reformatted")
PARSED_RTL_DIR = os.path.join(PARSER_DIR, "parsed_rtl")
