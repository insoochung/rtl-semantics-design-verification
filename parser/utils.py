def get_log_fn(verbose=True):
  """Return a function that prints to stdout"""
  def log_fn(str=""):
    pass
  if verbose:
    log_fn = print

  return log_fn

def get_indent_str(indent):
  return " " * indent