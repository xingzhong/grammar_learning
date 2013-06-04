# the model is determined by rules, emission and init.
class State(object):
  def __init__(self, args):
    self._symbol = args[0]
    self._left = args[1]
    self._right = args[2]

  def symbol(self):
    return self._symbol

  def left(self):
    return self._left

  def right(self):
    return self._right

