from common.numpy_utils import (
  filter_strictly_increasing,
  get_max_increasing_interval
)

def test_get_max_increasing_interval():
  arr = [1, 2, 4, 2, 4, 1, 6, 7, 8, 9, 10, 12, 18, 1, 4, 2, 6, 7, 8]
  i1, i2 = get_max_increasing_interval(arr)
  assert i1 == 5
  assert i2 == 13

  arr = [5, 1]
  i1, i2 = get_max_increasing_interval(arr)
  assert i1 == 0
  assert i2 == 1

  arr = [0]
  i1, i2 = get_max_increasing_interval(arr)
  assert i1 == 0
  assert i2 == 1
