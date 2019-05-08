import tensorflow as tf

double_module = tf.load_op_library('./double.so')

class TestDoubleOp(tf.test.TestCase):
  def test_double(self):
    arr = [1, 2, 3, 4, 5]
    doubled_arr = [2, 4, 6, 8, 10]

    with self.session():
      result = double_module.double(arr)
      self.assertAllEqual(result.eval(), doubled_arr)

if __name__ == '__main__':
  tf.test.main()
