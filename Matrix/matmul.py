import tensorflow as tf

matriz1 = tf.constant([[3,3,3],[3,3,3],[3,3,3]])
matriz2 = tf.constant([[2,2,2],[2,2,2],[2,2,2]])

@tf.function
def mathmul():
  return tf.matmul(matriz1, matriz2)
"""
Retorno:

tf.Tensor(
[[18 18 18]
 [18 18 18]
 [18 18 18]], shape=(3, 3), dtype=int32)

"""

print(mathmul())