import tensorflow as tf

matriz1 = tf.constant([[3,3,3],[3,3,3],[3,3,3]])
matriz2 = tf.constant([[2,2,2],[2,2,2],[2,2,2]])

@tf.function()
def somaTF():
  soma = tf.add(matriz1, matriz2)
  return soma
"""
Retorno:

tf.Tensor(
[[5 5 5]
 [5 5 5]
 [5 5 5]], shape=(3, 3), dtype=int32)
 
"""

def soma():
  soma = matriz1 + matriz2
  return soma
"""
Retorno:

tf.Tensor(
[[5 5 5]
 [5 5 5]
 [5 5 5]], shape=(3, 3), dtype=int32)
 
"""