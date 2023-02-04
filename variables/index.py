import tensorflow as tf

v = tf.Variable(0)

def soma(valor):
        valor = tf.add(valor,1)
        return valor
    
for i in range(3):
    v = soma(v)
    print(v)
    
"""
Retorno:

tf.Tensor(1, shape=(), dtype=int32)
tf.Tensor(2, shape=(), dtype=int32)
tf.Tensor(3, shape=(), dtype=int32)

"""