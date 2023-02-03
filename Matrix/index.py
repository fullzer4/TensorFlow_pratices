import tensorflow as tf

print(tf.__version__)

Matrix_one = tf.constant([[1,2,3],[2,3,4],[3,4,5]])
Matrix_two = tf.constant([[2,2,2],[2,2,2],[2,2,2]])

def add():
    add_1_operation = tf.add(Matrix_one, Matrix_two)
    return add_1_operation

def mathmul():
  return tf.matmul(Matrix_one, Matrix_two)

print ("Defined using tensorflow function :")
add_1_operation = add()
print(add_1_operation)
print ("Defined using normal expressions :")
add_2_operation = Matrix_one + Matrix_two
print(add_2_operation)

mul_operation = mathmul()

print ("Defined using tensorflow function :")
print(mul_operation)