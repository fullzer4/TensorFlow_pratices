import tensorflow as tf

v = tf.Variable(0)

def increment_by_one(v):
        v = tf.add(v,1)
        return v
    
for i in range(3):
    v = increment_by_one(v)
    print(v)