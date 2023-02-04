import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("FuelConsumption.csv")
train_x = df["ENGINESIZE"] # variavel idenpendente
train_y = df["CO2EMISSIONS"] # variavel dependente

print(train_x.dtype)
print(train_y.dtype)

a = tf.Variable(20.0)
b = tf.Variable(30.2)

def funcaoLinear(x):
   y = a*x + b
   return y

def perda(y,train_y) :
    return tf.reduce_mean(tf.square(y - train_y))

otimizador = tf.optimizers.SGD(learning_rate=0.01)

for i in range(1000): # 1000 epochs
    with tf.GradientTape() as tape:
        prever = funcaoLinear(train_x)
        vperda = perda(prever, train_y)
        gradients = tape.gradient(vperda, [a,b])
        otimizador.apply_gradients(zip(gradients, [a,b]))
    if i % 100 == 0:
        print("Epoch {}, a: {}, b: {}".format(i, a.numpy(), b.numpy()))
        
print("Resultado final dos valores a: {} e b: {}".format(a.numpy(), b.numpy()))

plt.scatter(train_x, train_y)
plt.plot(train_x, funcaoLinear(train_x).numpy(), c='r')
plt.show()