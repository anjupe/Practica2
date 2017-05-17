import tensorflow as tf
import numpy as np


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


data = np.genfromtxt('iris.data', delimiter=",")  # iris.data file loading
np.random.shuffle(data)  # we shuffle the data

n_train = len(data)*70/100
n_vali = len(data)*15/100
#Dividir conjunto de datos.
x_data = data[:n_train, 0:4].astype('f4')  # the samples are the four first rows of data
y_data = one_hot(data[:n_train, 4].astype(int), 3)  # the labels are in the last row. Then we encode them in one hot code

x_vali = data[n_train:n_train+n_vali, 0:4].astype('f4') #Conjunto de validacion
y_vali = one_hot(data[n_train:n_train+n_vali, 4].astype(int), 3)

x_test = data[n_train+n_vali:, 0:4].astype('f4')#Conjunto de test.
y_test = one_hot(data[n_train+n_vali:, 4].astype(int), 3)

print "\nSome samples..."
for i in range(20):
    print x_data[i], " -> ", y_data[i]
print

x = tf.placeholder("float", [None, 4])  # samples
y_ = tf.placeholder("float", [None, 3])  # labels

W1 = tf.Variable(np.float32(np.random.rand(4, 5)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(5, 3)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(3)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

#init = tf.initialize_all_variables() #deprecated
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

print "----------------------"
print "   Start training...  "
print "----------------------"

batch_size = 21 #para que se usen todos los datos.

epoch = 0
e_ant = 200.

while True:
#for epoch in xrange(300):
    for jj in xrange(len(x_data) / batch_size):
        batch_xs = x_data[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_data[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    e_actual = sess.run(loss, feed_dict={x: x_vali, y_: y_vali}) #Se hace la validacion.
    epoch += 1
    if e_ant <= e_actual and e_actual < 7: #Se ha llegado a un minimo
        break
    else:
        e_ant = e_actual
print "---------------------------"
print "   Fin del Entrenamiento   "
print "---------------------------"
print "Se han realizado ", epoch, " epocas. \nEl ultimo error de validacion fue: ", e_actual, "\n"
print "*** Realizacion del test. ***"
aciertos = 0
result = sess.run(y, feed_dict={x: x_test})
for b, r in zip(y_test, result):
    if (np.argmax(b) == np.argmax(r)):
        aciertos += 1
    print b, "-->", r
print "----------------------------------------------------------------------------------"
Porcentaje = float(100*aciertos)/len(y_test)
print"\nEl numero de aciertos del conjunto de test fue: ", aciertos, "\nPorcentaje: ", Porcentaje,"%"
