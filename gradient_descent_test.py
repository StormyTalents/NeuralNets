import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Placeholders for training data
x = tf.placeholder("float")
y = tf.placeholder("float")

# Weights
w = tf.Variable([0.0] * 2, name="w")
# model of y = a*x + b
y_model = tf.mul(x, w[0]) + w[1]

# Loss is squared of the distances
loss = tf.square(y - y_model)

# Gradient descent optimizer
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

###############################################################################

model = tf.global_variables_initializer()

use_GPU = False
config = tf.ConfigProto(device_count = {'GPU': 1 if use_GPU == True else 0})

# Linear equation to learn
def linear_function(x):
    return -1/2.0 * x + 6

# Settings
show_display = True
decay_rate = 0.30
loss_threshold = 1e-8

with tf.Session(config=config) as session:
    session.run(model)
    
    if show_display:
        fig = plt.figure()
        
    done = False
    iteration = 0
    sustained_loss = 0.0
    loss_values = []
    
    while not done:
        # Creates training data and desired output according to w = [2, 6]
        x_val = np.random.rand()
        y_val = linear_function(x_val)
        
        # Trains on the data
        _, loss_val = session.run([train_op, loss], feed_dict={x: x_val, y: y_val})
        sustained_loss = decay_rate * sustained_loss + (1.0 - decay_rate) * loss_val
        
        # Threshold check
        if sustained_loss < loss_threshold:
            done = True
        
        loss_values.append(loss_val)
        iteration += 1
        
        # Display lines
        if iteration % 10 == 0 and show_display:
            w_val = session.run(w)
            fig.canvas.set_window_title('Iteration {}'.format(iteration))
            fig.clear()
            plt.plot([0, 10], [linear_function(0), linear_function(10)], color='green', linewidth=2)
            plt.plot([0, 10], [w_val[0] * 0 + w_val[1], w_val[0] * 10 + w_val[1]], color='red', linewidth=2)
            plt.pause(0.001)
    
    # Display results
    w_value = session.run(w)
    print("Iteration count: {}".format(iteration))
    print("Predicted model: {a:.3f}x + {b:.3f}".format(a=w_value[0], b=w_value[1]))
    
    plt.figure('Loss')
    plt.plot(loss_values)
    plt.show()

