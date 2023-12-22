import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mnist import MNIST
import math
import time
from display_utils import DynamicConsoleTable
from sound_utils import Sounds

###############################################################################

def model(learning_rate=0.01):
    # Parameters
    weights = {
        'conv1': tf.Variable(tf.random_normal([5, 5, 1, 20], stddev=np.sqrt(2./(5*5*1)))),
        'out': tf.Variable(tf.random_normal([28*28*20, 10], stddev=np.sqrt(2./(28*28*20)))),
    }
    biases = {
        'conv1': tf.Variable(tf.zeros(20)),
        'out': tf.Variable(tf.zeros(10)),
    }

    # Placeholders for training data
    x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    y = tf.placeholder(tf.int64, [None])

    # Input -> Conv + ReLU
    conv1 = tf.nn.conv2d(x, weights['conv1'], strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.relu(tf.nn.bias_add(conv1, biases['conv1']))
    lrn1 = tf.nn.local_response_normalization(conv1, depth_radius=5, bias=1.0, alpha=1e-4, beta=0.75)
    
    # FC -> Output FC
    out = tf.reshape(lrn1, [-1, weights['out'].get_shape().as_list()[0]])
    out = tf.add(tf.matmul(out, weights['out']), biases['out'])

    # Loss and optimizer
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(out, y))
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    
    # Accuracy
    accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(out, y, 1), tf.float32))
    
    return train_op, x, y, out, loss, accuracy, weights, biases

###############################################################################
### Settings

# Training settings
# Note: Training terminates when the sustained loss is below loss_threshold, or when training has reached max_epochs
max_epochs = 1000
batch_size = 100
validation_set_size = 1000
learning_rate = 0.01
loss_threshold = 0 #1e-12
decay_rate = 0.30 # Exponential decay used to calculate sustained loss
use_GPU = True # Use CUDA acceleration

# Weights display settings
show_weights = False
weights_display_step = 50 # in batches
interpolation = None # None to use default (eg. "nearest", "bilinear")
cmap = None # None to use default (eg. "gray", "inferno")
weights_delay = 0.001
weights_title = 'Weights'

# Loss display settings
show_loss = False
loss_display_step = 50
loss_saved_iterations = 500
loss_starting_iteration = 20
loss_delay = 0.001
loss_title = 'Loss'

# Accuracy display settings
show_accuracy = False
accuracy_display_step = 50
accuracy_saved_iterations = float('inf')
accuracy_starting_iteration = 0
accuracy_delay = 0.001
accuracy_title = 'Accuracy'

# Console output settings
progress_bar_size = 20

# Sound settings
use_sounds = True

##############################################################################
# Data loader and setup
print 'Loading images..'
mndata = MNIST('./mnist')
training_images, training_labels = mndata.load_training()
validation_images, validation_labels = mndata.load_testing()
print 'Training images: {}'.format(len(training_images))
print 'Validation images: {}'.format(len(validation_images))

print 'Reshaping images..'
for i in range(len(training_images)):
    training_images[i] = np.expand_dims(np.reshape(training_images[i], (28, 28)), axis=2)
for i in range(len(validation_images)):
    validation_images[i] = np.expand_dims(np.reshape(validation_images[i], (28, 28)), axis=2)
    
print 'Creating batches..'
assert validation_set_size <= len(validation_images), 'validation_set_size must be smaller than len(validation_images)'
assert float(len(training_images)) / batch_size % 1 == 0, 'batch_size must evenly divide len(training_images)'
assert float(validation_set_size) / batch_size % 1 == 0, 'batch_size must evenly divide validation_set_size'
num_training_batches = len(training_images) / batch_size
num_validation_batches = validation_set_size / batch_size

training_image_batches = []
training_label_batches = []
validation_image_batches = []
validation_label_batches = []
for i in range(num_training_batches):
    training_image_batches.append(training_images[i*batch_size:(i+1)*batch_size])
    training_label_batches.append(training_labels[i*batch_size:(i+1)*batch_size])
for i in range(num_validation_batches):
    validation_image_batches.append(validation_images[i*batch_size:(i+1)*batch_size])
    validation_label_batches.append(validation_labels[i*batch_size:(i+1)*batch_size])
    
print 'Done'
###############################################################################

###############################################################################
# Pyplot setup
plt.ion() # Enable interactive mode
###############################################################################

###############################################################################
# Progress display setup
weights_fig = None
if show_weights:
    weights_fig = plt.figure(weights_title)
    weights_fig.show()
def display_weights(weights_val):
    weights_fig = plt.figure(weights_title)
    weights_fig.clear()
    plot_height = int(weights_val['conv1'].shape[3] ** 0.5)
    plot_width = math.ceil(float(weights_val['conv1'].shape[3]) / plot_height)
    for j in range(weights_val['conv1'].shape[3]):
        ax = weights_fig.add_subplot(plot_height, plot_width, j + 1)
        ax.imshow(weights_val['conv1'][:,:,0,j], interpolation=interpolation, cmap=cmap)
    plt.pause(weights_delay)
###############################################################################

###############################################################################
# Loss display setup
loss_fig = None
if show_loss:
    loss_fig = plt.figure(loss_title)
    loss_fig.show()
def display_loss(loss_values, iteration):
    loss_fig = plt.figure(loss_title)
    loss_fig.clear()
    plt.plot(list(range(iteration - len(loss_values) + 1, iteration + 1)), loss_values)
    plt.pause(loss_delay)
###############################################################################

###############################################################################
# Accuracy display setup
accuracy_fig = None
if show_loss:
    accuracy_fig = plt.figure(accuracy_title)
    accuracy_fig.show()
def display_accuracy(validation_accuracy_values, max_accuracy_values, iteration):
    accuracy_fig = plt.figure(accuracy_title)
    accuracy_fig.clear()
    plt.plot(list(range(iteration - len(validation_accuracy_values) + 1, iteration + 1)), validation_accuracy_values)
    #plt.plot(list(range(iteration - len(max_accuracy_values) + 1, iteration + 1)), max_accuracy_values)
    #plt.plot([iteration - len(max_accuracy_values) + 1, iteration], [max_accuracy_values[-1]] * 2)
    plt.pause(accuracy_delay)
###############################################################################

###############################################################################
# Sound setup
sounds = Sounds()
if use_sounds:
    sounds.open()
###############################################################################

###############################################################################
# Output control
def update_output(iteration, weights_val, loss_values, validation_accuracy_values, max_accuracy_values, override=False):
    
    # Show/update weights display
    if iteration % weights_display_step == 0 and show_weights or override:
        display_weights(weights_val)
        if use_sounds:
            sounds.alert()

    # Show/update loss display
    if iteration % loss_display_step == 0 and show_loss or override:
        display_loss(loss_values, iteration)

    # Show/update accuracy display
    if iteration % accuracy_display_step == 0 and show_accuracy or override:
        display_accuracy(validation_accuracy_values, max_accuracy_values, iteration)
        
###############################################################################

# Build model and get variable handles
train_op, x, y, out, loss, accuracy, weights, biases = model(learning_rate)

# Initialize environment
initialize = tf.global_variables_initializer()

# Session config
config = tf.ConfigProto(device_count = {'GPU': 1 if use_GPU == True else 0})

# Run model
done = False
epoch = 0
iteration = 0
sustained_loss = 0.0
loss_values = []
validation_accuracy_values = []
max_accuracy_values = []

max_accuracy = 0.0
max_accuracy_weights = None
max_accuracy_biases = None
    
with tf.Session(config=config) as session:
    session.run(initialize)
    
    print '=========='
    print 'GPU ' + ('enabled' if use_GPU else 'disabled')
    print
    
    # Show weight initialization
    if show_weights:
        weights_val = session.run(weights)
        display_weights(weights_val)
    
    layout = [
        dict(name='Ep.', width=3, align='center'),
        dict(name='Batch', width=2*len(str(num_training_batches))+1, suffix='/'+str(num_training_batches)),
        dict(name='Loss', width=8),
        dict(name='Val Acc', width=6, suffix='%'),
        dict(name='Max Acc', width=6, suffix='%'),
        dict(name='Time', width=progress_bar_size+2, align='center'),
    ]
    table = DynamicConsoleTable(layout)
    table.print_header()
    
    while not done:
        epoch += 1
        
        if use_sounds:
            sounds.alert()

        # Trains on the data, in batches
        for i in range(num_training_batches):
            iteration += 1
                        
            images_batch = training_image_batches[i]
            labels_batch = training_label_batches[i]
            _, loss_val = session.run([train_op, loss], feed_dict={x: images_batch, y: labels_batch})
            sustained_loss = decay_rate * sustained_loss + (1.0 - decay_rate) * loss_val
            
            if len(loss_values) == loss_saved_iterations:
                loss_values.pop(0)
            if iteration >= loss_starting_iteration:
                loss_values.append(loss_val)
            
            images_batch = validation_image_batches[iteration % num_validation_batches]
            labels_batch = validation_label_batches[iteration % num_validation_batches]
            
            validation_accuracy = 0.0
            for j in range(num_validation_batches):
                images_batch = validation_image_batches[j]
                labels_batch = validation_label_batches[j]
                accuracy_val = session.run(accuracy, feed_dict={x: images_batch, y: labels_batch})
                validation_accuracy += accuracy_val
            validation_accuracy /= num_validation_batches
            
            if len(validation_accuracy_values) == accuracy_saved_iterations:
                validation_accuracy_values.pop(0)
            if iteration >= accuracy_starting_iteration:
                validation_accuracy_values.append(validation_accuracy)
            
            if validation_accuracy > max_accuracy:
                weights_val, biases_val = session.run([weights, biases])
                max_accuracy = validation_accuracy
                max_accuracy_weights = weights_val
                max_accuracy_biases = biases_val
                if use_sounds:
                    sounds.success()
                    
            if len(max_accuracy_values) == accuracy_saved_iterations:
                max_accuracy_values.pop(0)
            if iteration >= accuracy_starting_iteration:
                max_accuracy_values.append(max_accuracy)
            
            progress = int(math.ceil(progress_bar_size * float((iteration - 1) % num_training_batches) / (num_training_batches - 1)))
            progress_string = '[' + '#' * progress + ' ' * (progress_bar_size - progress) + ']'
            if iteration % num_training_batches == 0:
                progress_string = time.strftime("%I:%M:%S %p", time.localtime())
            table.update(epoch,
                         (iteration - 1) % num_training_batches + 1,
                         sustained_loss,
                         validation_accuracy * 100,
                         max_accuracy * 100,
                         progress_string)
            
            # Termination condition
            if sustained_loss < loss_threshold:
                done = True
                break

            update_output(iteration, weights_val, loss_values, validation_accuracy_values, max_accuracy_values)
        
        table.finalize()
            
        # Termination condition
        if epoch >= max_epochs or sustained_loss < loss_threshold:
            done = True
            update_output(iteration, weights_val, loss_values, validation_accuracy_values, max_accuracy_values, override=True)
            plt.pause(0)

