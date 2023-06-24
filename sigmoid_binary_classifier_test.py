import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

###############################################################################

def model(learning_rate=0.01):
    # Parameters
    weights = {
        'fc1': tf.Variable(tf.random_normal([2, 4], stddev=np.sqrt(2./(2)))),
        'out': tf.Variable(tf.random_normal([4, 1], stddev=np.sqrt(2./(4)))),
    }
    biases = {
        'fc1': tf.Variable(tf.zeros(4)),
        'out': tf.Variable(tf.zeros(1)),
    }

    # Placeholders for training data
    x = tf.placeholder(tf.float32, [None, None])
    y = tf.placeholder(tf.int64, [None])

    # Input -> FC + sigmoid
    fc1 = tf.add(tf.matmul(x, weights['fc1']), biases['fc1'])
    fc1 = tf.nn.sigmoid(fc1)
    
    # FC -> Output FC
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    # Loss and optimizer
    loss = tf.reduce_mean(tf.square(tf.maximum(0.0, tf.transpose(-out) * tf.to_float(y))))
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    
    return train_op, x, y, out, loss, weights, biases

###############################################################################
### Settings

# Training settings
# Note: Training terminates when the sustained loss is below loss_threshold, or when training has reached max_epochs
max_epochs = 50000
learning_rate = 0.05
loss_threshold = 1e-12
decay_rate = 0.30 # Exponential decay used to calculate sustained loss
use_GPU = False # Use CUDA acceleration

# Display settings
show_progress = True
display_step = 200
delay = 0.001
interpolation = None # None to use default (eg. "nearest", "bilinear")
resolution = 15
margin = 0.5
boundary_blur_size = 0.5

#point_label_map = {}
#n = 20
#x1 = np.random.normal(2, 2, n)
#y1 = np.random.normal(7, 1, n)
#x2 = np.random.normal(8, 2, n)
#y2 = np.random.normal(4, 1, n)
#for i in range(n):
#    point_label_map[(x1[i], y1[i])] = 1
#    point_label_map[(x2[i], y2[i])] = -1

point_label_map = {
    (2, 1): -1,
    (3, 2): -1,
    (4, 1): -1,
    (4, 2): -1,
    (4, 3): -1,
    (5, 4): -1,
    (6, 2): -1,
    
    (1, 2): 1,
    (1, 4): 1,
    (2, 3): 1,
    (2, 5): 1,
    (3, 4): 1,
    (4, 6): 1,
    (5, 5): 1,
}

#point_label_map = {
#    (1, 6): -1,
#    (1, 5): -1,
#    (1, 4): -1,
#    (1, 3): -1,
#    (2, 2): -1,
#    (3, 1): -1,
#    (4, 1): -1,
#    (5, 1): -1,
#    (6, 1): -1,
#    (7, 2): -1,
#    (8, 3): -1,
#    (8, 4): -1,
#    (8, 5): -1,
#    (8, 6): -1,
#    (7, 7): -1,
#    (6, 8): -1,
#    (5, 8): -1,
#    (4, 8): -1,
#    (3, 8): -1,
#    (2, 7): -1,
#    
#    (4, 4): 1,
#    (4, 5): 1,
#    (5, 4): 1,
#    (5, 5): 1,
#}

###############################################################################
### Display setup
x_values = [point[0] for point in point_label_map]
y_values = [point[1] for point in point_label_map]
x_min = min(x_values)
x_max = max(x_values)
y_min = min(y_values)
y_max = max(y_values)
x_range = x_max - x_min
y_range = y_max - y_min
x_left = x_min - int(x_range * margin)
x_right = x_max + int(x_range * margin)
y_bottom = y_min - int(y_range * margin)
y_top = y_max + int(y_range * margin)

def transform_x(x):
    return (x - x_left) * resolution
def transform_y(y):
    return (y - y_bottom) * resolution
def untransform_x(x):
    return float(x) / resolution + x_left
def untransform_y(y):
    return float(y) / resolution + y_bottom

x_limits = [transform_x(x_left), transform_x(x_right)]
y_limits = [transform_y(y_bottom), transform_y(y_top)]
hm_width = int(x_right - x_left + 1) * resolution
hm_height = int(y_top - y_bottom + 1) * resolution

fig = plt.figure()
test_points = [[untransform_x(i), untransform_y(j)] for j in range(hm_height) for i in range(hm_width)]

heatmap = np.zeros((hm_height, hm_width))
def formatter_x(x, p):
    return "{}".format(int(x / resolution + x_left)) if x / resolution % 1 == 0 else ""
def formatter_y(y, p):
    return "{}".format(int(y / resolution + y_bottom)) if y / resolution % 1 == 0 else ""
def format_display():
    fig.canvas.set_window_title('Epoch {}'.format(epoch))
    fig.clear()
    plt.xlim(x_limits)
    plt.ylim(y_limits)
    axes = plt.gca()
    axes.get_xaxis().set_major_formatter(ticker.FuncFormatter(formatter_x))
    axes.get_yaxis().set_major_formatter(ticker.FuncFormatter(formatter_y))
    
def display(session, loss_val, points, labels, out_val, done):
    # Format display
    format_display()
    
    # Draw heatmap
    global heatmap
    test_result = session.run([out], feed_dict={x: test_points})
    test_result[0] /= boundary_blur_size * (np.max(test_result[0]) - np.min(test_result[0]))
    
    for i in range(hm_height):
        for j in range(hm_width):
            n = i * hm_width + j
            heatmap[i][j] = 1 / (1 + np.exp(test_result[0][n] * 100)) * 2 - 1
    hm = plt.imshow(heatmap, interpolation=interpolation, origin='lower')

    # Draw points
    for i in range(len(points)):
        correct = np.sign(out_val[i][0]) == labels[i] or np.sign(out_val[i][0]) == 0 and labels[i] == 1
        color = ('#4560ff' if labels[i] == 1 else '#ff534a')
        plt.scatter(transform_x(points[i][0]), transform_y(points[i][1]), color=color, s=60, edgecolors=('black' if correct else 'white'), linewidth=2)
        
    # Update text
    plt.title("Loss: {0:.2E}".format(loss_val))

    # Delay
    plt.pause(delay)
###############################################################################

# Build model and get variable handles
train_op, x, y, out, loss, weights, biases = model(learning_rate)

# Initialize environment
initialize = tf.global_variables_initializer()

# Session config
config = tf.ConfigProto(device_count = {'GPU': 1 if use_GPU == True else 0})

# Run model
with tf.Session(config=config) as session:
    session.run(initialize)
    
    # Get training data
    points = point_label_map.keys()
    labels = point_label_map.values()
        
    done = False
    epoch = 0
    sustained_loss = 0.0
    loss_values = []
    while not done:

        # Trains on the data
        _, loss_val, out_val = session.run([train_op, loss, out], feed_dict={x: points, y: labels})
        sustained_loss = decay_rate * sustained_loss + (1.0 - decay_rate) * loss_val
        loss_values.append(loss_val)
            
        epoch += 1
        print "Epoch {}".format(epoch)
        
        # Termination condition
        if epoch >= max_epochs or sustained_loss < loss_threshold:
            done = True
        
        # Show/update display
        if epoch % display_step == 0 and show_progress or done:
            display(session, loss_val, points, labels, out_val, done)

# Display results
print("Epoch count: {}".format(epoch))    
plt.show()

#plt.figure('Loss')
#plt.plot(loss_values)
#plt.show()

