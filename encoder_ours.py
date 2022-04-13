import tensorflow.compat.v1 as tf
from tensorflow.python import pywrap_tensorflow


WEIGHT_INIT_STDDEV = 0.1
DENSE_layers = 3
DECAY = .9
EPSILON = 1e-4

class Encoder(object):            
    def __init__(self, model_pre_path):
        self.weight_vars = []
        self.model_pre_path = model_pre_path

        with tf.variable_scope('encoder'):
            self.weight_vars.append(self._create_variables(1, 16, 3, scope='conv1_1'))
            self.weight_vars.append(self._create_variables(16, 16, 5, scope='conv1_2'))
            self.weight_vars.append(self._create_variables(16, 16, 5, scope='conv1_3'))
            self.weight_vars.append(self._create_variables(16, 16, 3, scope='conv1_4'))
            self.weight_vars.append(self._create_variables(16, 16, 5, scope='conv1_5'))
            self.weight_vars.append(self._create_variables(16, 16, 5, scope='conv1_6'))
            #self.weight_vars.append(self._create_variables(16, 16, 5, scope='conv1_7'))

            self.weight_vars.append(self._create_variables(64, 16, 3, scope='conv1_7'))
            self.weight_vars.append(self._create_variables(16, 16, 5, scope='conv1_8'))
            self.weight_vars.append(self._create_variables(16, 16, 5, scope='conv1_9'))
            self.weight_vars.append(self._create_variables(16, 16, 3, scope='conv1_10'))
            self.weight_vars.append(self._create_variables(16, 16, 5, scope='conv1_11'))
            self.weight_vars.append(self._create_variables(16, 16, 5, scope='conv1_12'))
            #self.weight_vars.append(self._create_variables(16, 16, 5, scope='conv1_14'))

            self.weight_vars.append(self._create_variables(64, 16, 3, scope='conv1_13'))
            self.weight_vars.append(self._create_variables(16, 16, 5, scope='conv1_14'))
            self.weight_vars.append(self._create_variables(16, 16, 5, scope='conv1_15'))
            self.weight_vars.append(self._create_variables(16, 16, 3, scope='conv1_16'))
            self.weight_vars.append(self._create_variables(16, 16, 5, scope='conv1_17'))
            self.weight_vars.append(self._create_variables(16, 16, 5, scope='conv1_18'))
            #self.weight_vars.append(self._create_variables(16, 16, 5, scope='conv1_21'))


            #self.weight_vars.append(self._create_variables(16, 64, 3, scope='dense_block_conv1'))
            #self.weight_vars.append(self._create_variables(80, 16, 3, scope='conv1_2'))
            #self.weight_vars.append(self._create_variables(16, 64, 3, scope='dense_block_conv2'))
            #self.weight_vars.append(self._create_variables(80, 16, 3, scope='conv1_3'))
            #self.weight_vars.append(self._create_variables(16, 64, 3, scope='dense_block_conv3'))
             
            #self.weight_vars.append(self._create_variables(64, 64, 3, scope='conv1_2'))

    def _create_variables(self, input_filters, output_filters, kernel_size, scope):
        shape = [kernel_size, kernel_size, input_filters, output_filters]
        if self.model_pre_path:
            reader = pywrap_tensorflow.NewCheckpointReader(self.model_pre_path)
            with tf.variable_scope(scope):
                kernel = tf.Variable(reader.get_tensor('encoder/' + scope + '/kernel'), name='kernel')
                bias = tf.Variable(reader.get_tensor('encoder/' + scope + '/bias'), name='bias')
        else:
            with tf.variable_scope(scope):
                kernel = tf.Variable(tf.truncated_normal(shape, stddev=WEIGHT_INIT_STDDEV), name='kernel')
                bias = tf.Variable(tf.zeros([16]), name='bias')
        return (kernel, bias)

    def encode(self, image):
        #dense_indices = (1, 3, 5)
        #final_layer_idx = len(self.weight_vars) - 1

        out = image
#1        
        kernel0, bias0 = self.weight_vars[0]
        out = conv2d(out, kernel0, bias0, use_relu=True)
        print(out.shape)
        kernel1, bias1 = self.weight_vars[1]
        out1 = conv2d(out, kernel1, bias1, use_relu=True)
        print(out1.shape)
        kernel2, bias2 = self.weight_vars[2]
        out2 = conv2d_SAME(out1, kernel2, bias2, use_relu=True)
        print(out2.shape)
        
        kernel3, bias3 = self.weight_vars[3]
        out3 = conv2d(out, kernel3, bias3, use_relu=True)
        print(out3.shape) 
        #kernel4, bias4 = self.weight_vars[4]
        #out4 = conv2d(out3, kernel4, bias4, use_relu=True)
        #print(out4.shape) 
        
        
        kernel5, bias5 = self.weight_vars[4]
        out5 = conv2d(out, kernel5, bias5, use_relu=True)
        print(out5.shape)
        kernel6, bias6 = self.weight_vars[5]
        out6 = conv2d_SAME(out5, kernel6, bias6, use_relu=True)
        print(out6.shape)
        out = tf.concat([out2, out3, out6, out], 3)
        #return out
#2  
        kernel0, bias0 = self.weight_vars[6]
        out = conv2d(out, kernel0, bias0, use_relu=True)
        print(out.shape)
        kernel1, bias1 = self.weight_vars[7]
        out1 = conv2d(out, kernel1, bias1, use_relu=True)
        print(out1.shape)
        kernel2, bias2 = self.weight_vars[8]
        out2 = conv2d_SAME(out1, kernel2, bias2, use_relu=True)
        print(out2.shape)
        
        kernel3, bias3 = self.weight_vars[9]
        out3 = conv2d(out, kernel3, bias3, use_relu=True)
        print(out3.shape) 
        
        #kernel4, bias4 = self.weight_vars[11]
        #out4 = conv2d(out3, kernel4, bias4, use_relu=True)
        #print(out4.shape)
        kernel5, bias5 = self.weight_vars[10]
        out5 = conv2d(out, kernel5, bias5, use_relu=True)
        print(out5.shape)
        kernel6, bias6 = self.weight_vars[11]
        out6 = conv2d_SAME(out5, kernel6, bias6, use_relu=True)
        print(out6.shape)
        out = tf.concat([out2, out3, out6, out], 3)
        #return out
#3      
        kernel0, bias0 = self.weight_vars[12]
        out = conv2d(out, kernel0, bias0, use_relu=True)
        print(out.shape)
        kernel1, bias1 = self.weight_vars[13]
        out1 = conv2d(out, kernel1, bias1, use_relu=True)
        print(out1.shape)
        kernel2, bias2 = self.weight_vars[14]
        out2 = conv2d_SAME(out1, kernel2, bias2, use_relu=True)
        print(out2.shape)
        
        kernel3, bias3 = self.weight_vars[15]
        out3 = conv2d(out, kernel3, bias3, use_relu=True)
        print(out3.shape) 
        
        #kernel4, bias4 = self.weight_vars[18]
        #out4 = conv2d(out3, kernel4, bias4, use_relu=True)
        #print(out4.shape)
        kernel5, bias5 = self.weight_vars[16]
        out5 = conv2d(out, kernel5, bias5, use_relu=True)
        print(out5.shape)
        kernel6, bias6 = self.weight_vars[17]
        out6 = conv2d_SAME(out5, kernel6, bias6, use_relu=True)
        out = tf.concat([out2, out3, out6, out], 3)
        #return out
        return out
'''
        #for i in range(len(self.weight_vars)):
            #kernel, bias = self.weight_vars[i]

            # if i == final_layer_idx:
            #     out = transition_block(out, kernel, bias)
            # el
            #if i in dense_indices:
                #out = conv2d_dense(out, kernel, bias, use_relu=True)
            #else:
                #out = conv2d(out, kernel, bias, use_relu=True)
'''            

        


def conv2d(x, kernel, bias, use_relu=True):
    # padding image with reflection mode
    x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')

    # conv and add bias
    # num_maps = x_padded.shape[3]
    # out = __batch_normalize(x_padded, num_maps)
    # out = tf.nn.relu(out)
    out = tf.nn.conv2d(x_padded, kernel, strides=[1, 1, 1, 1], padding='VALID')
    out = tf.nn.bias_add(out, bias)
    out = tf.nn.relu(out)

    return out

def conv2d_SAME(x, kernel, bias, use_relu=True):
    # padding image with reflection mode
    x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')

    # conv and add bias
    # num_maps = x_padded.shape[3]
    # out = __batch_normalize(x_padded, num_maps)
    # out = tf.nn.relu(out)
    out = tf.nn.conv2d(x_padded, kernel, strides=[1, 1, 1, 1], padding='SAME')
    out = tf.nn.bias_add(out, bias)
    out = tf.nn.relu(out)

    return out

def batch_norm(x):
    batch_mean, batch_var = tf.nn.moments(x, [0])
    return tf.nn.batch_normalization(x, batch_mean, batch_var, offset=None, scale=None, variance_epsilon=EPSILON)
'''
def conv2d_dense(x, kernel, bias, use_relu=True):
    # padding image with reflection mode
    x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')

    # conv and add bias
    # num_maps = x_padded.shape[3]
    # out = __batch_normalize(x_padded, num_maps)
    # out = tf.nn.relu(out) if x<0, out =0; if x>0, out = x
    shape = [1, 1, 16, 16]
    kernel = tf.truncated_normal(shape, stddev=WEIGHT_INIT_STDDEV)
    shape1 = [3, 3, 16, 16]
    kernel1 = tf.truncated_normal(shape1, stddev=WEIGHT_INIT_STDDEV)
    shape2 = [5, 5, 16, 16]
    kernel2 = tf.truncated_normal(shape2, stddev=WEIGHT_INIT_STDDEV)
    shape3 = [7, 7, 16, 16]
    kernel3 = tf.truncated_normal(shape3, stddev=WEIGHT_INIT_STDDEV)


    #bias = tf.Variable(tf.zeros([16]), name='bias')
    out0 = tf.nn.conv2d(x_padded, kernel, strides=[1, 1, 1, 1], padding='VALID')
    out0 = tf.nn.bias_add(out0, bias)
    #out0 = batch_norm(out0)
    out0 = tf.nn.relu(out0)
    #out0 = homomorphicRELU(HE,out0)
    print(out0.shape)
    out0 = tf.nn.conv2d(out0, kernel1, strides=[1, 1, 1, 1], padding='VALID')
    out0 = tf.nn.bias_add(out0, bias)
    #out0 = batch_norm(out0)
    out0 = tf.nn.relu(out0)
    print(out0.shape)
    
    
    out1 = tf.nn.conv2d(x_padded, kernel1, strides=[1, 1, 1, 1], padding='VALID')
    out1 = tf.nn.bias_add(out1, bias)
    #out1 = batch_norm(out1)
    out1 = tf.nn.relu(out0)
    print(out1.shape)
   
    out2 = tf.nn.conv2d(x_padded, kernel, strides=[1, 1, 1, 1], padding='SAME')
    out2 = tf.nn.bias_add(out2, bias)
    #out2 = batch_norm(out2)
    out2 = tf.nn.relu(out0)
    print(out2.shape)
    out2 = tf.nn.conv2d(out2, kernel1, strides=[1, 1, 1, 1], padding='VALID')
    out2 = tf.nn.bias_add(out2, bias)
    #out2 = batch_norm(out2)
    out2 = tf.nn.relu(out0)
    print(out2.shape)
    print(x.shape)
    out = tf.concat([out0, out1, out2, x], 3)
    print(out.shape)
    #out4 = tf.concat([out3, out4], 3)
    
    
    return out

'''
def transition_block(x, kernel, bias):

    num_maps = x.shape[3]
    out = __batch_normalize(x, num_maps)
    out = tf.nn.relu(out)
    out = conv2d(out, kernel, bias, use_relu=False)

    return out


def __batch_normalize(inputs, num_maps, is_training=True):
    # Trainable variables for scaling and offsetting our inputs
    # scale = tf.Variable(tf.ones([num_maps], dtype=tf.float32))
    # offset = tf.Variable(tf.zeros([num_maps], dtype=tf.float32))

    # Mean and variances related to our current batch
    batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])

    # # Create an optimizer to maintain a 'moving average'
    # ema = tf.train.ExponentialMovingAverage(decay=DECAY)
    #
    # def ema_retrieve():
    #     return ema.average(batch_mean), ema.average(batch_var)
    #
    # # If the net is being trained, update the average every training step
    # def ema_update():
    #     ema_apply = ema.apply([batch_mean, batch_var])
    #
    #     # Make sure to compute the new means and variances prior to returning their values
    #     with tf.control_dependencies([ema_apply]):
    #         return tf.identity(batch_mean), tf.identity(batch_var)
    #
    # # Retrieve the means and variances and apply the BN transformation
    # mean, var = tf.cond(tf.equal(is_training, True), ema_update, ema_retrieve)
    bn_inputs = tf.nn.batch_normalization(inputs, batch_mean, batch_var, None, None, EPSILON)

    return bn_inputs
