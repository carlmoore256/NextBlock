

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, Conv1DTranspose, Activation, BatchNormalization, Input, Concatenate



def create_layer(input, filters, kernel_size, strides, activation, use_bias, upsampling):
  if upsampling:
    layer = Conv1DTranspose(filters, kernel_size=(kernel_size), strides=strides, padding='same', use_bias=use_bias)(input)
  else:
    layer = Conv1D(filters, kernel_size=(kernel_size), strides=strides, padding='same', use_bias=use_bias)(input)

  layer = Activation(activation)(layer)
  layer = BatchNormalization()(layer)
  output_shape = layer.shape
  return layer, output_shape

def create_model(input_shape, filters, 
                 kernel_size, bottleneck, 
                 strides=2, activation='tanh', 
                 use_bias=False):
  kernel_size = (kernel_size,)
  model_input = Input(shape=input_shape)

  encoder_layers = []

  encoder, output_shape = create_layer(model_input, filters, 
                                       kernel_size, strides,
                                       activation, use_bias,
                                       upsampling=False)
  encoder_layers.append(encoder)
  current_filt = filters//2
  while output_shape[1] > bottleneck:
    encoder, output_shape = create_layer(encoder, current_filt, 
                                         kernel_size, strides,
                                         activation, use_bias,
                                         upsampling=False)
    encoder_layers.append(encoder)
    current_filt = current_filt//2

  decoder, output_shape = create_layer(encoder, bottleneck, 
                                       kernel_size, strides, 
                                       activation, use_bias,
                                       upsampling=False)
  current_filt *= 2

  while output_shape[1] < input_shape[0]//2:
    decoder, output_shape = create_layer(decoder, current_filt, 
                                         kernel_size, strides, 
                                         activation, use_bias,
                                         upsampling=True)
    skip_layer = encoder_layers.pop(-1)
    decoder = Concatenate()([decoder, skip_layer])
    current_filt *= 2
  
  model_output, _ = create_layer(decoder, 2, kernel_size, 
                                 strides, activation, use_bias,
                                 upsampling=True)
  
  model = Model(inputs=[model_input], outputs=[model_output])
  return model