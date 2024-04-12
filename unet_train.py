import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import CSVLogger

IMG_SIZE = 512
B_SIZE = 1

print("\n\nNum GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


"""
DATA GENERATORS TO ALLOCATE DATA IN GPU MEMORY TO AVOID OUT OF MEMORY
"""


datagen = ImageDataGenerator(
    rescale = 1./255,
)

train_imgs_gen = datagen.flow_from_directory(
    'workdata/train/',
    target_size = (IMG_SIZE,IMG_SIZE),
    color_mode = 'rgb',
    batch_size = B_SIZE,
    class_mode = None,
    seed = 1,
    shuffle = True,
    classes = ['imgs']
)

train_mask_gen = datagen.flow_from_directory(
    'workdata/train/',
    target_size = (IMG_SIZE,IMG_SIZE),
    color_mode = 'grayscale',
    batch_size = B_SIZE,
    class_mode = None,
    seed = 1,
    shuffle = True,
    classes = ['mask']
)

val_imgs_generator = datagen.flow_from_directory(
    'workdata/validation/',
    target_size = (IMG_SIZE,IMG_SIZE),
    color_mode = 'rgb',
    batch_size = B_SIZE,
    class_mode = None,
    seed = 1,
    shuffle = True,
    classes = ['imgs']
)

val_mask_generator = datagen.flow_from_directory(
    'workdata/validation/',
    target_size = (IMG_SIZE,IMG_SIZE),
    color_mode = 'grayscale',
    batch_size = B_SIZE,
    class_mode = None,
    seed = 1,
    shuffle = True,
    classes = ['mask']
)

train_generator = (pair for pair in zip(train_imgs_gen, train_mask_gen))
val_generator = (pair for pair in zip(val_imgs_generator, val_mask_generator))

"""
UNET ARCHITECTURE

Added bach normalization layers to boost training 
"""

def conv_block(input_tensor, num_filters):
    x = keras.layers.Conv2D(num_filters, 3, padding="same")(input_tensor)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    x = keras.layers.Conv2D(num_filters, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    return x

def encoder_block(input_tensor, num_filters):
    x = conv_block(input_tensor, num_filters)
    p = keras.layers.MaxPool2D( (2,2) )(x)

    return x, p

def decoder_block(input_tensor, skip_features, num_filters):
    x = keras.layers.Conv2DTranspose( num_filters, (2,2), strides=2, padding="same")(input_tensor)
    x = keras.layers.Concatenate()( [x, skip_features] )
    x = conv_block(x, num_filters)

    return x

def build_unet(input_shape):
    inputs = keras.layers.Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = tf.keras.models.Model(inputs, outputs, name="U-NET")

    return model

"""
METRICS FUNCTIONS
"""
def dice_coefficient_t(mask, predicted_mask, t=0.5):
    mask = tf.cast(mask > 0.5, dtype=tf.float32)  #threshold and convert to float32
    predicted_mask = tf.cast(predicted_mask > t, dtype=tf.float32)  

    intersection = tf.reduce_sum(tf.cast(tf.logical_and(tf.cast(mask, dtype=tf.bool), tf.cast(predicted_mask, dtype=tf.bool)), dtype=tf.float32))
    intersection_area = tf.reduce_sum(intersection)

    mask_area = tf.reduce_sum(mask)
    predicted_mask_area = tf.reduce_sum(predicted_mask)

    dice = (2.0 * intersection_area) / (mask_area + predicted_mask_area)
    return dice

def ev_dice(mask, predicted_mask, t=0.5):

    dice = dice_coefficient_t(mask, predicted_mask, t=t)

    with tf.Session() as sess:
        dice_value = sess.run(dice)
    return dice_value


def jaccard_coefficient_t(mask, predicted_mask, t=0.5):
    mask = tf.cast(mask, dtype=tf.float32)
    predicted_mask = tf.cast(predicted_mask > t, dtype=tf.float32)

    intersection = tf.reduce_sum(tf.cast(tf.logical_and(tf.cast(mask, dtype=tf.bool), tf.cast(predicted_mask, dtype=tf.bool)), dtype=tf.float32))
    intersection_area = tf.reduce_sum(intersection)

    union = tf.reduce_sum(tf.cast(tf.logical_or(tf.cast(mask, dtype=tf.bool), tf.cast(predicted_mask, dtype=tf.bool)), dtype=tf.float32))
    union_area = tf.reduce_sum(union)

    jaccard = intersection_area / union_area
    return jaccard

def ev_jaccard(mask, predicted_mask, t=0.5):

    jaccard = jaccard_coefficient_t(mask, predicted_mask, t=t)

    with tf.Session() as sess:
        jaccard_value = sess.run(jaccard)
    return jaccard_value

"""
UNET INITIALIZATION
"""
input_shape = (IMG_SIZE , IMG_SIZE , 3)

model = build_unet(input_shape)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[dice_coefficient_t, jaccard_coefficient_t])

model.summary()

"""
CALLBACKS
"""
csv_logger_callback = CSVLogger('training_history.csv')

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='model_checkpoint.h5', 
    save_best_only=True,
    monitor='dice_coefficient',
    mode='max'
)

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='logs'  #TensorBoard logs
)


"""
MODEL TRAINING
"""

#model.fit_generator(train_generator, epochs=10, steps_per_epoch=1503)


history = model.fit_generator(
    train_generator,
    epochs=32,
    steps_per_epoch=5472,
    validation_data=val_generator,
    validation_steps=40,
    callbacks=[csv_logger_callback, checkpoint_callback, tensorboard_callback]
)



model.save_weights('Test03.h5')
