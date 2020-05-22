from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.callbacks import LearningRateScheduler, TensorBoard
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.keras import layers


def create_model(conf):
    """
    """
    if conf["model"] == 'EfficientNetB0': 
        from efficientnet import EfficientNetB0 as EfficientNet # 5.3M params
    elif conf["model"] == 'EfficientNetB1': 
        from efficientnet import EfficientNetB1 as EfficientNet # 7.8M params
    elif conf["model"] == 'EfficientNetB2':
        from efficientnet import EfficientNetB2 as EfficientNet # 9.2M params
    elif conf["model"] == 'EfficientNetB3':
        from efficientnet import EfficientNetB3 as EfficientNet # 12M params
        
    efficientnet_base = EfficientNet(
        weights="imagenet",
        include_top=False, 
        input_shape=conf["img_shape"]
    )
    
    # Unfreeze the layers. I.E we're just using the pre-trained 
    # weights as initial weigths and biases and train over them
    efficientnet_base.trainable = True

    # Define model
    model = Sequential()
    model.add(efficientnet_base)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(conf["dropout"]))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(conf["dropout"]))
    model.add(layers.Dense(conf["num_classes"], activation=conf["final_activation"]))
    
    # from keras import regularizers
    # teacher_model = Sequential()
    # teacher_model.add(efficientnet_base)
    # teacher_model.add(layers.GlobalAveragePooling2D())
    # teacher_model.add(layers.Dropout(0.5))
    # teacher_model.add(layers.Dense(512, kernel_regularizer=regularizers.l2(0.001),
    #                                activation='relu'))
    # teacher_model.add(layers.Dropout(0.5))
    # teacher_model.add(layers.Dense(512, kernel_regularizer=regularizers.l2(0.001),
    #                                activation='relu'))
    # teacher_model.add(layers.Dropout(0.4))
    # teacher_model.add(layers.Dense(256, kernel_regularizer=regularizers.l2(0.001),
    #                                activation='relu'))
    # teacher_model.add(layers.Dropout(0.3))
    # teacher_model.add(layers.Dense(params["num_classes"], activation=conf["final_activation"]))

    if conf['optimizer'] == 'Adam':
        opt = tf.keras.optimizers.Adam(learning_rate=conf["learning_rate"])
    elif conf['optimizer'] == 'SGD':
        opt = tf.keras.optimizers.SGD(learning_rate=conf["learning_rate"])

    model.compile(
        optimizer=opt,
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy']
    )

    model.summary()
    
    return model



def create_callbacks(conf):
    """
    """
    # By using LearnignRateScheduler
    initial_learning_rate = conf["learning_rate"]
    decay_steps = conf["steps"]["train"]
    batch_size = conf['batch_size']
    decay_rate = conf['decay_rate']

    def schedule(epoch):
        # calculate new learning rate
        learning_rate = initial_learning_rate / (1 + decay_rate * (epoch*batch_size) / decay_steps)
        # update tensorboard
        tf.summary.scalar(name='learning_rate', data=learning_rate, step=epoch)
        return learning_rate

    file_writer = tf.summary.create_file_writer(conf["log_dir"] + "/metrics")
    file_writer.set_as_default()

    lr_schedule_cb = LearningRateScheduler(schedule, verbose=1)
    earlystopp_cb = EarlyStopping(
        monitor='val_loss', 
        verbose=1, 
        patience=conf["early_stopp_patience"], 
        restore_best_weights=True
    )
    checkpoint_cb = ModelCheckpoint(
        filepath=conf["log_dir"]+'/best_cp-{epoch:03d}.hdf', 
        monitor='val_loss', 
        save_best_only=True, 
        mode='auto'
    )
    tensorboard_cb = TensorBoard(
        log_dir=conf["log_dir"], 
        update_freq='batch'
    )

    callbacks = [tensorboard_cb]
    if conf["early_stopp"]: callbacks.append(earlystopp_cb)
    if conf["learning_schedule"]: callbacks.append(lr_schedule_cb)
    if conf["checkpoint"]: callbacks.append(checkpoint_cb)
        
    return callbacks