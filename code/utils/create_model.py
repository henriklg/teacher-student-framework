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
        from efficientnet.tfkeras import EfficientNetB0 as EfficientNet # 4.7M params
    elif conf["model"] == 'EfficientNetB1': 
        from efficientnet.tfkeras import EfficientNetB1 as EfficientNet # 7.2M params
    elif conf["model"] == 'EfficientNetB2':
        from efficientnet.tfkeras import EfficientNetB2 as EfficientNet # 8.5M params
    elif conf["model"] == 'EfficientNetB3':
        from efficientnet.tfkeras import EfficientNetB3 as EfficientNet # 11.5M params
    elif conf["model"] == 'EfficientNetB4':
        from efficientnet.tfkeras import EfficientNetB4 as EfficientNet # 18.6M params
    elif conf["model"] == 'EfficientNetB5':
        from efficientnet.tfkeras import EfficientNetB5 as EfficientNet # 29.5M params
    elif conf["model"] == 'EfficientNetB6':
        from efficientnet.tfkeras import EfficientNetB6 as EfficientNet # 42.2M params
    elif conf["model"] == 'EfficientNetB7':
        from efficientnet.tfkeras import EfficientNetB7 as EfficientNet # 65.4M params
        
    efficientnet_base = EfficientNet(
        weights=conf["weights"],    # "imagenet", or "noisy-student"
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
    

    if conf['optimizer'] == 'Adam':
        opt = tf.keras.optimizers.Adam(learning_rate=conf["learning_rate"])
    elif conf['optimizer'] == 'SGD':
        opt = tf.keras.optimizers.SGD(learning_rate=conf["learning_rate"])

    model.compile(
        optimizer=opt,
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy']
    )
    
    if conf["verbosity"]:
        model.summary()
    
    return model



def create_callbacks(conf):
    """
    Create callbacks used during training of model.
    Returns a list of callbacks.
    """
    callbacks = []
    
    # Tensorboard
    if conf["tensorboard"]:
        file_writer = tf.summary.create_file_writer(conf["log_dir"] + "/metrics")
        file_writer.set_as_default()
        
        tensorboard_cb = TensorBoard(
            log_dir=conf["log_dir"], 
            update_freq='batch'
        )
        callbacks.append(tensorboard_cb)
        
    # Inverse Time Decay LR Scheduler
    if conf["learning_schedule"]:
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

        lr_schedule_cb = LearningRateScheduler(schedule, verbose=1)
        callbacks.append(lr_schedule_cb)
        
    # Early stopping
    if conf["early_stopp"]: 
        earlystopp_cb = EarlyStopping(
            monitor='val_loss', 
            verbose=1, 
            patience=conf["early_stopp_patience"], 
            restore_best_weights=True
        )
        callbacks.append(earlystopp_cb)
        
    # Save checkpoints during training
    if conf["checkpoint"]: 
        checkpoint_cb = ModelCheckpoint(
            filepath=conf["log_dir"]+'/checkpoints/best_cp-{epoch:03d}.hdf', 
            monitor='val_loss', 
            save_best_only=True, 
            mode='auto'
        )
        callbacks.append(checkpoint_cb)
    
    return callbacks