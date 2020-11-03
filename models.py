from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Input, Dense, Reshape, Conv2D, Conv2DTranspose, Flatten, Dropout, LeakyReLU, BatchNormalization, Activation
from keras.utils.vis_utils import plot_model
from keras.models import Model
 
# Define Discriminator Model
def defineDiscriminator(df, img_size):

    def d_block(layer_input, filters, strides=1, bn=True):
        """Discriminator layer"""
        d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
        if bn:
            d = BatchNormalization(momentum=0.9)(d)
        d = LeakyReLU(alpha=0.2)(d)

        return d

    # Input img = generated image
    d0 = Input(shape=(img_size, img_size, 3))

    d = d_block(d0, df, strides=2, bn=False)
    d = d_block(d, df*2, strides=2)
    d = d_block(d, df*4, strides=2)
    d = d_block(d, df*8, strides=2)

    d = Flatten()(d)
    validity = Dense(1, activation='sigmoid')(d)
    model = Model(d0, validity)
    return model

def defineDiscriminator2(df, img_size):

    def d_block(layer_input, filters, strides=1, bn=True):
        """Discriminator layer"""
        d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
        if bn:
            d = BatchNormalization(momentum=0.9)(d)
        d = LeakyReLU(alpha=0.2)(d)

        return d

    # Input img = generated image
    d0 = Input(shape=(img_size, img_size, 3))

    d = d_block(d0, df, strides=2, bn=False)
    d = d_block(d, df*2, strides=2)
    d = d_block(d, df*4, strides=2)
    d = d_block(d, df*8, strides=2)

    d = Flatten()(d)
    validity = Dense(1, activation='sigmoid')(d)
    model = Model(d0, validity)
    model.compile(loss='mse',
                  optimizer=Adam(learning_rate=1e-5, beta_1=0.5, decay=0.00005),
                  metrics=['binary_accuracy'])
    return model

def defineGenerator(latent_dim, gf, img_size=64):
    noise = Input(shape=(latent_dim,))

    def deconv2d(layer_input, filters=256, kernel_size=(5, 5), strides=(2, 2), bn_relu=True):
        u = Conv2DTranspose(filters, kernel_size=kernel_size, strides=strides, padding='same')(layer_input)
        if bn_relu:
            u = BatchNormalization(momentum=0.9)(u)
            u = Activation('relu')(u)
        return u

    generator = Dense(16 * gf * img_size // 16 * img_size // 16, activation="relu")(noise)
    generator = Reshape((img_size // 16, img_size // 16, gf * 16))(generator)
    generator = BatchNormalization()(generator)
    generator = Activation('relu')(generator)
    generator = deconv2d(generator, filters=gf * 8)
    generator = deconv2d(generator, filters=gf * 4)
    generator = deconv2d(generator, filters=gf * 2)
    generator = deconv2d(generator, filters=gf    )
    generator = deconv2d(generator, filters=3, kernel_size=(3,3), strides=(1,1), bn_relu=False)

    gen_img = Activation('sigmoid')(generator)
    model = Model(noise, gen_img)
    return model

def defineGenerator2(latent_dim, gf, img_size=64):
    noise = Input(shape=(latent_dim,))

    def deconv2d(layer_input, filters=256, kernel_size=(5, 5), strides=(2, 2), bn_relu=True):
        u = Conv2DTranspose(filters, kernel_size=kernel_size, strides=strides, padding='same')(layer_input)
        if bn_relu:
            u = BatchNormalization(momentum=0.9)(u)
            u = Activation('relu')(u)
        return u

    generator = Dense(16 * gf * img_size // 16 * img_size // 16, activation="relu")(noise)
    generator = Reshape((img_size // 16, img_size // 16, gf * 16))(generator)
    generator = BatchNormalization()(generator)
    generator = Activation('relu')(generator)
    generator = deconv2d(generator, filters=gf * 8)
    generator = deconv2d(generator, filters=gf * 4)
    generator = deconv2d(generator, filters=gf * 2)
    generator = deconv2d(generator, filters=gf    )
    generator = deconv2d(generator, filters=3, kernel_size=(3,3), strides=(1,1), bn_relu=False)

    gen_img = Activation('sigmoid')(generator)
    model = Model(noise, gen_img)
    model.compile(loss='mse',
                  optimizer=Adam(learning_rate=1e-4, beta_1=0.5, decay=0.00005),
                  metrics=['mse'])
    return model

def defineGan(genModel, discModel):
    # make discriminator not trainable
    discModel.trainable = False
    
    # connect the models
    model = Sequential()
    model.add(genModel)
    model.add(discModel)
    
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

def make_trainable(model, val):
    model.trainable = val
    for lay in model.layers:
        lay.trainable = val

def defineGan2(genModel, discModel):
    z = Input(shape=(41,))
    fake_img = genModel(z)

    make_trainable(discModel, False)

    validity = discModel(fake_img)

    combined = Model([z], [validity])
    combined.compile(loss=['binary_crossentropy'],
                          optimizer=Adam(learning_rate=1e-4, beta_1=0.5, decay=0.00005))
    return combined