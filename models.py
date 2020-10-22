from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Reshape, Conv2D, Conv2DTranspose, Flatten, Dropout, LeakyReLU, BatchNormalization, Activation

# Define Discriminator Model
def defineDiscriminator(df, img_size=(192,160)):
    def d_block(layer_input, filters, strides=1, bn=True):
        d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
        if bn:
            d = BatchNormalization(momentum=0.9)(d)
        d = LeakyReLU(alpha=0.2)(d)

        return d
    
    d0 = Input(shape=(img_size[0], img_size[1], 3))

    d = d_block(d0, df, strides=2, bn=False)
    d = d_block(d, df*2, strides=2)
    d = d_block(d, df*4, strides=2)
    d = d_block(d, df*8, strides=2)

    d = Flatten()(d)
    validity = Dense(1, activation='sigmoid')(d)
    model = Model(d0, validity)
    return model

#Define Generator Model
def defineGenerator(latent_dim, gf, img_size=(192,160)):
    noise = Input(shape=(latent_dim,))

    def deconv2d(layer_input, filters=256, kernel_size=(5, 5), strides=(2, 2), bn_relu=True):
        u = Conv2DTranspose(filters, kernel_size=kernel_size, strides=strides, padding='same')(layer_input)
        if bn_relu:
            u = BatchNormalization(momentum=0.9)(u)
            u = Activation('relu')(u)
        return u

    generator = Dense(16 * gf * img_size[0] // 16 * img_size[1] // 16, activation="relu")(noise)
    generator = Reshape((img_size[0] // 16, img_size[1] // 16, gf * 16))(generator)
    generator = BatchNormalization()(generator)
    generator = Activation('relu')(generator)
    generator = deconv2d(generator, filters=gf * 8)
    generator = deconv2d(generator, filters=gf * 4)
    generator = deconv2d(generator, filters=gf * 2)
    generator = deconv2d(generator, filters=gf    )
    generator = deconv2d(generator, filters=3, kernel_size=(3,3), strides=(1,1), bn_relu=False)

    gen_img = Activation('tanh')(generator)
    model = Model(noise, gen_img)
    return model