from model import *
from data import *
from keras import objectives
import datetime

class GAN():
    def __init__(self):
        #初始化图像大小
        self.channels = 3
        self.i_height = 256
        self.i_width = 256
        self.i_shape = (self.i_height,self.i_width, self.channels)
        self.o_shape = (self.i_height,self.i_width, self.channels)

        #卷积核的数量
        self.gf = 64
        self.df = 64

        self.disc_patch = (256,256,1)

        #构建判别器
        optimizer = Adam(0.0002, 0.5)
        #self.discriminator = self.build_discriminator()
        self.discriminator = self.discriminator_image()
        self.discriminator.compile(loss='mse',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        #构建生成器
        self.generator = self.build_generator()

        img_org = Input(shape=self.i_shape)
        img_out = Input(shape=self.o_shape)

        fake_out = self.generator(img_org)

        #冻结权重 在整体网络中只训练生成器
        self.discriminator.trainable = False

        validity = self.discriminator(fake_out)

        #形成gan网络
        self.combined = Model([img_org,img_out],[validity,fake_out])
        #validity是由原始图像经过生成器和判别器形成的
        #fake_out是由生成器生成的
        #相当于训练两次 但是给的是两个不同的目标对生成器进行训练
        self.combined.compile(loss=['binary_crossentropy', 'mse'],
                              loss_weights=[1e-3, 1e-3],
                              optimizer=optimizer)
    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.i_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False)
        d2 = conv2d(d1, self.gf * 2)
        d3 = conv2d(d2, self.gf * 4)
        d4 = conv2d(d3, self.gf * 8)
        d5 = conv2d(d4, self.gf * 8)
        d6 = conv2d(d5, self.gf * 8)
        d7 = conv2d(d6, self.gf * 8)

        # Upsampling
        u1 = deconv2d(d7, d6, self.gf * 8)
        u2 = deconv2d(u1, d5, self.gf * 8)
        u3 = deconv2d(u2, d4, self.gf * 8)
        u4 = deconv2d(u3, d3, self.gf * 4)
        u5 = deconv2d(u4, d2, self.gf * 2)
        u6 = deconv2d(u5, d1, self.gf)

        u7 = UpSampling2D(size=2)(u6)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

        return Model(d0, output_img)

    def discriminator_image(self):
        """
        discriminator network (patch GAN)
          stride 2 conv X 2
            max pooling X 4
        fully connected X 1
        """
        img_size=(256,256)
        n_filters = 32
        # set image specifics
        k = 3  # kernel size
        s = 2  # stride
        img_ch = 3  # image channels
        out_ch = 1  # output channel
        img_height, img_width = img_size[0], img_size[1]
        padding = 'same'  # 'valid'

        inputs = Input((img_height, img_width, img_ch))

        conv1 = Conv2D(n_filters, kernel_size=(k, k), strides=(s, s), padding=padding)(inputs)
        conv1 = BatchNormalization(scale=False, axis=3)(conv1)
        conv1 = Activation('relu')(conv1)
        conv1 = Conv2D(n_filters, kernel_size=(k, k), padding=padding)(conv1)
        conv1 = BatchNormalization(scale=False, axis=3)(conv1)
        conv1 = Activation('relu')(conv1)
        pool1 = MaxPooling2D(pool_size=(s, s))(conv1)

        conv2 = Conv2D(2 * n_filters, kernel_size=(k, k), strides=(s, s), padding=padding)(pool1)
        conv2 = BatchNormalization(scale=False, axis=3)(conv2)
        conv2 = Activation('relu')(conv2)
        conv2 = Conv2D(2 * n_filters, kernel_size=(k, k), padding=padding)(conv2)
        conv2 = BatchNormalization(scale=False, axis=3)(conv2)
        conv2 = Activation('relu')(conv2)
        pool2 = MaxPooling2D(pool_size=(s, s))(conv2)

        conv3 = Conv2D(4 * n_filters, kernel_size=(k, k), padding=padding)(pool2)
        conv3 = BatchNormalization(scale=False, axis=3)(conv3)
        conv3 = Activation('relu')(conv3)
        conv3 = Conv2D(4 * n_filters, kernel_size=(k, k), padding=padding)(conv3)
        conv3 = BatchNormalization(scale=False, axis=3)(conv3)
        conv3 = Activation('relu')(conv3)
        pool3 = MaxPooling2D(pool_size=(s, s))(conv3)

        conv4 = Conv2D(8 * n_filters, kernel_size=(k, k), padding=padding)(pool3)
        conv4 = BatchNormalization(scale=False, axis=3)(conv4)
        conv4 = Activation('relu')(conv4)
        conv4 = Conv2D(8 * n_filters, kernel_size=(k, k), padding=padding)(conv4)
        conv4 = BatchNormalization(scale=False, axis=3)(conv4)
        conv4 = Activation('relu')(conv4)
        pool4 = MaxPooling2D(pool_size=(s, s))(conv4)

        conv5 = Conv2D(16 * n_filters, kernel_size=(k, k), padding=padding)(pool4)
        conv5 = BatchNormalization(scale=False, axis=3)(conv5)
        conv5 = Activation('relu')(conv5)
        conv5 = Conv2D(16 * n_filters, kernel_size=(k, k), padding=padding)(conv5)
        conv5 = BatchNormalization(scale=False, axis=3)(conv5)
        conv5 = Activation('relu')(conv5)

        gap = GlobalAveragePooling2D()(conv5)
        outputs = Dense(1, activation='sigmoid')(gap)

        d = Model(inputs, outputs)
        '''
        def d_loss(y_true, y_pred):
            L = objectives.binary_crossentropy(K.batch_flatten(y_true),
                                               K.batch_flatten(y_pred))
            #         L = objectives.mean_squared_error(K.batch_flatten(y_true),
            #                                            K.batch_flatten(y_pred))
            return L

        d.compile(optimizer=Adam(lr=init_lr, beta_1=0.5), loss=d_loss, metrics=['accuracy'])
        '''
        return d
    def build_discriminator(self):

        ndf = 64
        input_size = (256, 256, 3)
        n_layers, use_sigmoid = 3, False
        inputs = Input(shape=input_size)

        x = Conv2D(filters=ndf, kernel_size=(4, 4), strides=2, padding='same')(inputs)
        x = LeakyReLU(0.2)(x)

        nf_mult, nf_mult_prev = 1, 1
        for n in range(n_layers):
            nf_mult_prev, nf_mult = nf_mult, min(2 ** n, 8)
            x = Conv2D(filters=ndf * nf_mult, kernel_size=(4, 4), strides=2, padding='same')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(0.2)(x)

        nf_mult_prev, nf_mult = nf_mult, min(2 ** n_layers, 8)
        x = Conv2D(filters=ndf * nf_mult, kernel_size=(4, 4), strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)

        x = Conv2D(filters=1, kernel_size=(4, 4), strides=1, padding='same')(x)
        if use_sigmoid:
            x = Activation('sigmoid')(x)

        x = Flatten()(x)
        x = Dense(1024, activation='tanh')(x)
        validity = Dense(1, activation='sigmoid')(x)



        return Model( inputs, validity)

    def train(self,epochs,batch_size = 30):

        start_time = datetime.datetime.now()

        for epoch in range(epochs):
            self.generator.load_weights('unet_G_membrane2.hdf5')
            self.discriminator.load_weights('unet_D_membrane2.hdf5')
            #gan_testGenerator("data/membrane/train/paper_image",num_image = 1)
            img_real = gan_testGenerator("data/membrane/train/paper_label",num_image = 1)
            img_org = gan_testGenerator("data/membrane/train/paper_image",num_image = 1)

            img_fake=[]
            for i in range(batch_size):
                #x = i%30
                x = 0
                img = np.reshape(img_org[x], (1,) + img_org[x].shape)
                img_g = self.generator.predict(img)
                img_fake.append(img_g)

            valid = np.ones((1, 1))
            fake = np.zeros((1, 1))

            #训练判别器
            for i in range(batch_size):
                #x = i % 30
                x = 0
                img = np.reshape(img_real[x],(1,)+img_real[x].shape)
                #在前面加一个维度
                d_loss_real = self.discriminator.train_on_batch(img, valid)

                img0 = img_fake[x]
                d_loss_fake = self.discriminator.train_on_batch(img0, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                #print('d_loss:%f' % (d_loss))
            self.discriminator.save_weights('unet_D_membrane2.hdf5')
            self.discriminator.trainable = False

            #训练生成器
            for i in range(batch_size*10):
                #x = i % 30
                x = 0
                img0 = np.reshape(img_real[x], (1,) + img_real[x].shape)
                img1 = np.reshape(img_org[x], (1,) + img_org[x].shape)
                g_loss = self.combined.train_on_batch([img1,img0 ], [valid,img0])
                elapsed_time = datetime.datetime.now() - start_time
                print("%d %d time: %s" % (i, epoch, elapsed_time))
                #print("  g_loss:%f\n" % (g_loss))
            self.generator.save_weights('unet_G_membrane2.hdf5')

            #输出信息
    def LoadWeight(self):
        self.generator.load_weights('unet_G_membrane2.hdf5')
        self.discriminator.load_weights('unet_D_membrane2.hdf5')


if __name__ == '__main__':
    gan =  GAN()
    gan.LoadWeight()
    #gan.train(epochs=5, batch_size=1)
    testGene1 = testGenerator("data/membrane/train/paper_image")
    results = gan.generator.predict_generator(testGene1, 1, verbose=1)
    saveResult("data/membrane/train/out", results)