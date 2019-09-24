from model import *
from data import *
from model1 import *
from PIL import Image
from keras.models import Model
import cv2
import keras
import os
from LoadData import *
from keras.applications.vgg16 import VGG16


#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def unet1():
    data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
    myGene = trainGenerator(2,'data/membrane/train','paper_image','paper_label',data_gen_args,save_to_dir = None)
    #myGene = trainGenerator(2, 'data/membrane/train', 'orgin', 'orgin_label', data_gen_args, save_to_dir=None)
    #myGene = trainGenerator1('data/membrane/train/orgin','data/membrane/train/orgin_label',num_image=1)
    model = unet()
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    model_checkpoint = ModelCheckpoint('unet_membrane1.hdf5', monitor='loss',verbose=1, save_best_only=True)

    #model.load_weights('unet_membrane1.hdf5')
    #训练函数
    model.fit_generator(myGene,steps_per_epoch=20,epochs=1,callbacks=[model_checkpoint])

    #for i in  range(10):
        #loss = model.train_on_batch(myGene[0],myGene[1])
        #print(i, loss[0] ,loss[1])
    #model.save_weights('unet_membrane1.hdf5')
    #载入已有函数
    #model.load_weights('unet_membrane.hdf5')
    testGene = testGenerator("data/membrane/train/paper_image",num_image = 1)

    results = model.predict_generator(testGene,1,verbose=1)
    saveResult("data/membrane/train/out",results)
'''
def unetgan():

    model_G=G_unet()
    model_D=D_unet_lpgan()
    # 形成gan网络
    gan_input = keras.Input((256, 256, 1))
    gan_output = model_D(model_G(gan_input))
    gan = keras.models.Model(gan_input, gan_output)

    #gan.load_weights('gan.hdf5')
    gan_optimizer = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    gan.compile(optimizer=gan_optimizer, loss=wasserstein_loss)

    #gan_checkpoint = ModelCheckpoint('gan.hdf5', monitor='loss', verbose=1, save_best_only=True)
    model_D.load_weights('unet_D_membrane.hdf5')
    model_G.load_weights('unet_G_membrane1.hdf5')  # 载入已经训练好的生成器
    model_G.predict(np.zeros([1,256,256,1]))
    #model_G_checkpoint = ModelCheckpoint('unet_G_membrane.hdf5', monitor='loss', verbose=1, save_best_only=True)
    model_D_checkpoint = ModelCheckpoint('unet_D_membrane.hdf5', monitor='loss', verbose=1, save_best_only=True)

    #训练过程

    #生成器训练完成
    real_image = LoadRealImage('data/membrane/train/label',30)
    #提取出真实文件
    fake_image= PreFakeImage('data/membrane/train/image',model_G,30)
    #通过生成器生成假的图片

    #model_D.fit_generator(real_image,steps_per_epoch=30, epochs=1, callbacks=[model_D_checkpoint])
    #model_D.fit_generator(fake_image,steps_per_epoch=30, epochs=1, callbacks=[model_D_checkpoint])
    #训练判别器
    model_D.trainable = False
    #冻结判别器权重


    #输入准备好的图像
    #对生成器进行训练

    target = np.ones((30, 1))
    testGene = gan_testGenerator("data/membrane/train/image")
    loss = gan.train_on_batch(testGene,target)
    print (loss)
    model_G.save_weights('unet_G_membrane1.hdf5')

    #测试
    testGene1 = testGenerator("data/membrane/1")
    results = model_G.predict_generator(testGene1, 30, verbose=1)
    saveResult("data/membrane/gan", results)

'''

unet1()
#unetgan()
'''
image_shape = (256, 256, 1)
def generator_containing_discriminator_multiple_outputs(generator, discriminator):
    inputs = Input(shape=image_shape)
    generated_image = generator(inputs)
    outputs = discriminator(generated_image)
    model = Model(inputs=inputs, outputs=[generated_image, outputs])
    return model

def perceptual_loss(y_true, y_pred):
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=image_shape)
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False
    return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))

def unetgan1():
    g=G_unet()
    d=D_unet_lpgan()
    d_on_g_opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    d_on_g = generator_containing_discriminator_multiple_outputs(g, d)
    d.trainable = True
    d.compile(optimizer=Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08), loss=wasserstein_loss)
    d.trainable= False
    loss = [perceptual_loss, wasserstein_loss]
    loss_weights = [100, 1]
    d_on_g.compile(optimizer=d_on_g_opt, loss=loss, loss_weights=loss_weights)
    d.trainable = True
'''