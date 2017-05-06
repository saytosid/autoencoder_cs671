from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
from keras.preprocessing import image
import os
import matplotlib.pyplot as plt

def get_img(img_path):
    img = image.load_img(img_path, target_size=(128, 128)).convert(mode="L")
    # print "img ",
    x = image.img_to_array(img)
    # print "x ",x.shape
    # x = np.expand_dims(x, axis=0)
    x = x.reshape((16384,1))
    # plt.imshow(x)
    # plt.gray()
    # plt.show()
    # print "x ",x.shape
    return x

def load_imgs_from_folder(folder_path):
    print folder_path
    imgs = os.listdir(folder_path)
    return_list = []
    for i in imgs:
        img_path = folder_path+i
        return_list.append(get_img(img_path))
    return return_list

def load_my_data():
    class0_folder = "data/Ear/"
    class1_folder = "data/Iris/"
    class2_folder = "data/Knuckle/"
    class3_folder = "data/Palm/"
    data = []
    data_test = []
    l1 = load_imgs_from_folder(class0_folder)
    l2 = load_imgs_from_folder(class1_folder)
    l3 = load_imgs_from_folder(class2_folder)
    l4 = load_imgs_from_folder(class3_folder)
    ctr = 0
    for i in l1:
        if ctr>(0.8*len(l1)):
            data_test.append(i)
        ctr += 1
        data.append(i)
    ctr = 0
    for i in l2:
        if ctr>(0.8*len(l2)):
            data_test.append(i)
        ctr += 1
        data.append(i)
    ctr = 0
    for i in l3:
        if ctr>(0.8*len(l3)):
            data_test.append(i)
        ctr += 1
        data.append(i)
    ctr = 0
    for i in l4:
        if ctr>(0.8*len(l4)):
            data_test.append(i)
        ctr += 1
        data.append(i)
    return np.array(data),np.array(data_test)
if __name__=="__main__":
    # get_img("data/Ear/ear_roi1.JPEG")
    # this is the size of our encoded representations
    encoding_dim = 64  # 32 floats -> compression of factor 24.5, assuming the input is 16384 floats

    # this is our input placeholder
    input_img = Input(shape=(16384,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(16384, activation='sigmoid')(encoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)

    # this model maps an input to its encoded representation
    encoder = Model(input_img, encoded)

    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_dim,))
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


    # (x_train, _), (x_test, _) = mnist.load_data()
    x_train, x_test = load_my_data() 
    print len(x_train)
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    print x_train.shape
    print x_test.shape

    autoencoder.fit(x_train, x_train,
                    epochs=10,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(x_test, x_test))

    # encode and decode some digits
    # note that we take them from the *test* set
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)

    # use Matplotlib (don't ask)
    import matplotlib.pyplot as plt

    n = 5  # how many imgs we will display
    plt.figure()
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape((128,128)))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape((128,128)))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()