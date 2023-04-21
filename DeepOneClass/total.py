
from keras.applications import MobileNetV2, VGG16
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense
from keras import backend as K
from keras.engine.network import Network

input_shape = (96, 96, 3)
classes = 10
batchsize = 128
#feature_out = 512 #secondary network out for VGG16
feature_out = 1280 #secondary network out for MobileNet
alpha = 0.5 #for MobileNetV2
lambda_ = 0.1 #for compact loss

#損失関数
def original_loss(y_true, y_pred):
    lc = 1/(classes*batchsize) * batchsize**2 * K.sum((y_pred -K.mean(y_pred,axis=0))**2,axis=[1]) / ((batchsize-1)**2)
    return lc

#学習
def train(x_target, x_ref, y_ref, epoch_num):

    # VGG16読み込み, S network用
    print("Model build...")
    #mobile = VGG16(include_top=False, input_shape=input_shape, weights='imagenet')

    # mobile net読み込み, S network用
    mobile = MobileNetV2(include_top=True, input_shape=input_shape, alpha=alpha,
                         , weights='imagenet')
