import train
import time

data = 'coco'
saveto = 'vse/%s' %data
encoder = 'lstm'

if __name__ == "__main__":
    begin_time = time.time()
    train.trainer(data=data, dim_image=4096, lrate=0.001, margin=0.2, encoder=encoder, max_epochs=100000, batch_size=300,
                dim=1000, dim_word=1000, maxlen_w=150, dispFreq=10, validFreq=100, early_stop=10, saveto=saveto)

    print('Using %.2f s' %(time.time()-begin_time))