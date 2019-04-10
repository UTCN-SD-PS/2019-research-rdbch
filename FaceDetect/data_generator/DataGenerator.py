import random

from NeuralNetworks.SqueezeDet.config import make_config
from NeuralNetworks.SqueezeDet.data_generator.LoadDataset import read_image_and_label

#================================ OPEN PATHS =============================================
def openPaths(path):
    with open(path, 'r') as f:
        paths = f.readlines()
        for i in range(len(paths)):
            paths[i] = paths[i].strip()

    return paths

#================================ DATA GENERATOR =========================================
def data_generator_path(imgNames, lblNames, config, shuffle=True ):
    assert len(imgNames) == len(lblNames), "Number of images and ground truths not equal"

    #permutate images
    if shuffle:
        shuffled = list(zip(imgNames, lblNames))
        random.shuffle(shuffled)
        imgNames, lblNames = zip(*shuffled)

    count = 1
    epoch = 0
    noBatches, noSkipped = divmod(len(imgNames), config.BATCH_SIZE)

    while True:
        epoch += 1
        i, j = 0, config.BATCH_SIZE

        for _ in range(noBatches):
            #print(i,j)
            imgNames_batch = imgNames[i:j]
            lblNames_batch = lblNames[i:j]
            
            try:
                #get images and ground truths
                imgs, gts = read_image_and_label(imgNames_batch, lblNames_batch, config)
                 
                yield (imgs, gts)

            except IOError as err:
                print( err ) 
                count -= 1

            i = j
            j += config.BATCH_SIZE
            count += 1

# if __name__ == '__main__':
    
#     annPath = 'C:\\Users\\beche\\Documents\\GitHub\\DokProject-master\\Assets\\DummyObjectDataset\\labels.txt'
#     imgPath = 'C:\\Users\\beche\\Documents\\GitHub\\DokProject-master\\Assets\\DummyObjectDataset\\images.txt'

#     annPaths = openPaths(annPath)
#     imgPaths = openPaths(imgPath)

#     config = make_config.load('C:\\Users\\beche\\Documents\GitHub\\DokProject-master\\NeuralNetworks\\SqueezeDet\\config\\SQD.config')

#     a = data_generator_path(imgPaths, annPaths, config) 

#     while next(a):
#         print( next(a) ) 