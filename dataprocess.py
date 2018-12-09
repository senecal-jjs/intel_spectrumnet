from SpectralNet import GeoTiffDataset
import os 
from torch.utils.data import DataLoader
import torch 


def update(existingAggregate, newValue):
    '''
        returns: std dev = sqrt( M2/(count-1) )
    '''
    (count, mean, M2) = existingAggregate
    count = count + 1 
    delta = newValue - mean
    mean = mean + delta / count
    delta2 = newValue - mean
    M2 = M2 + delta * delta2

    return (count, mean, M2)


if __name__ == "__main__":
    ROOT = os.path.abspath('/Users/senecal/Repos/hyperspectral/data/Tomato2/train')
    assert os.path.exists(ROOT)

    trainset = GeoTiffDataset.DatasetFolder(ROOT, ['.tiff'], num_bands=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    batch_size = 1
    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=4)

    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10 = (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)

    i = 0
    for inputs, labels, path in trainloader:
        print(inputs.size())
        cur_c1_mean = torch.mean(inputs[:, :, :, 0])
        c1 = update(c1, cur_c1_mean)

        cur_c2_mean = torch.mean(inputs[:, :, :, 1])
        c2 = update(c2, cur_c2_mean)

        cur_c3_mean = torch.mean(inputs[:, :, :, 2])
        c3 = update(c3, cur_c3_mean)

        cur_c4_mean = torch.mean(inputs[:, :, :, 3])
        c4 = update(c4, cur_c4_mean)

        cur_c5_mean = torch.mean(inputs[:, :, :, 4])
        c5 = update(c5, cur_c5_mean)

        cur_c6_mean = torch.mean(inputs[:, :, :, 5])
        c6 = update(c6, cur_c6_mean)

        cur_c7_mean = torch.mean(inputs[:, :, :, 6])
        c7 = update(c7, cur_c7_mean)

        cur_c8_mean = torch.mean(inputs[:, :, :, 7])
        c8 = update(c8, cur_c8_mean)

        cur_c9_mean = torch.mean(inputs[:, :, :, 8])
        c9 = update(c9, cur_c9_mean)

        cur_c10_mean = torch.mean(inputs[:, :, :, 9])
        c10 = update(c10, cur_c10_mean)

        if (i % 20 == 0):
            print("iteration: {0}".format(i))

        i += 1

    with open("norms.txt", "w") as f:
        f.write("Band 1: " + str(c1) + "\n")
        f.write("Band 2: " + str(c2) + "\n")
        f.write("Band 3: " + str(c3) + "\n")
        f.write("Band 4: " + str(c4) + "\n")
        f.write("Band 5: " + str(c5) + "\n")
        f.write("Band 6: " + str(c6) + "\n")
        f.write("Band 7: " + str(c7) + "\n")
        f.write("Band 8: " + str(c8) + "\n")
        f.write("Band 9: " + str(c9) + "\n")
        f.write("Band 10: " + str(c10) + "\n")
