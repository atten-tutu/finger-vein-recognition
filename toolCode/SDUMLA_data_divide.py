import os
import shutil

myDir = './SDUMLA/Finger Vein Database'  # 数据集路径
train_index = 1
test_index = 1

divideDir = "./fingerData/sdu_data_divide"
train_dir = divideDir + "/train/"
test_dir = divideDir + "/test/"

lists = os.listdir(myDir)

try:
    os.mkdir(divideDir)
    os.mkdir(divideDir + "/train")
    os.mkdir(divideDir + "/test")
except:
    pass

for i in lists:
    fpl = myDir + '/' + i + '/left'
    fpr = myDir + '/' + i + '/right'
    lists2 = os.listdir(fpl)
    if train_index < 10:
        i = "00" + str(train_index)
    elif train_index < 100:
        i = "0" + str(train_index)
    else:
        i = str(train_index)
    if int(train_index) + 1 < 10:
        i1 = "00" + str(int(train_index) + 1)
    elif int(train_index) + 1 < 100:
        i1 = "0" + str(int(train_index) + 1)
    else:
        i1 = str(int(train_index) + 1)
    if int(train_index) + 2 < 10:
        i2 = "00" + str(int(train_index) + 2)
    elif int(train_index) + 2 < 100:
        i2 = "0" + str(int(train_index) + 2)
    else:
        i2 = str(int(train_index) + 2)
    try:
        os.mkdir(train_dir + i)
        os.mkdir(train_dir + i1)
        os.mkdir(train_dir + i2)
        os.mkdir(test_dir + i)
        os.mkdir(test_dir + i1)
        os.mkdir(test_dir + i2)
    except:
        pass
    for j in lists2:
        if j[0] == 'i':
            if int(j[-5]) < 5:
                shutil.copyfile(fpl + '/' + j, train_dir + i + "/" + j)
            else:
                shutil.copyfile(fpl + '/' + j, test_dir + i + "/" + j)
        if j[0] == 'm':
            if int(j[-5]) < 5:
                shutil.copyfile(fpl + '/' + j, train_dir + i1 + "/" + j)
            else:
                shutil.copyfile(fpl + '/' + j, test_dir + i1 + "/" + j)
        if j[0] == 'r':
            if int(j[-5]) < 5:
                shutil.copyfile(fpl + '/' + j, train_dir + i2 + "/" + j)
            else:
                shutil.copyfile(fpl + '/' + j, test_dir + i2 + "/" + j)
    train_index += 3

    lists2 = os.listdir(fpr)
    if train_index < 10:
        i = "00" + str(train_index)
    elif train_index < 100:
        i = "0" + str(train_index)
    else:
        i = str(train_index)
    if int(train_index) + 1 < 10:
        i1 = "00" + str(int(train_index) + 1)
    elif int(train_index) + 1 < 100:
        i1 = "0" + str(int(train_index) + 1)
    else:
        i1 = str(int(train_index) + 1)
    if int(train_index) + 2 < 10:
        i2 = "00" + str(int(train_index) + 2)
    elif int(train_index) + 2 < 100:
        i2 = "0" + str(int(train_index) + 2)
    else:
        i2 = str(int(train_index) + 2)
    try:
        os.mkdir(train_dir + i)
        os.mkdir(train_dir + i1)
        os.mkdir(train_dir + i2)
        os.mkdir(test_dir + i)
        os.mkdir(test_dir + i1)
        os.mkdir(test_dir + i2)
    except:
        pass
    for j in lists2:
        if j[0] == 'i':
            if int(j[-5]) < 5:
                shutil.copyfile(fpr + '/' + j, train_dir + i + "/" + j)
            else:
                shutil.copyfile(fpr + '/' + j, test_dir + i + "/" + j)
        if j[0] == 'm':
            if int(j[-5]) < 5:
                shutil.copyfile(fpr + '/' + j, train_dir + i1 + "/" + j)
            else:
                shutil.copyfile(fpr + '/' + j, test_dir + i1 + "/" + j)
        if j[0] == 'r':
            if int(j[-5]) < 5:
                shutil.copyfile(fpr + '/' + j, train_dir + i2 + "/" + j)
            else:
                shutil.copyfile(fpr + '/' + j, test_dir + i2 + "/" + j)
    train_index += 3
