import os


def imageGenerator(folderPath):
    for filename in os.listdir(folderPath):
        # if it is a folder - open it and loop over the images
        if os.path.isdir(folderPath + filename):
            for image_path in os.listdir(folderPath + filename):
                print("hoi")
                yield image_path


def getImageList(folderPath):
    imageList = []
    for filename in os.listdir(folderPath):
        # if it is a folder - open it and loop over the images
        if os.path.isdir(folderPath + filename):
            for image_path in os.listdir(folderPath + filename):
                imageList.append(image_path)
    return imageList


print(getImageList("100 animals/"))
    # predict(image)
