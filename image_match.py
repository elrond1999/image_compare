#
#  Matches the Top and Bottom half of a picture to a database of pictures.
#

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

MIN_MATCH_COUNT = 10

#Class to hold an image match
class ImageMatch:
    #Orb
    orb = cv2.ORB_create()
    #sift = cv2.features2d.SIFT_create()
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def __init__(self, img, name, ref=None, **kwargs):
        self.img = img
        #Name of image
        self.name = name
        self.matches = None
        self.up_matches = None
        self.down_matches = None
        #Find keypoints on img
        print("Reading %s" % (name))
        self.kp, self.des = ImageMatch.orb.detectAndCompute(img,None)
        #self.kp, self.des = ImageMatch.sift.detectAndCompute(img,None)

        if ref != None:
            self.set_reference(ref)
        return super().__init__(**kwargs)
        
    #Use another image as reference
    def set_reference(self,ref):
        self.ref = ref
        self.matches = None
        self.up_matches = None
        self.down_matches = None

    def match(self):
        if self.matches == None:
            print("Matching %s vs %s" % (self.name, self.ref.name))
            # Match descriptors.
            #self.matches = ImageMatch.bf.knnMatch(self.des,self.ref.des, k=2)
            self.matches = ImageMatch.bf.match(self.des,self.ref.des)
            # Sort them in the order of their distance.
            self.matches = sorted(self.matches, key = lambda x:x.distance)
            
        return self.matches
    
    #Split matches in upper and lower part of image
    def up_or_down(self):
        y_max = self.img.shape[1]
        y_half = y_max / 2
        up = []
        down = []
        for m in self.match():
            kp = self.kp[m.queryIdx]
            y = kp.pt[1]
            if y > y_half:
                up.append(m)
            else:
                down.append(m)

        #Store them sorted
        self.up_matches   = sorted(up,   key = lambda x:x.distance)
        self.down_matches = sorted(down, key = lambda x:x.distance)

    def upper_matches(self):
        if self.up_matches == None:
            self.up_or_down()
        return self.up_matches

    def lower_matches(self):
        if self.down_matches == None:
            self.up_or_down()
        return self.down_matches

    def distance(self):
        #Average distance of the 10 best matches in full picture
        matches = self.match()[:10]
        return sum(m.distance for m in matches)/(len(matches) + 0.0001)

    def upper_distance(self):
        #Sum of distance of the 10 best matches in upper part of picture
        matches = self.upper_matches()[:10]
        return sum(m.distance for m in matches)/(len(matches) + 0.0001)


    def lower_distance(self):
        #Sum of distance of the 10 best matches in lower part of picture
        matches = self.lower_matches()[:10]
        return sum(m.distance for m in matches)/(len(matches) + 0.0001)


    def matchplot(self):
        fig = plt.figure()
        a = fig.add_subplot(1,1,1)
        a.set_title("%s distance %d" % (self.name, self.distance()))
        img = cv2.drawMatches(self.img    ,self.kp,
                              self.ref.img,self.ref.kp, self.match()[:10], None, flags=2)
        plt.imshow(img)
        




#Read images
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename), 0)
        if img is not None:
            images.append(ImageMatch(img,filename))
    return images


if __name__ == '__main__':
    
    #Read in the query image
    ref = ImageMatch(cv2.imread('questions/Screenshot_2.jpg',0),"Dec2")


    images = load_images_from_folder("pictures")
    for image in images:
        image.set_reference(ref)

    #For all matches.
    images   = sorted(images, key = lambda x:x.distance())
    images[0].matchplot()
    images[1].matchplot()
    images[2].matchplot()
    plt.show()


    #Sort based on distance
    up_images   = sorted(images, key = lambda x:x.upper_distance())
    down_images = sorted(images, key = lambda x:x.lower_distance())

    print("Best upper match: %s" % (up_images[0].name))
    print("Best lower match: %s" % (down_images[0].name))
    up_images[0].matchplot()
    down_images[0].matchplot()
    plt.show()
    
    print("Done")



