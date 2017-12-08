#!/bin/env python
#
#  Matches the Top and Bottom half of a picture to a database of pictures.
#

import numpy as np
import cv2
import os
import sys
import matplotlib.pyplot as plt
#import code
from tkinter import *

#Class to hold an image match
class ImageMatch:
    #Orb
    orb = cv2.ORB_create()
    #sift = cv2.xfeatures2d.SIFT_create()
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
            #self.matches = ImageMatch.bf.knnMatch(self.des,self.ref.des, k=1)
            self.matches = ImageMatch.bf.match(self.des,self.ref.des)
            # Sort them in the order of their distance.
            self.matches = sorted(self.matches, key = lambda x:x.distance)
            print("Distance %f, %d matches" % (self.distance(), len(self.matches)))

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
        avg = sum(m.distance for m in matches)/(len(matches) + 0.0001)
        avg_scaled = avg*(10/len(matches))
        return avg_scaled

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

    #Read in the query images
    refs = load_images_from_folder("questions")
    images = load_images_from_folder("pictures")

    master = Tk()
    master.title("Picure matcher")

    var = StringVar(master)

    #Ask to pick
    names = [x.name for x in refs]
    var.set(names[-1])

    option = OptionMenu(master, var, *names)
    option.pack()

    # Test 1 picture
    def match(ref):
        global images
        for image in images:
             image.set_reference(ref)

        #For all matches.
        images = sorted(images, key = lambda x:x.distance())
        for i in images[:20]:
            print("Image %s. Distance: %f, %d matches" % (i.name, i.distance(), len(i.matches)))

        #code.interact(local=locals())

        for i in images[:2]:
            i.matchplot()
        plt.show()


    # Press the ok..
    def ok():
        print("value is", var.get())
        idx = names.index(var.get())
        ref = refs[idx]
        match(ref)

    def exit():
        sys.exit()

    button = Button(master, text="OK", command=ok)
    button.pack()

    exit = Button(master, text="Exit", command=exit)
    exit.pack()


    master.mainloop()
