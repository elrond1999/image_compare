#!/bin/env python
#
#  Matches the Top and Bottom half of a picture to a database of pictures.
#

import numpy as np
import cv2
import os
import sys
import matplotlib.pyplot as plt
from natsort import natsorted
#import code
from tkinter import *

X = 1
Y = 0

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

    #Return if a kp is in the upper or lower part of the image
    def up_or_down(self, kpidx):
        y_max = self.img.shape[Y]
        y_half = y_max / 2
        kp = self.kp[kpidx]
        y = kp.pt[1]
        if y < y_half:
            return "up"
        else:
            return "down"

    #Split matches in upper and lower part of image
    def split_matches(self):
        up = []
        down = []        
        for m in self.match():
            if self.up_or_down(m.queryIdx) == "up" and self.ref.up_or_down(m.trainIdx) == "up":
                up.append (m)
            elif self.up_or_down(m.queryIdx) == "down" and self.ref.up_or_down(m.trainIdx) == "down":
                down.append (m)
            else:
                #Match is not correlated.
                pass

        #Store them sorted
        self.up_matches   = sorted(up,   key = lambda x:x.distance)
        self.down_matches = sorted(down, key = lambda x:x.distance)

    def upper_matches(self):
        if self.up_matches == None:
            self.split_matches()
        return self.up_matches

    def lower_matches(self):
        if self.down_matches == None:
            self.split_matches()
        return self.down_matches

    def distance(self, updown = None):
        N = 10
        if updown == "up":
            matches = self.upper_matches()[:N]
        elif updown == "down":
            matches = self.lower_matches()[:N]
        else:
            matches = self.match()[:N]

        if len(matches) == 0:
            return 99999

        avg = sum(m.distance for m in matches)/len(matches)
        avg_scaled = avg*(N/len(matches))
        return avg_scaled

    def print(self):
        print("Image %s: Shape: %dx %dy" % (self.name, self.img.shape[X], self.img.shape[Y]))
        print("    Total Distance: %f, %d matches" % (self.distance(), len(self.matches)))
        print("    Upper Distance: %f, %d matches" % (self.distance("up"), len(self.up_matches)))
        print("    Lower Distance: %f, %d matches" % (self.distance("down"), len(self.down_matches)))

    def matchplot(self, updown = None):
        N = 20
        if updown == "up":
            matches = self.upper_matches()[:N]
        elif updown == "down":
            matches = self.lower_matches()[:N]
        else:
            matches = self.match()[:N]

        fig = plt.figure()
        a = fig.add_subplot(1,1,1)
        a.set_title("%s distance %0.2f, up %0.2f, down %0.2f" % (self.name, self.distance(), self.distance("up"), self.distance("down")))
        img = cv2.drawMatches(self.img    ,self.kp,
                              self.ref.img,self.ref.kp, matches, None, flags=2)
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
    refs = natsorted(refs, key = lambda x:x.name)
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
        print("\n\nBest match for Total distance")
        images = sorted(images, key = lambda x:x.distance())
        for i in images[:10]:            
            i.print()

        #for i in images[:1]:
                #i.matchplot()
                #i.matchplot("up")
                #i.matchplot("down")

        for p in ["up", "down"]:
            print("\n\nBest match for %s distance" % (p))
            images = sorted(images, key = lambda x:x.distance(p))
            for i in images[:10]:            
                i.print()

            for i in images[:1]:
                i.matchplot(p)

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
