from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
from cv2 import cv2
import math
import glob
import pickle

class dataBundle:
    def __init__(self):
        self.images = []
    
    def addImage(self, image):
        self.images.append(image)

    def serialize(self):
        pickle.dump(self, open("bundle.p", "wb"))
        return pickle.dumps(self)

    def deserialize(self):
        return pickle.load(open('bundle.p', 'rb'))

    def getData(self):
        return self.images


class imageProcessor:
    def __init__(self, filename):
        self.filename=filename
        self.img=cv2.imread(self.filename)
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.thresh = cv2.threshold(self.gray,127,255,1)[1]
        self.contours,h = cv2.findContours(self.thresh,1,2)
    #warning you are now entering a bruh moment zone
    def serialize(self):
        pickle.dump(self, open("out.p", "wb"))
        return pickle.dumps(self)
    #warning you are now exiting a bruh moment zone
    def readPickle(self):
        return pickle.load(open('out.p', 'rb'))

    def show(self):
        cv2.imshow('img',self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def detail(self):
        details = []
        
        #Prints the results to screen
        #print("Number Of Objects Detected: "+str(len(self.contours)))
        for cnt in self.contours:
            #Approx is the amount of edges in the shape (circles just have a lot of faces))
            approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
            #approx is used for specific functions, cnt is used for the drawing.

            #print(len(approx))
            centre = self.getCentre(cnt)
            #print("Corners: "+str(self.getCorners(approx)))
            details.append(self.getCorners(approx))
            #print("Angle Of Shape: "+str(self.getOrientation(cnt, self.img)))
            
            if len(self.getEdgeLengths(approx)) < 9:
                #print("Lengths Of Edges: "+str(self.getEdgeLengths(approx)))
                details.append(self.getEdgeLengths(approx))
            else:
                Perimeter = 0
                for i in self.getEdgeLengths(approx):
                    Perimeter += i
                #print("Lengths Of Edges: " + str(Perimeter))
                details.append(Perimeter)
            if len(self.getEdgeLengths(approx)) < 9:
                #print("Sum of Angles: "+str(self.sumOfAngles(approx)))
                details.append(self.sumOfAngles(approx))
            if len(self.DistanceFromCentre(approx)) < 9:
                details.append(self.DistanceFromCentre(approx))
                #print("Distance From Center Of Object: "+str(self.DistanceFromCentre(approx)))
            else:
                #print("Distance From Center Of Object: "+str(self.DistanceFromCentre(approx)[0]))
                details.append(self.DistanceFromCentre(approx))
            

            #Draw the object on the image. show() should be called after detail()
            cv2.drawContours(self.img,[cnt],0,-1)
            #cv2.circle(self.img, (centre[0], centre[1]), 3, (255, 0, 0), -1)
            details = np.array(details, dtype = object)
            return details
    def save(self, name):
        cv2.imwrite(name, self.img)

    #Both Functions get the relative angle, both work to varying levels of success
    def getAngleDP(self, p1, p2):
        len1 = math.sqrt(p1[0]**2 + p1[1]**2)
        len2 = math.sqrt(p1[0]**2 + p1[1]**2)
    #Dot product is the point between the two contours being compared
        dot = p1[0] * p2[0] + p1[1] + p2[1]

        angle = dot / (len1 * len2)

        return math.atan(angle) * 180 / math.pi

    def getAngleAtan(self, p1, p2):
        #atan2 gets the angle of the contour, but is extremely relative to the orientation of the object
        angle = math.atan2((p1[1] - p2[1]) , (p1[0] - p2[1])) * 180 / math.pi
        return angle
        
    #Gets the angles in the shape, isn't working too well right now.
    def sumOfAngles(self, contours):

        conRect = []
        for i in range(len(contours)):
            conBB = cv2.minAreaRect(contours[i])
            
            conRect.append(conBB[0])
        AngleSet = 0
        for i in range(len(conRect)):
            
            Length1 = conRect[i-1]
            Length2 = conRect[i]

           # else:
                #Length1 = conRect[i-1]
               # Length2 = conRect[i+1]

            Angle = self.drawAxis(self.img, Length1, Length2, (0, 255, 0), 1)
            if Angle < 1:
                Angle *= -1
            AngleSet +=(Angle * 180 / math.pi)

        if AngleSet > 100 and AngleSet < 250:
            AngleSet = 180
        if AngleSet > 300 and AngleSet < 400:
            AngleSet = 360

        return AngleSet

    def drawAxis(self, img, p_, q_, colour, scale):
        p = list(p_)
        q = list(q_)
        ## [visualization1]
        angle = math.atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
        #angle = self.getAngleDP(p,q)
        hypotenuse = math.sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))

        # Here we lengthen the arrow by a factor of scale
        q[0] = p[0] - scale * hypotenuse * math.cos(angle)
        q[1] = p[1] - scale * hypotenuse * math.sin(angle)
        cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)

        # create the arrow hooks
        p[0] = q[0] + 9 * math.cos(angle + math.pi / 4)
        p[1] = q[1] + 9 * math.sin(angle + math.pi / 4)
        cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)

        p[0] = q[0] + 9 * math.cos(angle - math.pi / 4)
        p[1] = q[1] + 9 * math.sin(angle - math.pi / 4)
        cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)

        return angle
    
    #Gets the overall rotation of the shape, spanning from -180 to 180
    def getOrientation(self, contour, img):
        sz = len(contour)
        data_pts = np.empty((sz, 2), dtype=np.float64)
        for i in range(data_pts.shape[0]):
            data_pts[i,0] = contour[i,0,0]
            data_pts[i,1] = contour[i,0,1]

        # Perform PCA analysis
        mean = np.empty((0))
        mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)

        # Store the center of the object
        cntr = (int(mean[0,0]), int(mean[0,1]))
        ## [pca]

        ## [visualization]
        # Draw the principal components
        cv2.circle(img, cntr, 3, (255, 0, 255), 2)
        p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
        p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
        self.drawAxis(img, cntr, p1, (0, 255, 0), 1)
        self.drawAxis(img, cntr, p2, (255, 255, 0), 5)

        angle = math.atan2(eigenvectors[0,1], eigenvectors[0,0])*180/math.pi # orientation in degrees
        ## [visualization]

        return angle
        
    #Marks the vertices of the shape, can be used to get the number of them as well
    def getCorners(self, contours):
        corners = 0
        for i in range(len(contours)):
            rect = cv2.minAreaRect(contours[i])
            center = rect[0]
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(self.img,[box],0,(255,0,0),2)
            corners += 1
            
        return corners
    #Gets the centre of an entire object based on contour        
    def getCentre(self, contour):
        M = cv2.moments(contour)    
        try:
            cX = int(M["m10"] / M["m00"]) #TODO: Can get a zero division error  make try and catch
            cY = int(M["m01"] / M["m00"])
        except:
            cX = 0
            cY = 0
        return cX, cY
        
    #Gets the distance of each corner(I think) to the centre of the shape
    def DistanceFromCentre(self, contours):
        conRect = []
        
        centrePoint = self.getCentre(contours)
        
        for i in range(len(contours)):
            conBB = cv2.minAreaRect(contours[i])
            
            conRect.append(conBB[0])
        disFromCentre = []
        for i in range(len(conRect)):
            Corner = conRect[i-1]

            Hypot = np.sqrt((Corner[0] - centrePoint[0])**2 + (Corner[1] - centrePoint[1])**2)
            disFromCentre.append(Hypot)
            
        return disFromCentre
    #Gets the length of each edge and compiles them into a list
    def getEdgeLengths(self, contours):
        conRect = []
        for i in range(len(contours)):
            conBB = cv2.minAreaRect(contours[i])
            
            conRect.append(conBB[0])
        conLength = []
        for i in range(len(conRect)):
            Length1 = conRect[i-1]
            Length2= conRect[i]


            Hypot = np.sqrt((Length1[0] - Length2[0])**2 + (Length1[1] - Length2[1])**2)
            conLength.append(Hypot)
            
        return conLength
    #Checks if Edge lengths are equal, doesn't work well with circles
    def ifEdgeLengthsEqual(self, contours):

        Edges = (self.getEdgeLengths(contours))
            
        if len(Edges) == 4:
            if len(np.unique(Edges)) == 2:
                return "Rectangular"
            if len(np.unique(Edges)) == 1:
                return "Square"
            if len(np.unique(Edges)) > 2:
             return "Quadlilateral"
            
        if len(Edges) == 3:
            if len(np.unique(Edges)) == 1:
                return "EquilateralTriangle"
            if len(np.unique(Edges)) == 2:
                return "IsoscelesTriangle"
            if len(np.unique(Edges)) == 3:
                return "Triangle"

