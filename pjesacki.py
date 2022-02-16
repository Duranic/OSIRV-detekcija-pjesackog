import cv2 as cv
import numpy as np
from sklearn.linear_model import RANSACRegressor

def sortBox(box):
    #this function sorts the box points so that the top-left point has index 0

    sortedBox = np.zeros((4, 2), dtype="int32")
    # the top-left point will have the smallest sum,
    # the bottom-right point will have the largest sum
    sum = box.sum(axis=1)
    sortedBox[0] = box[np.argmin(sum)]
    sortedBox[2] = box[np.argmax(sum)]

    # the top-right point will have the smallest difference,
    # the bottom-left will have the largest difference
    diff = np.diff(box, axis=1)
    sortedBox[1] = box[np.argmin(diff)]
    sortedBox[3] = box[np.argmax(diff)]
    return sortedBox


def getLine(points, y, predict, img):
    # this function returns a np.array containing points that
    # are on the predicted line within a certain range (+-15%)

    line=[]
    for i in range(0,len(points)):
        # discard the points that are outside the range
        if predict[i] < y[i]*0.85 or predict[i] > y[i]*1.15:
            # draw the discarded points on the image in red
            cv.circle(img, (points[i]), radius=10, color=(0, 0, 255), thickness=-1)
            continue
        line.append(points[i])
    line=np.array(line)
    return line



img = cv.imread("img1.jpg")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
threshold, thresh=cv.threshold(gray,0,255,cv.THRESH_BINARY + cv.THRESH_OTSU)

kernel = np.ones((7,7),np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)

cv.imshow("transformed", closing)

contours, hierarchy = cv.findContours(closing, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

leftPoints=[]
rightPoints=[]
contourImg=img.copy()


for c in contours:
    # skip if contour is small
    if  cv.contourArea(c) < 1500:
        continue

    hull = cv.convexHull(c)
    areaRatio = cv.contourArea(hull)/cv.contourArea(c)
    # skip if convex hull has a >30% larger area than the contour
    # meaning the contour is probably noise, draw it in red
    if areaRatio>1.3:
        cv.drawContours(contourImg, [c], -1, (0, 0, 255), 5)
        continue

    rect = cv.minAreaRect(c)
    box = cv.boxPoints(rect)
    # convert all coordinates to int
    box = np.int0(box)

    # sort the contour points so that the 0 point is top left
    sortedBox=sortBox(box)

    leftPoint=sortedBox[0]
    leftPoints.append(leftPoint)
    rightPoint=sortedBox[1]
    rightPoints.append(rightPoint)
    
    cv.circle(contourImg, (leftPoint), radius=10, color=(0, 0, 255), thickness=-1)
    cv.circle(contourImg, (rightPoint), radius=10, color=(0, 0, 255), thickness=-1)
    cv.drawContours(contourImg, [box], -1, (0, 255, 0), 5)
cv.imshow("contours", contourImg)

rightPoints=np.array(rightPoints)
leftPoints=np.array(leftPoints)

model=RANSACRegressor()

# get x and y coordinates of all right contour points as vectors for the model
x=np.reshape(rightPoints[:,0], (-1, 1))
y=np.reshape(rightPoints[:,1], (-1, 1))

model.fit(x, y)
predict=model.predict(x)

rightLine=getLine(rightPoints, y, predict, img)
# keep only first and last points
rightLine=np.append(rightLine[0], rightLine[len(rightLine) - 1:])
rightLine=np.reshape(rightLine,(2,2))


# get x and y coordinates of all left points as vectors
x=np.reshape(leftPoints[:,0], (-1, 1))
y=np.reshape(leftPoints[:,1], (-1, 1))

model.fit(x, y)
predict=model.predict(x)

leftLine=getLine(leftPoints, y, predict, img)
# keep only first and last points (this time in opposite order to keep points in clockwise direction)
leftLine=np.append(leftLine[len(leftLine)-1:], leftLine[0])
leftLine=np.reshape(leftLine,(2,2))

pjesacki=np.append(leftLine, rightLine,0)
cv.drawContours(img, [pjesacki], -1, (0, 255, 0), 5)
cv.imshow("bounding box", img)
cv.waitKey(0)