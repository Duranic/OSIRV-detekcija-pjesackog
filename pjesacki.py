import cv2 as cv
from math import sqrt, pi, sin, cos, atan
import numpy as np
from sklearn.linear_model import RANSACRegressor

class LengthError(ValueError):
    pass


def calculateDirection(point, theta1, theta2):
    # if the angle is negative, add pi
    if theta1<0:
        theta1=theta1 + pi
    if theta2<0:
        theta2=theta2 + pi

    thetaAvg=(theta1+theta2) * 0.5
    # return a point moved by 100 in the direction of the average angle
    return point[0] - 100*cos(thetaAvg), point[1] - 100*sin(thetaAvg)


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


def adjustBrightnessContrast(img, contrast, brightness):
    new_image = np.clip(contrast*img + brightness, 0, 255)
    return np.array(new_image, np.uint8)


def getDistance(point1, point2):
    distance = sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
    return distance


def getLine(points, y, predict):
    # this function returns a np.array containing points that
    # are on the predicted line within a certain range (+-15%)

    line=[]
    for i in range(0,len(points)):
        # discard the points that are outside the range
        if predict[i] < y[i]*0.85 or predict[i] > y[i]*1.15:
            continue
        line.append(points[i])
    if len(line)<3:
        raise LengthError("Not enough contours in line")
    line=np.array(line)
    return line



img = cv.imread("img13.jpg")
if img is None:
    print ('File cannot be opened')
    exit()
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = adjustBrightnessContrast(gray, 1.3, -100)

threshold, thresh=cv.threshold(gray,0,255,cv.THRESH_BINARY + cv.THRESH_OTSU)


kernel = np.ones((7,7),np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)

cv.imshow("transformed", closing)

leftPoints=[]
rightPoints=[]
contourImg=img.copy()

contours, _ = cv.findContours(closing, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
for c in contours:
    # skip if contour is small
    if  cv.contourArea(c) < 2000:
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
    # take the points that are further apart as left and right points 
    if getDistance(sortedBox[0], sortedBox[1])>getDistance(sortedBox[1], sortedBox[2]):
        leftPoint=sortedBox[0]
        rightPoint=sortedBox[1]
    else:
        leftPoint=sortedBox[1]
        rightPoint=sortedBox[2]
    
    leftPoints.append(leftPoint)
    rightPoints.append(rightPoint)

    cv.circle(contourImg, (leftPoint), radius=10, color=(0, 0, 255), thickness=-1)
    cv.circle(contourImg, (rightPoint), radius=10, color=(0, 0, 255), thickness=-1)
    cv.drawContours(contourImg, [box], -1, (0, 255, 0), 5)
cv.imshow("contours", contourImg)


rightPoints=np.array(rightPoints)
leftPoints=np.array(leftPoints)

model=RANSACRegressor()

try:
    # get x and y coordinates of all right contour points as vectors for the model
    x=np.reshape(rightPoints[:,0], (-1, 1))
    y=np.reshape(rightPoints[:,1], (-1, 1))
    model.fit(x, y)
    predict=model.predict(x)
    theta1=atan(model.estimator_.coef_)

    rightLine=getLine(rightPoints, y, predict)
except LengthError as ex:
    print(ex.args[0])
    cv.waitKey(0)
    exit()
except:
    print("Not enough contours found")
    cv.waitKey(0)
    exit()


try:
    # get x and y coordinates of all left contour points as vectors for the model
    x=np.reshape(leftPoints[:,0], (-1, 1))
    y=np.reshape(leftPoints[:,1], (-1, 1))
    model.fit(x, y)
    predict=model.predict(x)
    theta2=atan(model.estimator_.coef_)

    leftLine=getLine(leftPoints, y, predict)
except LengthError as ex:
    print(ex.args[0])
    cv.waitKey(0)
    exit()
except:
    print("Not enough contours found")
    cv.waitKey(0)
    exit()


lines=np.append(leftLine, rightLine, axis=0)
pjesacki=cv.convexHull(lines)
cv.drawContours(img, [pjesacki], -1, (0, 255, 0), 5)

# calculate contour midpoint as average of all points
midpoint=lines.mean(axis=0)
midpoint = list(map(int, midpoint))
# calculate pedestrian direction related to the midpoint
point=calculateDirection(midpoint, theta1, theta2)
point=list(map(int, point))
# use the 2 points to draw an arrow
cv.arrowedLine(img, midpoint, point, (0, 255, 255), 10, tipLength=0.3)

cv.imshow("pjesacki", img)
cv.waitKey(0)