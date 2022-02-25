import cv2
import mediapipe as mp
import time
# mediapipe website :https://mediapipe.dev/
# use mediapipe to do the hand detection
# read the camera, for my case, 1 only can be used which is default webcam
# landmarks is the coodinates
cam = cv2.VideoCapture(1)
# to tell the mediapipe that we are using hands solution from mediapipe library
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)  # use default value
mpDraw = mp.solutions.drawing_utils
# call this to change design of hand landmarks and lines
handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=3)
handConnectionStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=5)
# hand detection is using rgb value but opencv is using bgr value default, need to convert it to rgb
# ctrl + click the Hands() to change the parameter

#find fps
previousTime = 0
currentTime = 0

while True:
    ret, img = cam.read()
    # if can read the camera, then show the pic
    if ret:
        # convert bgr to rgb so that we can use in hand detection
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(imgRGB)
        # it will print the 21 coordinates for hand if detected in terminal
        # print(result.multi_hand_landmarks)
        #now call the window
        imgHeight = img.shape[0]
        imgWidth = img.shape[1]
        if (result.multi_hand_landmarks):
            for handLms in result.multi_hand_landmarks:
                # mpDraw will draw the landmarks detected
                # 1st parameter is place to display, 2nd parameter is the coordinates for landmarks, 3rd parameter is to connect all the coordinates
                # 4rd parameter is for points design, 5rd parameter is line design
                #this line is to draw the 21 coordinates and line
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConnectionStyle)
                #now we want to this coordinate is belongs to which one de, so use other for loop
                #will have 21 as we have 21 coordinate
                #the coordinate is RATIO. if the window is 100 x 100, then x coordinate is 0.5 x 100 = 50
                #if want to find the actual coordinate, need to multiply by the window height and weight
                for i, landmark in enumerate(handLms.landmark):
                    #coordinates change to int
                    xPos = int(landmark.x * imgWidth)
                    yPos = int(landmark.y * imgHeight)
                    #print out the i-th coordinate to indicate the ordering of coordinates
                    cv2.putText(img, str(i), (xPos-25, yPos+5), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0,0,255), 2)
                    #to enlarge the thumb
                    # 4th position is thumb
                    if i == 4:
                        #it will draw the circl on the thumb
                        cv2.circle(img, (xPos, yPos), 10, (255,26,172), cv2.FILLED)
                    print(i, xPos, yPos)

        currentTime = time.time()
        fps = 1/(currentTime-previousTime)
        previousTime = currentTime
        cv2.putText(img, f"FPS : {int(fps)}", (30,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 1)
        cv2.imshow('img', img)

    if cv2.waitKey(1) == ord('q'):
        break
