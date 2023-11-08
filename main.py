import cv2
import mediapipe as mp
import numpy as np

from google.protobuf.json_format import MessageToDict 

cap=cv2.VideoCapture(0)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

with mp_hands.Hands(min_detection_confidence=0.65,min_tracking_confidence=0.35) as hands: # for lower confidence, it kept stating that my waves sweatshirt was a hand
    while True:
        success, frame=cap.read()
        
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)#opencv processes the image in RGB format, read function interprets in BGR format, hence i need to convert
        image.flags.writeable = False
        image = cv2.flip(image,1) #mirroring the feed 
        results = hands.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
               
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)
                
               

		 
            if len(results.multi_handedness) == 2: 
                cv2.putText(image, 'Both Hands', (250, 50), 
						cv2.FONT_HERSHEY_COMPLEX, 0.9, 
						(0, 0, 255), 2)
            else: 
                  for i in results.multi_handedness: 
                    label = MessageToDict(i)[ 
					'classification'][0]['label'] 
                    
                    if label == 'Left':
                        cv2.putText(image, label+' Hand', (20, 50), 
								cv2.FONT_HERSHEY_COMPLEX, 0.9, 
								(0, 0, 255), 2) 
                        
                    if label == 'Right': 
                        cv2.putText(image, label+' Hand', (460, 50), 
								cv2.FONT_HERSHEY_COMPLEX, 
								0.9, (0, 0 , 255), 2) 

        
        cv2.imshow("Hand Detection",image)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break

# I first tried labeling left or right on the hand itself(on the wrist) but it just wouldn't work for right hand
# This is what i tried:

# def figureouthand(index, hand, results):
#     output = None
#     for idx, classification in enumerate(results.multi_handedness):
#         if classification.classification[0].index == index:
                         
#             label = classification.classification[0].label
         
#             coords = tuple(np.multiply(
#                 np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),
#             [640,480]).astype(int))
            
#             output = label, coords
            
#     return output


## and this
# if figureouthand(num, hand, results):
#     label, coord = figureouthand(num, hand, results)
#     cv2.putText(frame, label, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
#     print(label)
