import cv2
import cognitive_face as CF
from urllib.request import urlopen
from PIL import Image
import matplotlib.pyplot as plt

cap = cv2.VideoCapture("video.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0

KEY = '196f6913f1ad4394821ce3d8915ec2dd'
CF.Key.set(KEY)

BASE_URL = 'https://westeurope.api.cognitive.microsoft.com/face/v1.0/'
CF.BaseUrl.set(BASE_URL)

images = []
while frame_count < 1500:
    ret, frame = cap.read()
    frame_count+=1
    time = float(frame_count)/fps
    if(frame_count>=1200 and time%(time//time)==0):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imsave('img.jpg', frame)
        result = CF.face.detect('img.jpg')
        result = result[0]['faceRectangle']
        print(result)
        print('time: ', time)
        original_img = Image.fromarray(frame)
        cropped_face = original_img.crop((result['left'], result['top'], result['left']+result['width'], result['top']+result['height']))
        cropped_face = cropped_face.resize((100,100),Image.ANTIALIAS)
        images.append(cropped_face)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

images[0].save('result.gif', save_all=True, append_images=images[1:], loop=0)
