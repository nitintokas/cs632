import cv2
cam = cv2.VideoCapture(0)

cam.set(3,150);
cam.set(4,150);

import os
cwd = "/Applications/XAMPP/xamppfiles/htdocs/projects/deep_learning"


def generate_screenshots(folder):
    test_path = cwd+"/img/test/"+folder
    train_path = cwd+"/img/train/"+folder
    for x in range(1, 51):
        s, im_old = cam.read() # captures image
        im = cv2.resize(im_old, (64, 64)) 
        img_name = cwd+"/img/"+str(x)+".jpg"
        
        if x%5==0:
            cv2.imwrite(os.path.join(train_path , str(x)+'.jpg'), im)
        else:
            cv2.imwrite(os.path.join(test_path , str(x)+'.jpg'), im)
        #print(os.path.join(test_path , str(x)+'.jpg'))


generate_screenshots("img2")

cv2.destroyAllWindows()
cv2.VideoCapture(0).release()
cam.release()

