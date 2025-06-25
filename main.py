import cv2
import numpy as np

class ROI():
    
    def __init__(self):
        self.cap = cv2.VideoCapture("trafik.mp4")

    def video(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            cv2.imshow("video",frame)
            key = cv2.waitKey(30)

            if key == ord("m"):
                roi = cv2.selectROI("Roi", frame)
                self.img = frame
                cv2.destroyAllWindows()
                break
            
            if key ==ord("q"):
                cv2.destroyAllWindows()
                break
        self.cap.release()
        self.template(roi)
        
    
    def template(self, roi):
        x, y, w, h = roi
        template = self.img[y:y+h, x:x+w]
        self.templateMatch(template)

    def templateMatch(self, template):
        self.cap = cv2.VideoCapture("trafik.mp4")
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        w,h = template_gray.shape[::-1]

        while True:
            ret, frame = self.cap.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            res = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_SQDIFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            cv2.imshow("video", frame)
            cv2.waitKey(30)

            if min_val < 0.005:
                bottom_right = (min_loc[0] + w, min_loc[1] + h)
                cv2.rectangle(frame, min_loc, bottom_right, (0,255,0), 2)
                cv2.imshow("video", frame)
                key = cv2.waitKey(0)

                if key == ord("q"):
                    break
            
            if not ret:
                break
        
        self.cap.release()


if __name__ == "__main__":
    baslat = ROI()
    baslat.video()




