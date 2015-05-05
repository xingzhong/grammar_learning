import numpy as np
import cv2
from template import template, POI

FONT = cv2.FONT_HERSHEY_SIMPLEX


def dm(theta, sx, sy, dx, dy):
    return np.float32(
        [[sx * np.cos(theta), -np.sin(theta), dx],
         [np.sin(theta), sy * np.cos(theta), dy],
         [0, 0, 1]])


def main():
    cap = cv2.VideoCapture('sep/0_0.avi')
    fn = 0
    ret, iframe = cap.read()
    H, W, _ = iframe.shape
    tpl = template()
    cRot = np.zeros((H, W, 3))
    theta = 0

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        fn += 1
        cRot = cv2.warpPerspective(
            tpl, dm(theta, 0.95, 1.2, -200, 100), (W, H))

        cv2.putText(frame, "#f %d" %
                    fn, (10, 30), FONT, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        cv2.imshow('tpl', cRot)
        if fn == 1:
            cv2.imwrite("test0.png", frame)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    main()
