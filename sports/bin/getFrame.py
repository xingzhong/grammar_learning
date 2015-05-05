import numpy as np
import cv2
import argparse


def getFrame(args):
    cap = cv2.VideoCapture(args.input)
    num = 0
    n = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        if args.n != -1 and n > args.n:
            break
        if num % 30 == 0:
            cv2.imwrite(args.o + 'f_%s.png' % num, frame)
            n += 1
        num += 1


def main():
    parser = argparse.ArgumentParser(description='split video to frame')
    parser.add_argument('input', help='video file')
    parser.add_argument(
        '--n', help='an integer for the total # frame', type=int)
    parser.add_argument('--o', help='dest folder')
    args = parser.parse_args()
    getFrame(args)

if __name__ == '__main__':
    main()
