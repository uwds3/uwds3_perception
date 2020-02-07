#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import cv2
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Record RGB snapshots for machine learning")
    parser.add_argument('label', type=str, help='The label used to name the data directory')
    parser.add_argument("-d", "--data_dir", type=str, default="/tmp/snapshots/", help="The root data directory (default '/tmp/snapshots/')")
    args = parser.parse_args()
    snapshot_directory = args.data_dir + args.label + "/"

    try:
        os.makedirs(snapshot_directory)
    except OSError as e:
        if not os.path.isdir(snapshot_directory):
            raise RuntimeError("{}".format(e))
    snapshot_index = 0

    capture = cv2.VideoCapture(0)
    while True:
        ok, frame = capture.read()
        if ok:
            k = cv2.waitKey(1) & 0xFF
            if k == 32:
                print("Save image "+str(snapshot_index)+".jpg !")
                cv2.imwrite(snapshot_directory+str(snapshot_index)+".jpg", frame)
                snapshot_index += 1
                cv2.imshow("Snapshot recorder", (255-frame))
            else:
                cv2.imshow("Snapshot recorder", frame)
    capture.release()
