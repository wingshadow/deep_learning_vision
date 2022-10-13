from __future__ import print_function
import os
import re
import scipy.misc
import warnings
import face_recognition.api as face_recognition
import sys
import numpy as np


def scanKnownPeople(knownPeopleImage):
    known_face_encodings = []

    img = face_recognition.load_image_file(knownPeopleImage)
    encodings = face_recognition.face_encodings(img)
    if len(encodings) == 1:
        known_face_encodings.append(encodings[0])
    return known_face_encodings

# 余弦定理
def sim(source_encoding,target_encoding):
    source_np = np.array(source_encoding)
    target_np = np.array(target_encoding)
    dist = np.linalg.norm(source_np - target_np)  # 二范数
    sim = 1.0 / (1.0 + dist)  #
    return sim

def checkImage(image_to_check, known_face_encodings):
    unknown_image = face_recognition.load_image_file(image_to_check)
    # Scale down image if it's giant so things run a little faster
    if unknown_image.shape[1] > 1600:
        scale_factor = 1600.0 / unknown_image.shape[1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            unknown_image = scipy.misc.imresize(unknown_image, scale_factor)

    unknown_encodings = face_recognition.face_encodings(unknown_image)
    if len(unknown_encodings) == 1:
        for unknown_encoding in unknown_encodings:
            result = face_recognition.compare_faces(known_face_encodings, unknown_encoding)
            distance = face_recognition.face_distance(known_face_encodings, unknown_encoding)
            sim_result = sim(known_face_encodings,unknown_encoding)
            print(sim_result)
            # print(distance[0])
            # print("True") if True in result else print("False ")

        return distance[0], result[0]
    else:
        return "0", "Many Faces or No Faces"


def image_files_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]


def main(known_people_folder, image_to_check):
    known_face_encodings = scanKnownPeople(known_people_folder)
    distance, result = checkImage(image_to_check, known_face_encodings)
    return distance, result


if __name__ == "__main__":
    distance, result = main(sys.argv[1], sys.argv[2])
    print(distance, result)
