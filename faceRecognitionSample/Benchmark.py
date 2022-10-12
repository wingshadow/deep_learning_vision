import timeit

# 这是一个非常简单的基准测试，可以让您了解人脸识别的每一步在您的系统上运行的速度。请注意，在较大的图像大小下，人脸检测变得非常缓慢
TEST_IMAGES = [
    "test_img/obama-240p.jpg",
    "test_img/obama-480p.jpg",
    "test_img/obama-720p.jpg",
    "test_img/obama-1080p.jpg"
]


# 测试函数
def run_test(setup, test, iterations_per_test=2, tests_to_run=3):
    """
    :param setup: 数据加载函数
    :param test: 数据测试函数
    :param iterations_per_test: 测试次数
    :param tests_to_run: 每轮测试调用函数多少次
    :return: execution_time单次函数推理时间，fps每秒处理次数
    """
    fastest_execution = min(timeit.Timer(test, setup=setup).repeat(tests_to_run, iterations_per_test))
    execution_time = fastest_execution / iterations_per_test
    fps = 1.0 / execution_time
    return execution_time, fps


# 以下设置不同的测试函数代码
# setup开头的是数据加载代码，test开头的是函数测试代码
setup_locate_faces = """
import face_recognition

image = face_recognition.load_image_file("{}")
"""

test_locate_faces = """
face_locations = face_recognition.face_locations(image)
"""

setup_face_landmarks = """
import face_recognition

image = face_recognition.load_image_file("{}")
face_locations = face_recognition.face_locations(image)
"""

test_face_landmarks = """
landmarks = face_recognition.face_landmarks(image, face_locations=face_locations)[0]
"""

setup_encode_face = """
import face_recognition

image = face_recognition.load_image_file("{}")
face_locations = face_recognition.face_locations(image)
"""

test_encode_face = """
encoding = face_recognition.face_encodings(image, known_face_locations=face_locations)[0]
"""

setup_end_to_end = """
import face_recognition

image = face_recognition.load_image_file("{}")
"""

test_end_to_end = """
encoding = face_recognition.face_encodings(image)[0]
"""

# 所有的基准测试都只使用一个CPU核心
print("Benchmarks (Note: All benchmarks are only using a single CPU core)")
print()

for image in TEST_IMAGES:
    size = image.split("-")[1].split(".")[0]
    print("Timings at {}:".format(size))

    # 测试人脸检测
    print(" - Face locations: {:.4f}s ({:.2f} fps)".format(
        *run_test(setup_locate_faces.format(image), test_locate_faces)))
    print(" - Face landmarks: {:.4f}s ({:.2f} fps)".format(
        *run_test(setup_face_landmarks.format(image), test_face_landmarks)))
    print(" - Encode face (inc. landmarks): {:.4f}s ({:.2f} fps)".format(
        *run_test(setup_encode_face.format(image), test_encode_face)))
    print(" - End-to-end: {:.4f}s ({:.2f} fps)".format(*run_test(setup_end_to_end.format(image), test_end_to_end)))
    print()