import os
from config import PROJECT_PATH
from models import StackedHourglassNetwork
from inference import predict
from visualization import draw_keypoints_on_image, draw_skeleton_on_image

if __name__ == "__main__":
    model = StackedHourglassNetwork(...)  # base.ipynb에서 쓰던 설정 그대로
    # WEIGHTS_PATH 로드 부분은 inference.predict 내부 or 여기 자유롭게
    test_image = os.path.join(PROJECT_PATH, 'data', 'test.png')

    image, keypoints = predict(model, test_image)
    draw_keypoints_on_image(image, keypoints)
    draw_skeleton_on_image(image, keypoints)
    