import os
import cv2

print(cv2.getNumThreads())  # 查看当前线程数
os.environ["OMP_NUM_THREADS"] = "4"  # OpenMP 线程数
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # OpenBLAS 线程数
os.environ["MKL_NUM_THREADS"] = "4"  # MKL 线程数
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # macOS Accelerate 框架
os.environ["NUMEXPR_NUM_THREADS"] = "4"  # numexpr 线程数


from mot.track import track
from video_data_loader import EnhancedShipDataLoader


def floder_image_loader(path: str):
    files = os.listdir(path)
    files = sorted(files)
    files = [os.path.join(path, file) for file in files 
             if file.endswith(('jpg', 'jpeg', 'png', 'JPG', "JPEG", "PNG"))]
    for file in files[7:]:
        print('---filename', file)
        yield cv2.imread(file)


def video_loader(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"video: {path} open error")
    
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print('finish')
            break
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
        yield frame


def video_loader_optimized(path: str, fps_target: int):
    cap = cv2.VideoCapture(path)
    fps_src = cap.get(cv2.CAP_PROP_FPS)
    
    # 计算跳帧步长（向下取整）
    step = max(1, int(fps_src // fps_target))
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        # 通过设置属性直接跳帧
        for _ in range(step-1):
            cap.grab()  # 跳过中间帧
        
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
        yield frame
    
    cap.release()


if __name__ == "__main__":
    camera_params = {
        'height': 17.5,
        'focal_length': 0.183,
        'incline_angle': 78,
        'photo_size': (0.01636, 0.03273),
    }
    camera_params = {
        'height': 13.2,
        'focal_length': 0.6896,
        'incline_angle': 78.7,
        'photo_size': (0.3454, 0.4606),
    }

    fps_target = 5

    loader = EnhancedShipDataLoader(
        data_path='/Users/trobr/Downloads/Track_gps.csv',
        video_path='/Users/trobr/Downloads/D03_20240910083145.mp4',
        fps_target=5,
    )
    track(loader, 'result.mp4', camera_params, fps_target)
