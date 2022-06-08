import cv2
import time
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
transform = midas_transforms.dpt_transform

cam = cv2.VideoCapture(0)
while cam.isOpened():
    _, frame = cam.read()
    # cv.imshow('video', frame)
    start = time.time()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(frame).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth_map = prediction.cpu().numpy()
    depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    end = time.time()
    totalTime = end - start
    fps = 1 / totalTime
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    depth_map = (depth_map*255).astype(np.uint8)
    depth_map = cv2.applyColorMap(depth_map , cv2.COLORMAP_MAGMA)
    cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
    cv2.imshow('Image', frame)
    cv2.imshow('Depth Map', depth_map)
        # h, w = frame.shape[:2]
        # center_point = (int(w//2), int(h//2))
        # depth_face = depth_map[int(center_point[1]), int(center_point[0])]
        # depth = -1.7 * depth_face + 2
        # print("Depth in cm: " + str(round(depth,2)*100))
        # depth_calc(frame)
        # pil_image = PIL.Image.fromarray(frame)
        # caption = generate_caption(pil_image, args, model)
        # read_caption(caption)
    keypress = cv2.waitKey(1000)
    # if keypress & 0xFF != ord('q'):
    if keypress & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()

def depth_calc(img):
    print(f'img.shape = {img.shape}')
    input_batch = transform(img).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth_map = prediction.cpu().numpy()
    depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    depth_map = (depth_map*255).astype(np.uint8)
    h, w = img.shape[:2]
    center_point = (int(w//2), int(h//2))
    depth_face = depth_map[int(center_point[1]), int(center_point[0])]
    depth = -1.7 * depth_face + 2
    print("Depth in cm: " + str(round(depth,2)*100))
