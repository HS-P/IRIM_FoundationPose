# 기존 GITHUB에서 가져온 FOUNDATION POSE LIBRARY 사용을 위해 이전 작업 공간 받기
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

# ESTIMATOR을 포함판 라이브러리 IMPORT하기
import cv2, numpy as np, pyrealsense2 as rs, imageio, trimesh, logging
import nvdiffrast.torch as dr
from estimater import *  # FoundationPose, ScorePredictor, PoseRefinePredictor, draw_xyz_axis, draw_posed_3d_box


# 전역 변수 선언 및 사용
# MESH FILE : OBJ 파일 경로
# K 값은 get_K를 통해서 얻은 후, 해당 array 사용하기
MESH_FILE = "data/galaxy.obj"
K = np.array([[607.17590332,   0.        , 321.9090271 ],
              [  0.        , 607.16156006, 241.65991211],
              [  0.        ,   0.        ,   1.        ]], dtype=np.float64)

EST_REFINE_ITERS = 5
TRACK_REFINE_ITERS = 2
DEBUG_DIR = "./debug_live"

def get_roi_mask(img_bgr):
    r = cv2.selectROI("select ROI (Enter/Space OK, c=cancel)", img_bgr, False, False)
    cv2.destroyWindow("select ROI (Enter/Space OK, c=cancel)")
    mask = np.zeros(img_bgr.shape[:2], np.uint8)
    x, y, w, h = map(int, r)
    if w > 0 and h > 0:
        mask[y:y+h, x:x+w] = 1
    return mask.astype(bool)

def main():
    set_logging_format(); set_seed(0)
    os.makedirs(f"{DEBUG_DIR}/track_vis", exist_ok=True)
    os.makedirs(f"{DEBUG_DIR}/ob_in_cam", exist_ok=True)

    # Mesh & estimator
    mesh = trimesh.load(MESH_FILE)
    mesh.vertices = mesh.vertices.astype(np.float32)
    if hasattr(mesh, "vertex_normals") and mesh.vertex_normals is not None:
        mesh.vertex_normals = mesh.vertex_normals.astype(np.float32)
    # K = K.astype(np.float32)  # 이미 float32면 생략 가능
    # mesh.vertices = mesh.vertices.astype(np.float32)  # dtype 통일
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices,
                         model_normals=mesh.vertex_normals,
                         mesh=mesh, scorer=scorer, refiner=refiner,
                         debug_dir=DEBUG_DIR, debug=1, glctx=glctx)
    logging.info("estimator ready")

    # RealSense
    pipeline, config = rs.pipeline(), rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    print("depth_scale =", depth_scale)

    pose = None; frame_id = 0; inited = False
    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            cf, df = frames.get_color_frame(), frames.get_depth_frame()
            if not cf or not df:
                continue

            bgr = np.asanyarray(cf.get_data())
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)  # contiguous float32
            depth = np.asanyarray(df.get_data()).astype(np.float32) * depth_scale

            if not inited:
                vis_init = bgr.copy()
                cv2.putText(vis_init, "Draw ROI around TARGET, Enter=OK",
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
                cv2.imshow("select ROI (Enter/Space OK, c=cancel)", vis_init)
                mask = get_roi_mask(vis_init)
                assert mask.any(), "Empty ROI; re-run and draw a box."
                pose = est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask,
                                    iteration=EST_REFINE_ITERS)
                inited = True
            else:
                pose = est.track_one(rgb=rgb, depth=depth, K=K,
                                     iteration=TRACK_REFINE_ITERS)

            center_pose = pose @ np.linalg.inv(to_origin)
            vis_rgb = draw_posed_3d_box(K, img=rgb, ob_in_cam=center_pose, bbox=bbox)
            vis_rgb = draw_xyz_axis(vis_rgb, ob_in_cam=center_pose, scale=0.10,
                                    K=K, thickness=3, transparency=0, is_input_rgb=True)
            cv2.imshow("FoundationPose Live", vis_rgb[:, :, ::-1])  # RGB->BGR
            if cv2.waitKey(1) == 27:  # ESC
                break

            if frame_id % 30 == 0:
                np.savetxt(f"{DEBUG_DIR}/ob_in_cam/{frame_id:06d}.txt", pose.reshape(4,4))
                imageio.imwrite(f"{DEBUG_DIR}/track_vis/{frame_id:06d}.png", vis_rgb.astype(np.uint8))
            frame_id += 1

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
