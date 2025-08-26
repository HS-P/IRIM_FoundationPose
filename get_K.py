import pyrealsense2 as rs, cv2, numpy as np
pipeline, config = rs.pipeline(), rs.config()
for s,f in [(rs.stream.color,rs.format.bgr8),(rs.stream.depth,rs.format.z16)]:
    config.enable_stream(s, 640, 480, f, 30)
profile = pipeline.start(config); align = rs.align(rs.stream.color)
scale = profile.get_device().first_depth_sensor().get_depth_scale()
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
K = np.array([[intr.fx,0,intr.ppx],[0,intr.fy,intr.ppy],[0,0,1]],float); print("K=\n",K," depth_scale=",scale)
try:
    while True:
        f = align.process(pipeline.wait_for_frames())
        c = np.asanyarray(f.get_color_frame().get_data())
        d = np.asanyarray(f.get_depth_frame().get_data())*scale  # meters
        cv2.imshow("color", c); cv2.imshow("depth(m)", (d/5.0).astype("float32"))
        if cv2.waitKey(1)==27: break
finally:
    pipeline.stop(); cv2.destroyAllWindows()
