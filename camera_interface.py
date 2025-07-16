import pyrealsense2 as rs

class CameraInterface:
    def __init__(self):
        self.pipeline = rs.pipeline() # Initialize the camera pipeline
        self.align = rs.align(rs.stream.color) # Align depth to color stream

    def setup_streams(self):
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) # 640x480 resolution, 16-bit Z16 format, 30 FPS
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) # 640x480 resolution, 8-bit BGR format, 30 FPS
        self.pipeline.start(config)

    def get_frames(self):
        try:
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            depth = aligned.get_depth_frame()
            color = aligned.get_color_frame()
            if not depth or not color:
                return None, None
            return color, depth
        except:
            return None, None
        
    def get_intrinsics(self):
        profile = self.pipeline.get_active_profile()
        stream = profile.get_stream(rs.stream.color)
        intr = stream.as_video_stream_profile().get_intrinsics()

        import open3d as o3d
        return o3d.camera.PinholeCameraIntrinsic(
            width=intr.width,
            height=intr.height,
            fx=intr.fx,
            fy=intr.fy,
            cx=intr.ppx,
            cy=intr.ppy
        )

    def stop(self):
        self.pipeline.stop()
