import pyrealsense2 as rs

class LabStream:
    def __init__(self):
        self.context = rs.context()
        self.pipeline = rs.pipeline()
        pass

    def streamStart(self):
        pass

    def streamStop(self):
        pass

    def streamEnd(self):
        pass