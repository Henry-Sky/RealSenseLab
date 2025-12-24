

def main():
    import pyrealsense2 as rs
    ctx = rs.context()
    dev = ctx.query_devices()[0]  # 第一台 D455
    print('USB type:', dev.get_info(rs.camera_info.usb_type_descriptor))

if __name__ == '__main__':
    main()