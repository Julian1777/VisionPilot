from beamng_sim.lane_detection import thresholds, color_threshold, combine_threshold, warp, get_histogram, slide_window, draw_lane_lines
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera
import cv2

bng = BeamNGpy('localhost', 64256, home=r'D:\SteamLibrary\steamapps\common\BeamNG.drive')
bng.open()

scenario = Scenario('west_coast_usa', 'lane_detection')
vehicle = Vehicle('ego_vehicle', model='etk800', licence='PYTHON')
scenario.add_vehicle(vehicle, pos=(-717, 101, 118))
scenario.make(bng)

bng.scenario.load(scenario)
bng.scenario.start()

camera = Camera(
    'front_cam',
    bng,
    vehicle,
    pos=(0, 2, 1.5),
    dir=(0, 1, 0),
    resolution=(1280, 720),
    field_of_view_y=90,
    is_render_colours=True,
    is_render_annotations=False,
    is_render_instance=False,
    is_render_depth=False
)
vehicle.attach_sensor('front_cam', camera)

config = {'fov': 90}
vehicle.set_player_mode('free', config)

def lane_detection_on_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    grad_binary = thresholds(rgb_frame)
    s_binary = color_threshold(rgb_frame)
    combined_binary = combine_threshold(s_binary, grad_binary)
    binary_warped, Minv = warp(combined_binary)
    histogram = get_histogram(binary_warped)
    ploty, left_fit, right_fit = slide_window(binary_warped, histogram)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    draw_info = {
        'leftx': left_fitx,
        'rightx': right_fitx,
        'left_fitx': left_fitx,
        'right_fitx': right_fitx,
        'ploty': ploty
    }
    result = draw_lane_lines(frame, binary_warped, Minv, draw_info)
    return result

for _ in range(1000):
    sensors = bng.poll_sensors(vehicle)
    img = sensors['front_cam']['colour']
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    result = lane_detection_on_frame(img_bgr)
    cv2.imshow('Lane Detection', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
bng.close()