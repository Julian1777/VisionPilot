import carla


client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.load_world('Town01')

carla_map = world.get_map()
spawn_points = carla_map.get_spawn_points()


start = spawn_points[0].location
end = spawn_points[10].location

start_waypoint = carla_map.get_waypoint(start)
end_waypoint = carla_map.get_waypoint(end)

route_waypoints = [start_waypoint]
current_waypoint = start_waypoint

max_steps = 1000
steps = 0
while current_waypoint.transform.location.distance(end) > 2.0 and steps < max_steps:
    next_waypoints = current_waypoint.next(2.0)
    if not next_waypoints:
        break
    current_waypoint = next_waypoints[0]
    route_waypoints.append(current_waypoint)
    steps += 1


for waypoint in route_waypoints:
    world.debug.draw_point(waypoint.transform.location, size=0.2, color=carla.Color(0,255,0), life_time=60.0)

for waypoint in route_waypoints:
    print(waypoint.transform.location)
    world.debug.draw_point(waypoint.transform.location, size=0.2, color=carla.Color(0,255,0), life_time=60.0)
