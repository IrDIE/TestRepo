


def convert(x_max, y_max, x_min, y_min, id, shape_name):
    shape_names = { 1 : "hexagon",
                2 : "triangle",
                3 : "circle",
                4 : "rhombus"}
    origin_x = x_min
    origin_y =  y_min
    width = x_max - x_min
    height = y_max - y_min
    if shape_name == -1 :
        shape_name_str = "hexagon"
    else:
        shape_name_str = shape_names[shape_name]
    result = {
        "id": f"{id}",
        "name": f"{shape_name_str}",
        "region": {
            "origin": {
                "x": f"{origin_x}",
                "y": f"{origin_y}"
            },
            "size": {
                "width": f"{width}",
                "height": f"{height}"
            }
        }
    }
    return result



