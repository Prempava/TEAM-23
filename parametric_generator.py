# src/parametric_generator.py
import json
from math import sqrt
TEMPLATES = {
    "single_storey_house": {"rooms": ["living","kitchen","bedroom","bath"], "min_area_m2": 50},
    "duplex": {"rooms": ["living","kitchen","bedroom","bedroom","bath"], "min_area_m2": 120},
    "warehouse": {"rooms": ["open_space","office","toilet"], "min_area_m2": 300},
    "shop_small": {"rooms": ["retail","storage","toilet"], "min_area_m2": 40}
}

def generate_floorplan(template_name, area_m2):
    tpl = TEMPLATES.get(template_name)
    if not tpl:
        raise ValueError("Unknown template")
    rooms = tpl["rooms"]
    weights = [1]*len(rooms)
    total_w = sum(weights)
    room_areas = [round(area_m2 * (w/total_w),2) for w in weights]
    side = sqrt(area_m2)
    floorplan = {
        "template": template_name,
        "total_area_m2": area_m2,
        "approx_side_m": round(side,2),
        "rooms": []
    }
    for r, a in zip(rooms, room_areas):
        width = round(sqrt(a/1.5),2)
        height = round(a/width,2)
        floorplan["rooms"].append({
            "name": r,
            "area_m2": a,
            "approx_width_m": width,
            "approx_height_m": height
        })
    return floorplan

if __name__ == "__main__":
    print(json.dumps(generate_floorplan("single_storey_house", 120), indent=2))
