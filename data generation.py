import csv
import random
import math
from pathlib import Path
OUT = Path(__file__).resolve().parents[1] / "data" / "synthetic_data.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)
shapes = ["rectangular", "irregular", "L-shape"]
soil_types = ["clay", "sand", "loam", "laterite"]
templates = ["single_storey_house", "duplex", "warehouse", "shop_small"]
def choose_template(area, slope, bearing, req):
    if req == "warehouse" or area > 1200:
        return "warehouse"
    if bearing < 100 and area < 200:
        return "single_storey_house"
    if req == "house" and area > 200 and bearing >= 100:
        return "duplex"
    if req == "shop":
        return "shop_small"
    return random.choice(templates)
with open(OUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["plot_length_m","plot_width_m","area_m2","plot_shape","orientation","slope_percent","soil_type","bearing_capacity_kpa","project_requirement","num_floors","budget_usd","label_template"])
    for _ in range(2000):
        l = round(random.uniform(6,80),2)
        w = round(random.uniform(6,60),2)
        area = round(l*w,2)
        shape = random.choice(shapes)
        orientation = random.choice(["N","S","E","W"])
        slope = round(random.uniform(0,12),2)
        soil = random.choice(soil_types)
        bearing_map = {"clay":150,"sand":120,"loam":180,"laterite":200}
        bearing = bearing_map[soil] + random.randint(-40,40)
        req = random.choice(["house","shop","warehouse","house"])
        floors = random.choice([1,1,2,3])
        budget = int(area * random.uniform(50,200))
        label = choose_template(area, slope, bearing, req)
        writer.writerow([l,w,area,shape,orientation,slope,soil,bearing,req,floors,budget,label])
print("Synthetic data written to:", OUT)
