# src/material_estimator.py
def estimate_materials(area_m2, num_floors=1, wall_thickness_m=0.2):
    slab_thickness_m = 0.12  
    concrete_volume_m3 = area_m2 * slab_thickness_m * num_floors
    concrete_volume_m3 *= 1.05
    steel_kg_per_m2 = 60
    steel_kg = steel_kg_per_m2 * area_m2 * num_floors
    import math
    perim = 4 * math.sqrt(area_m2)
    wall_height_m = 3 * num_floors
    wall_area = perim * wall_height_m
    brick_area_m2 = 0.075  
    bricks = int(wall_area / brick_area_m2)
    return {
        "concrete_m3": round(concrete_volume_m3, 2),
        "steel_kg": int(steel_kg),
        "bricks_count": bricks
    }

if __name__ == "__main__":
    print(estimate_materials(120, num_floors=1))
