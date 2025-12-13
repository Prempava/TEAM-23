def estimate_cost(materials):
    rates = {
        "concrete": 120,   # per m³
        "steel": 1.2,      # per kg
        "brick": 0.6       # per brick
    }

    cost = (
        materials["concrete_m3"] * rates["concrete"] +
        materials["steel_kg"] * rates["steel"] +
        materials["bricks_count"] * rates["brick"]
    )

    return round(cost, 2)
