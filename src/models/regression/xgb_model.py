from xgboost import XGBRegressor


def create_xgb_regressor():
    return XGBRegressor(
        objective="reg:squarederror",
        random_state=42
    )