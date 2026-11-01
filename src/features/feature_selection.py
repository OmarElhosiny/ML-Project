
# src/features/feature_selection.py

# Set A: 19 features
SET_A_FEATURES = [
    "LeadTime",
    "ArrivalDateYear",
    "ArrivalDateMonth",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "Meal",
    "ReservedRoomType",
    "DepositType",
    "Country",
    "MarketSegment",
    "DistributionChannel",
    "Agent",
    "CustomerType",
    "IsRepeatedGuest",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
]

# Set B: Set A + 2 features (21)
SET_B_FEATURES = SET_A_FEATURES + [
    "TotalOfSpecialRequests",
    "RequiredCarParkingSpaces",
]
