# garbohom 
SET_A_FEATURES = [
    "LeadTime",
    "ArrivalDateYear",
    "ArrivalDateMonth",
    "ArrivalDateWeekNumber",
    "ArrivalDateDayOfMonth",
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

SET_B_FEATURES = SET_A_FEATURES + [
    "TotalOfSpecialRequests",
    "RequiredCarParkingSpaces",
]

SET_C_FEATURES = SET_A_FEATURES + [
    "RequiredCarParkingSpaces",
]

SET_D_FEATURES = SET_A_FEATURES + [
    "TotalOfSpecialRequests",
]

