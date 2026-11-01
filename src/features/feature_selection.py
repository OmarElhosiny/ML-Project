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
