import matplotlib.pyplot as plt
from src.preprocessing.pipeline import clean_data

df = clean_data()
print("\nOverall Cancellation Rate:")
print(df["IsCanceled"].mean()*100)
counts = df["IsCanceled"].value_counts()

plt.bar(["Not Canceled", "Canceled"], [counts[0], counts[1]])

plt.title("Booking Cancellation")
plt.xlabel("Booking Status")
plt.ylabel("Number of Bookings")

plt.savefig("outputs/figures/booking_cancellation.png")
plt.show()

deposit_rate = df.groupby("DepositType")["IsCanceled"].mean()*100
print("\nCancellation Rate by Deposit Type:")
print(deposit_rate)

plt.figure()
plt.bar(deposit_rate.index, deposit_rate.values)
plt.title("Cancellation Rate by Deposit Type")
plt.xlabel("Deposit Type")
plt.ylabel("Cancellation Rate (%)")
plt.savefig("outputs/figures/cancellation_by_deposit.png")
plt.show()

market_rate = df.groupby("MarketSegment")["IsCanceled"].mean()*100
print("\nCancellation Rate by Market Segment :")
print(market_rate)

plt.figure(figsize=(9, 5))
plt.barh(market_rate.index, market_rate.values)
plt.title("Cancellation Rate by Market Segment")
plt.xlabel("Cancellation Rate (%)")
plt.ylabel("Market Segment")
plt.tight_layout()
plt.savefig("outputs/figures/cancellation_by_market.png")
plt.show()

month_rate = df.groupby("ArrivalDateMonth")["IsCanceled"].mean() * 100
print("\nCancellation Rate by Month:")
print(month_rate)

months = [
    "January", "February", "March", "April",
    "May", "June", "July", "August",
    "September", "October", "November", "December"
]

month_rate = month_rate.reindex(months)
plt.figure(figsize=(10, 5))
plt.plot(month_rate.index, month_rate.values, marker="o")
plt.title("Cancellation Rate by Month")
plt.xlabel("Arrival Month")
plt.ylabel("Cancellation Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/figures/cancellation_by_month.png")
plt.show()

import pandas as pd

df["LeadTimeGroup"] = pd.cut(
    df["LeadTime"],
    bins=[-1, 30, 90, 180, 365, float("inf")],
    labels=["0-30", "31-90", "91-180", "181-365", "365+"]
)

lead_rate = df.groupby("LeadTimeGroup", observed=True)["IsCanceled"].mean() * 100

print("\nCancellation Rate by Lead Time:")
print(lead_rate)

plt.figure(figsize=(8, 5))
plt.bar(lead_rate.index, lead_rate.values)
plt.title("Cancellation Rate by Lead Time")
plt.xlabel("Lead Time (Days)")
plt.ylabel("Cancellation Rate (%)")
plt.tight_layout()
plt.savefig("outputs/figures/cancellation_by_leadtime.png")
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(df["LeadTime"], bins=30)
plt.title("Lead Time Distribution")
plt.xlabel("Lead Time (Days)")
plt.ylabel("Number of Bookings")
plt.tight_layout()
plt.savefig("outputs/figures/leadtime_distribution.png")
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(df["ADR"], bins=30)
plt.title("ADR Distribution")
plt.xlabel("Average Daily Rate")
plt.ylabel("Number of Bookings")
plt.tight_layout()
plt.savefig("outputs/figures/adr_distribution.png")
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(df["ADR"], bins=30, range=(0, 300))
plt.title("ADR Distribution (0-300)")
plt.xlabel("Average Daily Rate")
plt.ylabel("Number of Bookings")
plt.tight_layout()
plt.savefig("outputs/figures/adr_distribution_zoomed.png")
plt.show()

df["TotalNights"] = df["StaysInWeekendNights"] + df["StaysInWeekNights"]
print("\nTotal Nights Summary:")
print(df["TotalNights"].describe())

plt.figure(figsize=(8, 5))
plt.hist(df["TotalNights"], bins=30)
plt.title("Total Nights Distribution")
plt.xlabel("Total Nights")
plt.ylabel("Number of Bookings")
plt.tight_layout()
plt.savefig("outputs/figures/total_nights_distribution.png")
plt.show()

adr_month = df.groupby("ArrivalDateMonth")["ADR"].mean()
months = [
    "January", "February", "March", "April",
    "May", "June", "July", "August",
    "September", "October", "November", "December"
]
adr_month = adr_month.reindex(months)
print("\nAverage ADR by Month:")
print(adr_month)

plt.figure(figsize=(10, 5))
plt.plot(adr_month.index, adr_month.values, marker="o")
plt.title("Average ADR by Month")
plt.xlabel("Arrival Month")
plt.ylabel("Average Daily Rate")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/figures/adr_by_month.png")
plt.show()

adr_year = df.groupby("ArrivalDateYear")["ADR"].mean().sort_index()
print("\nAverage ADR by Year:")
print(adr_year)

plt.figure(figsize=(8, 5))
plt.bar(adr_year.index.astype(str), adr_year.values)
plt.title("Average ADR by Year")
plt.xlabel("Arrival Year")
plt.ylabel("Average Daily Rate")
plt.tight_layout()
plt.savefig("outputs/figures/adr_by_year.png")
plt.show()

adr_room = df.groupby("ReservedRoomType")["ADR"].mean().sort_values()
print("\nAverage ADR by Room Type:")
print(adr_room)

plt.figure(figsize=(9, 5))
plt.bar(adr_room.index, adr_room.values)
plt.title("Average ADR by Room Type")
plt.xlabel("Reserved Room Type")
plt.ylabel("Average Daily Rate")
plt.tight_layout()
plt.savefig("outputs/figures/adr_by_room_type.png")
plt.show()

df["TotalGuests"] = df["Adults"] + df["Children"] + df["Babies"]
adr_guests = df.groupby("TotalGuests")["ADR"].mean()
print("\nAverage ADR by Guest Count:")
print(adr_guests)

plt.figure(figsize=(9, 5))
plt.bar(adr_guests.index, adr_guests.values)
plt.title("Average ADR by Guest Count")
plt.xlabel("Number of Guests")
plt.ylabel("Average Daily Rate")
plt.tight_layout()
plt.savefig("outputs/figures/adr_by_guest_count.png")
plt.show()