"""Fill an empty database with realistic, made-up demo data (a fictional baby, "Ava").

For screenshots, screen recordings and trying the app without real data.
Run inside the app container:  python -m scripts.seed_demo   (add --reset to wipe and re-seed)

All times are relative to now, so "today", "this week" and "next feed" always look current.
"""
import random
import sys
from datetime import datetime, time, timedelta, timezone

from app.models import Base, BabyProfile, Event, SessionLocal, engine
from app.services import baby
from app.services.timeutil import get_tz

TZ = get_tz()
NOW = datetime.now(timezone.utc).replace(tzinfo=None)
AGE_DAYS = 16
rng = random.Random(42)


def local(days_ago: int, hh: int, mm: int = 0) -> datetime:
    """Naive UTC datetime for a local clock time `days_ago` days before today."""
    day = datetime.now(TZ).date() - timedelta(days=days_ago)
    return datetime.combine(day, time(hh, mm), tzinfo=TZ).astimezone(timezone.utc).replace(tzinfo=None)


def add(db, ts, type_, data, source="demo"):
    if ts <= NOW:
        db.add(Event(type=type_, data=data, timestamp=ts, source=source))


def seed(db):
    born = datetime.now(TZ).date() - timedelta(days=AGE_DAYS)
    baby.save_profile(db, {
        "name": "Ava", "birth_date": born.isoformat(), "birth_time": "04:30",
        "birth_weight_g": 3350, "birth_length_cm": 50.5, "birth_head_cm": 34.5,
        "notes": "Demo data. Blood type O+. No known allergies. Born at the city hospital.",
    })

    for days_ago in range(14, -1, -1):
        this_week = days_ago < 7
        # Feeds every 3 hours: breast milk with timer, formula top-ups twice a day.
        for i, hh in enumerate((1, 4, 7, 10, 13, 16, 19, 22)):
            mm = rng.choice((0, 10, 20, 35))
            start = local(days_ago, hh, mm)
            if i in (3, 7):
                ml = rng.choice((60, 75)) if this_week else rng.choice((75, 90))
                data = {"type": "formula", "amount_ml": ml, "formula_name": "Enfamil",
                        "duration_sec": rng.choice((600, 720, 900))}
            else:
                ml = rng.choice((50, 60, 70)) if this_week else rng.choice((60, 70, 80))
                data = {"type": "breast", "amount_ml": ml, "duration_sec": rng.choice((900, 1080, 1200, 1500))}
            data["started"] = start.replace(tzinfo=timezone.utc).isoformat()
            add(db, start + timedelta(seconds=data["duration_sec"]), "feed", data, "demo")
        # Diapers: about 7 wet and 3 dirty a day.
        for hh in (1, 5, 8, 11, 14, 18, 21):
            add(db, local(days_ago, hh, 45), "diaper", {"type": "wet"})
        for hh in (7, 13, 20):
            add(db, local(days_ago, hh, 50), "diaper", {"type": "dirty"})
        # Sleep stretches ending through the day.
        for hh, mins in ((4, 150), (9, 110), (15, 120), (21, 95)):
            end = local(days_ago, hh, 0)
            if end <= NOW:  # log the start and end of each finished sleep, like the Start/End Sleep buttons do
                add(db, end - timedelta(minutes=mins), "sleep", {"action": "start"})
                add(db, end, "sleep", {"action": "end", "duration_sec": mins * 60})

    # Weights: small loss after birth, then a steady 20-30 g/day gain.
    for days_ago, grams in ((13, 3180), (9, 3300), (6, 3380), (3, 3460), (1, 3510)):
        add(db, local(days_ago, 11, 0), "weight", {"grams": grams})

    add(db, local(3, 16, 20), "note", {"text": "Small red rash on her cheeks, not bothering her."})

    add(db, local(2, 10, 30), "visit", {
        "transcript": ("Public health nurse home visit. She's 3460 grams today, so she's past her birth weight "
                       "and gaining well. Keep feeding on demand, about 8 to 12 times a day. The little rash on "
                       "her cheeks is baby acne. It goes away on its own, so no creams. Start tummy time, a few "
                       "minutes a few times a day while she's awake. Keep giving vitamin D drops, 400 IU every day. "
                       "Call Health Link 811 if she has a fever, fewer than 6 wet diapers in a day, or is too "
                       "sleepy to feed. Next visit is her two-month vaccines."),
        "plan": {"summary": "Nurse home visit. Ava is past her birth weight and gaining well; the cheek rash is baby acne.",
                 "weight_g": 3460, "length_cm": None, "head_circumference_cm": None,
                 "feeding_plan_updates": "Feed on demand, about 8 to 12 times a day",
                 "medications": "Vitamin D drops, 400 IU daily",
                 "red_flags": "Fever, fewer than 6 wet diapers a day, too sleepy to feed",
                 "next_visit": "Two-month vaccines"}}, source="scribe")

    db.commit()
    baby.add_immunization(db, "Vitamin K", born.isoformat(), TZ, place="City hospital")
    baby.add_immunization(db, "Erythromycin eye ointment", born.isoformat(), TZ, place="City hospital")
    baby.add_immunization(db, "RSV (Beyfortus / nirsevimab)", (born + timedelta(days=1)).isoformat(), TZ,
                          place="City hospital", notes="Right thigh, no reaction")


def main():
    db = SessionLocal()
    try:
        if "--reset" in sys.argv:
            Base.metadata.drop_all(bind=engine)
            Base.metadata.create_all(bind=engine)
        if db.query(Event).first() or db.get(BabyProfile, 1):
            print("Database already has data; not seeding. Use --reset to wipe it and seed demo data.")
            return
        seed(db)
        print(f"Seeded demo data: {db.query(Event).count()} events for 'Ava' ({AGE_DAYS} days old).")
    finally:
        db.close()


if __name__ == "__main__":
    main()
