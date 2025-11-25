import json
import random
import os
from datetime import datetime, timedelta

def generate_pii_data(num_examples=800, output_file="data/train.jsonl"):
    """Generate synthetic PII data with noisy STT patterns"""
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    credit_cards = [
        "4242 4242 4242 4242",
        "5555 5555 5555 4444",
        "3782 822463 10005",
        "four two four two 4242 4242 4242",
        "five five five five 5555 5555 4444"
    ]
    
    phones = [
        "9876543210",
        "555 123 4567",
        "plus one 415 555 0123",
        "nine eight seven six five four three two one oh",
        "double nine zero one two three four five"
    ]
    
    emails = [
        "john.doe@gmail.com",
        "ramesh sharma at gmail dot com",
        "contact at example dot org",
        "support@company.co.uk"
    ]
    
    names = [
        "John Smith", "Ramesh Sharma", "Alice Johnson", "Bob Wilson",
        "Priya Patel", "Michael Brown", "Sarah Connor", "David Kumar"
    ]
    
    dates = [
        "December 25, 2023",
        "12/25/2023",
        "25th December twenty twenty three",
        "January 1, 2024"
    ]
    
    cities = ["New York", "Mumbai", "San Francisco", "London", "Toronto", "Delhi"]
    locations = ["California", "Maharashtra", "Texas", "England", "Ontario"]
    
    templates = [
        "my credit card is {cc} and phone is {phone}",
        "call me at {phone} or email {email}",
        "my name is {name} from {city}",
        "contact {name} in {location} on {date}",
        "the order was placed on {date} by {name}",
        "card number {cc} expires on {date}",
        "reached {city} on {date}",
        "{name} can be reached at {phone} or {email}",
        "located in {location}, contact {phone}",
        "date of birth is {date}, name is {name}"
    ]
    
    examples = []
    
    for i in range(num_examples):
        template = random.choice(templates)
        text = template.format(
            cc=random.choice(credit_cards),
            phone=random.choice(phones),
            email=random.choice(emails),
            name=random.choice(names),
            date=random.choice(dates),
            city=random.choice(cities),
            location=random.choice(locations)
        )
        
        entities = []
        
        # Find CC
        for cc in credit_cards:
            if cc in text:
                start = text.find(cc)
                entities.append({
                    "start": start,
                    "end": start + len(cc),
                    "label": "CREDIT_CARD"
                })
                break
        
        # Find phone
        for phone in phones:
            if phone in text:
                start = text.find(phone)
                entities.append({
                    "start": start,
                    "end": start + len(phone),
                    "label": "PHONE"
                })
                break
        
        # Find email
        for email in emails:
            if email in text:
                start = text.find(email)
                entities.append({
                    "start": start,
                    "end": start + len(email),
                    "label": "EMAIL"
                })
                break
        
        # Find name
        for name in names:
            if name in text:
                start = text.find(name)
                entities.append({
                    "start": start,
                    "end": start + len(name),
                    "label": "PERSON_NAME"
                })
                break
        
        # Find date
        for date in dates:
            if date in text:
                start = text.find(date)
                entities.append({
                    "start": start,
                    "end": start + len(date),
                    "label": "DATE"
                })
                break
        
        # Find city
        for city in cities:
            if city in text:
                start = text.find(city)
                entities.append({
                    "start": start,
                    "end": start + len(city),
                    "label": "CITY"
                })
                break
        
        # Find location
        for location in locations:
            if location in text:
                start = text.find(location)
                entities.append({
                    "start": start,
                    "end": start + len(location),
                    "label": "LOCATION"
                })
                break
        
        examples.append({
            "id": f"utt_{i:04d}",
            "text": text,
            "entities": entities
        })
    
    with open(output_file, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"Generated {len(examples)} examples in {output_file}")

if __name__ == "__main__":
    # Generate train, dev, and stress sets
    generate_pii_data(800, "../data/train.jsonl")
    generate_pii_data(150, "../data/dev.jsonl")
    generate_pii_data(100, "../data/stress.jsonl")
    print("Data generation complete!")