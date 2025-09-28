"""
Simple Food Waste Data Sender
Drop this into your teammate's project and they can send data with 2 lines of code
"""

from supabase import create_client


def send_waste_data(data, supabase_url, supabase_key, location="Lab Analysis"):
    """
    Send food waste data to Supabase

    Args:
        data: Dictionary of {food_name: disposal_mass} OR list of dictionaries
        supabase_url: Your Supabase project URL
        supabase_key: Your Supabase anon key
        location: Where the data was collected

    Example:
        # Option 1: Simple dictionary
        waste_results = {
            "Pizza": 15.2,
            "Salad": 8.7,
            "Soup": 12.3
        }
        send_waste_data(waste_results, "https://your-project.supabase.co", "your-key")

        # Option 2: Detailed list
        waste_results = [
            {"food_name": "Pizza", "disposal_mass": 15.2, "location": "Cafeteria A"},
            {"food_name": "Salad", "disposal_mass": 8.7, "location": "Cafeteria B"}
        ]
        send_waste_data(waste_results, "https://your-project.supabase.co", "your-key")
    """

    # Create Supabase client
    supabase = create_client(supabase_url, supabase_key)

    # Convert data to list of records
    if isinstance(data, dict):
        # Convert {food_name: mass} to list of records
        records = [
            {
                "food_name": food_name,
                "disposal_mass": disposal_mass,
                "location": location,
                "session_id": f"analysis_{hash(str(data))}"
            }
            for food_name, disposal_mass in data.items()
        ]
    else:
        # Assume it's already a list of dictionaries
        records = []
        for item in data:
            record = {
                "food_name": item.get("food_name") or item.get("name"),
                "disposal_mass": item.get("disposal_mass") or item.get("mass") or item.get("weight"),
                "location": item.get("location", location),
                "session_id": item.get("session_id", f"analysis_{hash(str(data))}")
            }
            records.append(record)

    # Send to Supabase
    try:
        result = supabase.table('food_disposal_data').insert(records).execute()
        print(f"✅ Successfully sent {len(records)} food waste records to database")
        return result.data
    except Exception as e:
        print(f"❌ Error sending data: {str(e)}")
        raise


# ==========================================
# FOR YOUR TEAMMATE TO USE:
# ==========================================

if __name__ == "__main__":
    # Your teammate's actual Supabase credentials
    SUPABASE_URL = "https://rixsrynryswwrjhektdf.supabase.co"  # Replace this
    SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJpeHNyeW5yeXN3d3JqaGVrdGRmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTg5NjQ1NDksImV4cCI6MjA3NDU0MDU0OX0.oXWRPdWt_m5tQlU7Q9oPzxkSAnC9hMDQKRvk1vNpMCA"  # Replace this

    # Example 1: Your teammate's analysis results (simple dictionary)
    my_analysis_results = {
        "Overcooked Pasta": 23.4,
        "Soggy Vegetables": 15.7,
        "Burnt Toast": 8.2,
        "Cold Soup": 19.1
    }

    # Send to database (just 1 line!)
    send_waste_data(my_analysis_results, SUPABASE_URL, SUPABASE_KEY, location="Research Kitchen")

    # Example 2: More detailed data
    detailed_results = [
        {"food_name": "Pizza Slices", "disposal_mass": 45.2, "location": "Dining Hall A"},
        {"food_name": "Salad Mix", "disposal_mass": 12.8, "location": "Dining Hall B"},
        {"food_name": "Chicken Breast", "disposal_mass": 67.3, "location": "Dining Hall A"}
    ]

    send_waste_data(detailed_results, SUPABASE_URL, SUPABASE_KEY)
