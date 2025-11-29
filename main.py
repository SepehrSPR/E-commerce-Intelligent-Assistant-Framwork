from Route import process_query

print("\nسیستم فعال است. برای خروج بنویسید 'exit'.")

while True:
    query = input("\nسوال: ").strip()
    if query.lower() in ['exit', 'خروج']:
        print("پایان")
        break

    try:
        answer = process_query(query)
        print("\nپاسخ:\n", answer)
    except Exception as e:
        print("خطا:", e)
        
    
    
    
