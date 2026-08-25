# Bu dosyada LSA64 veri setindeki 64 işaretin isimleri duruyor.
# Video adları "sınıf_kişi_tekrar" biçiminde, örnek: 033_001_001.mp4 -> sınıf 33.
# Biz sınıfları 0'dan saydığımız için numaradan 1 çıkarıyoruz: 33 -> LABELS[32] = "Hungry".

LABELS = [
    "Opaque",  # 1
    "Red",  # 2
    "Green",  # 3
    "Yellow",  # 4
    "Bright",  # 5
    "Light-blue",  # 6
    "Colors",  # 7
    "Pink",  # 8
    "Women",  # 9   
    "Enemy",  # 10
    "Son",  # 11
    "Man",  # 12
    "Away",  # 13
    "Drawer",  # 14
    "Born",  # 15
    "Learn",  # 16
    "Call",  # 17
    "Skimmer",  # 18
    "Bitter",  # 19
    "Sweet milk",  # 20
    "Milk",  # 21
    "Water",  # 22
    "Food",  # 23
    "Argentina",  # 24
    "Uruguay",  # 25
    "Country",  # 26
    "Last name",  # 27
    "Where",  # 28
    "Mock",  # 29
    "Birthday",  # 30
    "Breakfast",  # 31
    "Photo",  # 32
    "Hungry",  # 33
    "Map",  # 34
    "Coin",  # 35
    "Music",  # 36
    "Ship",  # 37
    "None",  # 38
    "Name",  # 39
    "Patience",  # 40
    "Perfume",  # 41
    "Deaf",  # 42
    "Trap",  # 43
    "Rice",  # 44
    "Barbecue",  # 45
    "Candy",  # 46
    "Chewing-gum",  # 47
    "Spaghetti",  # 48
    "Yogurt",  # 49
    "Accept",  # 50
    "Thanks",  # 51
    "Shut down",  # 52
    "Appear",  # 53
    "To land",  # 54
    "Catch",  # 55
    "Help",  # 56
    "Dance",  # 57
    "Bathe",  # 58
    "Buy",  # 59
    "Copy",  # 60
    "Run",  # 61
    "Realize",  # 62
    "Give",  # 63
    "Find",  # 64
]

NUM_CLASSES = len(LABELS)


def label(i):
    # sınıf numarasını (0-63) okunabilir bir kelimeye çevirir
    if 0 <= i < NUM_CLASSES:
        return LABELS[i]
    return f"unknown ({i})"
