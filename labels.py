# Bu dosyada LSA64 veri setindeki 64 işaretin isimleri duruyor.
# Video adları "sınıf_kişi_tekrar" biçiminde, örnek: 033_001_001.mp4 -> sınıf 33.
# Biz sınıfları 0'dan saydığımız için numaradan 1 çıkarıyoruz: 33 -> LABELS[32] = "Hungry".

LABELS = [
    "Opaque", "Red", "Green", "Yellow", "Bright", "Light-blue", "Colors", "Pink",
    "Women", "Enemy", "Son", "Man", "Away", "Drawer", "Born", "Learn",
    "Call", "Skimmer", "Bitter", "Sweet milk", "Milk", "Water", "Food", "Argentina",
    "Uruguay", "Country", "Last name", "Where", "Mock", "Birthday", "Breakfast", "Photo",
    "Hungry", "Map", "Coin", "Music", "Ship", "None", "Name", "Patience",
    "Perfume", "Deaf", "Trap", "Rice", "Barbecue", "Candy", "Chewing-gum", "Spaghetti",
    "Yogurt", "Accept", "Thanks", "Shut down", "Appear", "To land", "Catch", "Help",
    "Dance", "Bathe", "Buy", "Copy", "Run", "Realize", "Give", "Find",
]

NUM_CLASSES = len(LABELS)


def label(i):
    # sınıf numarasını (0-63) okunabilir bir kelimeye çevirir
    if 0 <= i < NUM_CLASSES:
        return LABELS[i]
    return f"unknown ({i})"
