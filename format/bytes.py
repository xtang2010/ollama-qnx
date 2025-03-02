import math

Byte = 1

KiloByte = Byte * 1000
MegaByte = KiloByte * 1000
GigaByte = MegaByte * 1000
TeraByte = GigaByte * 1000

KibiByte = Byte * 1024
MebiByte = KibiByte * 1024
GibiByte = MebiByte * 1024

def HumanBytes(b:int) -> str :
	
    if b >= TeraByte:
        value = float(b) / TeraByte
        unit  = "TB"
    elif b >= GigaByte:
        value = float(b) / GigaByte
        unit  = "GB"
    elif b >= MegaByte:
        value = float(b) / MegaByte
        unit  = "MB"
    elif b >= KiloByte:
        value = float(b) / KiloByte
        unit  = "KB"
    else:
        return f"{b} B"

    if value >= 100:
        return f"{int(value)} {unit}"
    elif value >= 10:
        return f"{int(value)} {unit}"
    elif value != math.trunc(value):
        return f"{value:.1f} {unit}"
    else:
        return f"{int(value)} {unit}"

def HumanBytes2(b:int) -> str:
    if b >= GibiByte:
        return f"{float(b) / GibiByte:.1f} GiB"
    elif b >= MebiByte:
        return f"{float(b) / MebiByte:.1f} MiB"
    elif b >= KibiByte:
        return f"{float(b) / KibiByte:.1f} KiB"
    else:
        return f"{b} B"   
