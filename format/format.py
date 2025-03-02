import math

Thousand = 1000
Million  = Thousand * 1000
Billion  = Million * 1000

def HumanNumber(b: int) -> str: 
	
    if b >= Billion:
        number = float(b) / Billion
        if number == math.floor(number):
            return f"{number:.0f}B"         # no decimals if whole number
        else:
            return f"{number:.1f}B"         # one decimal if not a whole number
    elif b >= Million:
        number = float(b) / Million
        if number == math.floor(number):
            return f"{number:.0f}M"         # no decimals if whole number
        else:
            return f"{number:.2f}M"         # one decimal if not a whole number
    elif b >= Thousand:
        return f"{float(b) / Thousand : .0f}K"
    else:
        return f"{b}"
