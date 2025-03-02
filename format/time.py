import math
import datetime

# humanDuration returns a human-readable approximation of a
# duration (eg. "About a minute", "4 hours ago", etc.).
def humanDuration(d: datetime.timedelta) -> str:
    seconds = int(d.total_seconds())
    if seconds < 1:
        return "Less then a second"
    elif seconds == 1:
        return "1 second"
    elif seconds < 60:
        return f"{seconds} seconds"

    minutes = int(seconds / 60)
    if minutes == 1:
        return "About a minute"
    elif minutes < 60:
        return f"{minutes} minutes"
    
    hours = int(round(seconds / 3600))
    if hours == 1:
        return "About an hour"
    elif hours < 48:
        return f"{hours} hours"
    
    days = d.days
    if days < 7 * 2:
        return f"{days} days"
    elif days < 30 * 2:
        return f"{round(days / 7)} weeks"
    elif days < 365 * 2:
        return f"{round(days / 30)} months"

    return f"{round(days / 365)} years"

def HumanTime(t: str, zeroValue:str) -> str:
	return humanTime(t, zeroValue)

def HumanTimeLower(t: str, zeroValue:str) -> str:
	return humanTime(t, zeroValue).lower()

def humanTime(t :str, zeroValue :str) -> str:
     
	#if t.IsZero() {
	#	return zeroValue
	#}

    d = datetime.datetime.now() - datetime.datetime.fromisoformat(t).replace(tzinfo=None)
    seconds = d.total_seconds()
    if int(seconds / 3600) / 24 / 365 < -20:
        return "Forever"
    elif (seconds < 0):
        return humanDuration(-d) + " from now"
    
    return humanDuration(d) + " ago"

