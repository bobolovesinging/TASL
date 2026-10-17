def recent_rounds(rounds,window):
    if rounds < 1 or window < 1:
        raise ValueError('Positive round count and window required')
    return set(range(max(1, rounds-window+1), rounds+1))
