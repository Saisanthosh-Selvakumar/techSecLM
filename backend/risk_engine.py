def calculate_risk(domain, security, compliance):

    score = 0

    score += len(domain) * 2
    score += len(security) * 3
    score += len(compliance) * 2

    return score