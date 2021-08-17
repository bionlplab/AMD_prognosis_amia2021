def drusen(drusize: int) -> int:
    """
    DRSZWI: MAXIMUM DRUSEN W/I GRID

        0=None
        1=Quest
        2=<C0 (63)
        3=<C1 (125)
        4=<C2 (250)
        5=>=C2
        8=CG

    Returns:
        0, 1, 2, 88
    """
    drszwi = drusize
    if 0 <= drszwi <= 2:
        return 0
    elif drszwi == 3:
        return 1
    elif 4 <= drszwi <= 5:
        return 2
    elif drszwi == 8 or drszwi == 88:
        return 88
    else:
        raise KeyError('drarwi: %s' % drszwi)


def any_ga(geoawi):
    """
    GEOAWI: GA W/I GRID

        0=None
        1=Quest
        2=<I2
        3=<O2
        4=<1/2 DA
        5=<1DA
        6=<2DA
        7=>2DA
        8=CG

    Returns:
        0, 1, 88
    """
    if geoawi == 0:
        return 0
    elif 1 <= geoawi <= 7:
        return 1
    elif geoawi == 8:
        return 88
    else:
        raise KeyError('geoawi: %s' % geoawi)


def cga(geoact, geoacs, subff2):
    """
    GEOACT: GA CENTER POINT
    GEOACS: GA AREA C/SUB

        0=None
        1=Quest
        2=<I2
        3=<O2
        4=<1/2 DA
        5=<1DA
        6=<2DA
        7=>2DA
        8=CG

    Returns:
        0, 1, 88
    """
    if geoact == 2 or (geoact == 1 and 2 <= geoacs <= 4):
        cga = 1
    elif geoact == 8 or geoacs == 8:
        cga = 88
    else:
        cga = 0
    if cga == 1 and subff2 == 2:
        cga = 0
    return cga


def depigmentation(rpedwi):
    """
    RPEDWI: RPE DEPIGMENTATION AREA W/I GRID

        0=None
        1=Quest
        2=<I2
        3=<O2
        4=<1/2 DA
        5=<1DA
        6=<2DA
        7=>2DA
        8=CG

    Returns:
        0, 1, 88
    """
    if rpedwi == 0:
        return 0
    elif 1 <= rpedwi <= 7:
        return 1
    elif rpedwi == 8:
        return 88
    else:
        raise KeyError('rpedwi: %s' % rpedwi)


def increased_pigment(incpwi):
    """
    INCPWI: INCREASED PIGMENT AREA W/I GRID

    Returns:
        0, 1, 88
    """
    if incpwi == 0 or incpwi == 7:
        return 0
    elif 1 <= incpwi <= 6:
        return 1
    elif incpwi == 8:
        return 88
    else:
        raise KeyError('incpwi: %s' % incpwi)


def pigment(depigm, incpigm, cga, anyga):
    """
    any pigm abn (depig, inc pig, noncentral GA)

    Returns:
        0, 1, 88
    """
    if depigm == 1 or incpigm == 1 or (anyga == 1 and cga == 0):
        return 1
    elif depigm == 0 and incpigm == 0 and anyga == 0:
        return 0
    else:
        return 88


def combine_both_eyes_binary(le, re):
    if le == 1 or re == 1:
        return 1
    elif le == 0 and re == 0:
        return 0
    else:
        return 88


def bilateral_drusen_risk_factor(le_drusen, re_drusen):
    score = 0
    if le_drusen == 2:
        score += 1
    if re_drusen == 2:
        score += 1
    if le_drusen == 1 and re_drusen == 1:
        score += 1
    return score


def simplified_score(le_pigment, le_drusen, le_amd, re_pigment, re_drusen, re_amd):
    """
    If there is no advanced AMD in either eye
        Assign 1 risk factor for each eye with large drusen.
        Assign 1 risk factor for each eye with pigment abnormalities.
        Assign 1 risk factor if neither eye has large drusen and both eyes have intermediate drusen.
    If there is advanced AMD in one eye
        Assign 5 risk factor

    Returns:
        0, 1, 2, 3, 4, 5, 88
    """
    if le_pigment == 88 or le_drusen == 88 or le_amd == 88 \
            or re_pigment == 88 or re_drusen == 88 or re_amd == 88:
        score = 88
    elif le_amd == 1 or re_amd == 1:
        score = 5
    else:
        score = 0
        if le_pigment == 1:
            score += 1
        if re_pigment == 1:
            score += 1
        score += bilateral_drusen_risk_factor(le_drusen, re_drusen)
        assert 0 <= score <= 4
    return score

