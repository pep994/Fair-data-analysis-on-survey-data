"""
Voorbeeld: coalitieakkoord tussen twee partijen
================================================

Scenario
--------
Twee partijen onderhandelen over een coalitieakkoord.
Beide geven per beleidsthema aan:
  - hoe belangrijk (importantie, 0–10) het thema voor hen is
  - welke beleidspositie (0–10) ze wensen

De Fair Compromise Algorithm zoekt een wederkerig compromis:
geen simpel gemiddelde, maar een balans die elk speerpunt eert.

Thema's
-------
1. Woningbouw          – hoeveel sociale woningen (0 = geen, 10 = maximaal)
2. Klimaat             – ambitieniveau klimaatbeleid (0 = laag, 10 = hoog)
3. Veiligheid          – investering in politie/justitie (0 = laag, 10 = hoog)
4. Zorg                – uitbreiding zorgbudget (0 = laag, 10 = hoog)
5. Belastingen         – hoogte toptarief (0 = laag, 10 = hoog)
6. Onderwijs           – extra middelen onderwijs (0 = laag, 10 = hoog)
7. Digitalisering      – overheids-ICT-investering (0 = laag, 10 = hoog)
"""

from fair_compromise import FairCompromiseAlgorithm, Group, Topic

topics = [
    Topic("Woningbouw",     scale=10),
    Topic("Klimaat",        scale=10),
    Topic("Veiligheid",     scale=10),
    Topic("Zorg",           scale=10),
    Topic("Belastingen",    scale=10),
    Topic("Onderwijs",      scale=10),
    Topic("Digitalisering", scale=10),
]

#                      Woning  Klimaat  Veilig  Zorg  Belast  Onderwijs  Digital
links_importantie =   [  7,      9,       3,     8,     7,      6,         5   ]
links_posities    =   [  9,      9,       4,     8,     8,      7,         6   ]

rechts_importantie =  [  5,      3,       9,     5,     6,      5,         6   ]
rechts_posities    =  [  4,      3,       9,     5,     3,      5,         7   ]

groups = [
    Group("Links",  importances=links_importantie,  positions=links_posities),
    Group("Rechts", importances=rechts_importantie, positions=rechts_posities),
]

algo = FairCompromiseAlgorithm(
    topics=topics,
    groups=groups,
    spearpoint_threshold=1.4,   # > 40% boven eigen gemiddelde = speerpunt
    contrast_threshold=1.5,     # speerpunt alleen als ratio met andere groep > 1.5
    spearpoint_bonus=0.6,       # speerpunt-groep krijgt 60% extra gewicht per eenheid surplus
    compression_power=0.55,     # 0.5 = worteltransformatie (sterke compressie)
)

result = algo.run()
result.print_report()


# -----------------------------------------------------------------------
# Tweede voorbeeld: drie groepen (burgerpanel-scenario)
# -----------------------------------------------------------------------

print("\n\n" + "=" * 72)
print("  VOORBEELD 2 – DRIE GROEPEN (BURGERPANEL)")
print("=" * 72 + "\n")

topics2 = [
    Topic("Parkeerbeleid",   scale=10),
    Topic("Groenvoorziening",scale=10),
    Topic("Fietspaden",      scale=10),
    Topic("Winkelgebied",    scale=10),
    Topic("Geluidsnormen",   scale=10),
]

groups2 = [
    Group(
        "Bewoners",
        importances=[6, 8, 5, 4, 9],
        positions=  [3, 9, 7, 5, 8],
    ),
    Group(
        "Ondernemers",
        importances=[8, 3, 4, 9, 4],
        positions=  [8, 4, 5, 9, 4],
    ),
    Group(
        "Fietsers",
        importances=[5, 5, 9, 3, 5],
        positions=  [4, 7, 9, 4, 5],
    ),
]

algo2 = FairCompromiseAlgorithm(topics2, groups2)
result2 = algo2.run()
result2.print_report()
