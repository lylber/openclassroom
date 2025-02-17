data_dictionary = {
    'SK_ID_CURR': 'Identifiant unique pour chaque demande de crédit.',
    'TARGET': 'Indicateur binaire indiquant si le client a eu des difficultés de remboursement (1) ou non (0).',
    'NAME_CONTRACT_TYPE': 'Type de contrat du prêt.',
    'CODE_GENDER': 'Genre du client.',
    'FLAG_OWN_CAR': 'Indique si le client possède une voiture (1) ou non (0).',
    'FLAG_OWN_REALTY': 'Indique si le client possède une propriété immobilière (1) ou non (0).',
    'CNT_CHILDREN': "Nombre d'enfants du client.",
    'AMT_INCOME_TOTAL': 'Revenu total du client.',
    'AMT_CREDIT': 'Montant du crédit demandé par le client.',
    'AMT_ANNUITY': "Montant de l'annuité du crédit.",
    'AMT_GOODS_PRICE': "Prix des biens pour lesquels le crédit est demandé.",
    'NAME_TYPE_SUITE': 'Type de suite de noms pour le client.',
    'NAME_INCOME_TYPE': 'Source de revenu du client.',
    'NAME_EDUCATION_TYPE': "Niveau d'éducation du client.",
    'NAME_FAMILY_STATUS': 'Statut familial du client.',
    'NAME_HOUSING_TYPE': 'Type de logement du client.',
    'REGION_POPULATION_RELATIVE': 'Population relative de la région où vit le client.',
    'DAYS_BIRTH': 'Âge du client en jours négatifs.',
    'DAYS_EMPLOYED': 'Nombre de jours depuis le début de l\'emploi du client (positif si actif, négatif sinon).',
    'DAYS_REGISTRATION': 'Nombre de jours depuis l\'enregistrement de la demande.',
    'DAYS_ID_PUBLISH': 'Nombre de jours depuis la publication de l\'identification du client.',
    'OWN_CAR_AGE': "Âge de la voiture du client.",
    'FLAG_MOBIL': 'Indique si le client a un téléphone mobile (1) ou non (0).',
    'FLAG_EMP_PHONE': 'Indique si le client a un téléphone professionnel (1) ou non (0).',
    'FLAG_WORK_PHONE': 'Indique si le client a un téléphone professionnel pour le travail (1) ou non (0).',
    'FLAG_CONT_MOBILE': 'Indique si le client a un téléphone mobile (1) ou non (0).',
    'FLAG_PHONE': 'Indique si le client a un téléphone (1) ou non (0).',
    'FLAG_EMAIL': 'Indique si le client a fourni une adresse e-mail (1) ou non (0).',
    'OCCUPATION_TYPE': 'Type d\'occupation du client.',
    'CNT_FAM_MEMBERS': 'Nombre de membres de la famille.',
    'REGION_RATING_CLIENT': 'Évaluation du client basée sur la région.',
    'REGION_RATING_CLIENT_W_CITY': 'Évaluation du client basée sur la région avec la ville.',
    'WEEKDAY_APPR_PROCESS_START': 'Jour de la semaine où la demande a été traitée.',
    'HOUR_APPR_PROCESS_START': 'Heure du jour où la demande a été traitée.',
    'REG_REGION_NOT_LIVE_REGION': 'Indique si la région de l\'adresse permanente du client ne correspond pas à la région où il vit.',
    'REG_REGION_NOT_WORK_REGION': 'Indique si la région de l\'adresse permanente du client ne correspond pas à la région où il travaille.',
    'LIVE_REGION_NOT_WORK_REGION': 'Indique si la région où le client vit ne correspond pas à la région où il travaille.',
    'REG_CITY_NOT_LIVE_CITY': 'Indique si la ville de l\'adresse permanente du client ne correspond pas à la ville où il vit.',
    'REG_CITY_NOT_WORK_CITY': 'Indique si la ville de l\'adresse permanente du client ne correspond pas à la ville où il travaille.',
    'LIVE_CITY_NOT_WORK_CITY': 'Indique si la ville où le client vit ne correspond pas à la ville où il travaille.',
    'ORGANIZATION_TYPE': 'Type d\'organisation où le client travaille.',
    'EXT_SOURCE_1': 'Indicateur normalisé externe 1 pour évaluer la fiabilité du client.',
    'EXT_SOURCE_2': 'Indicateur normalisé externe 2 pour évaluer la fiabilité du client.',
    'EXT_SOURCE_3': 'Indicateur normalisé externe 3 pour évaluer la fiabilité du client.',
    'APARTMENTS_AVG': 'Moyenne des caractéristiques relatives au logement (appartements).',
    'BASEMENTAREA_AVG': 'Moyenne des caractéristiques relatives au logement (surface du sous-sol).',
    'YEARS_BEGINEXPLUATATION_AVG': 'Moyenne des caractéristiques relatives au logement (années de début d\'exploitation).',
    'YEARS_BUILD_AVG': 'Moyenne des caractéristiques relatives au logement (années de construction).',
    'COMMONAREA_AVG': 'Moyenne des caractéristiques relatives au logement (surface commune).',
    'ELEVATORS_AVG': 'Moyenne des caractéristiques relatives au logement (ascenseurs).',
    'ENTRANCES_AVG': 'Moyenne des caractéristiques relatives au logement (entrées).',
    'FLOORSMAX_AVG': 'Moyenne des caractéristiques relatives au logement (nombre d\'étages maximum).',
    'FLOORSMIN_AVG': 'Moyenne des caractéristiques relatives au logement (nombre d\'étages minimum).',
    'LANDAREA_AVG': 'Moyenne des caractéristiques relatives au logement (surface du terrain).',
    'LIVINGAPARTMENTS_AVG': 'Moyenne des caractéristiques relatives au logement (appartements habitables).',
    'LIVINGAREA_AVG': 'Moyenne des caractéristiques relatives au logement (surface habitable).',
    'NONLIVINGAPARTMENTS_AVG': 'Moyenne des caractéristiques relatives au logement (appartements non habitables).',
    'NONLIVINGAREA_AVG': 'Moyenne des caractéristiques relatives au logement (surface non habitable).',
    'APARTMENTS_MODE': 'Mode des caractéristiques relatives au logement (appartements).',
    'BASEMENTAREA_MODE': 'Mode des caractéristiques relatives au logement (surface du sous-sol).',
    'YEARS_BEGINEXPLUATATION_MODE': 'Mode des caractéristiques relatives au logement (années de début d\'exploitation).',
    'YEARS_BUILD_MODE': 'Mode des caractéristiques relatives au logement (années de construction).',
    'COMMONAREA_MODE': 'Mode des caractéristiques relatives au logement (surface commune).',
    'ELEVATORS_MODE': 'Mode des caractéristiques relatives au logement (ascenseurs).',
    'ENTRANCES_MODE': 'Mode des caractéristiques relatives au logement (entrées).',
    'FLOORSMAX_MODE': 'Mode des caractéristiques relatives au logement (nombre d\'étages maximum).',
    'FLOORSMIN_MODE': 'Mode des caractéristiques relatives au logement (nombre d\'étages minimum).',
    'LANDAREA_MODE': 'Mode des caractéristiques relatives au logement (surface du terrain).',
    'LIVINGAPARTMENTS_MODE': 'Mode des caractéristiques relatives au logement (appartements habitables).',
    'LIVINGAREA_MODE': 'Mode des caractéristiques relatives au logement (surface habitable).',
    'NONLIVINGAPARTMENTS_MODE': 'Mode des caractéristiques relatives au logement (appartements non habitables).',
    'NONLIVINGAREA_MODE': 'Mode des caractéristiques relatives au logement (surface non habitable).',
    'APARTMENTS_MEDI': 'Médiane des caractéristiques relatives au logement (appartements).',
    'BASEMENTAREA_MEDI': 'Médiane des caractéristiques relatives au logement (surface du sous-sol).',
    'YEARS_BEGINEXPLUATATION_MEDI': 'Médiane des caractéristiques relatives au logement (années de début d\'exploitation).',
    'YEARS_BUILD_MEDI': 'Médiane des caractéristiques relatives au logement (années de construction).',
    'COMMONAREA_MEDI': 'Médiane des caractéristiques relatives au logement (surface commune).',
    'ELEVATORS_MEDI': 'Médiane des caractéristiques relatives au logement (ascenseurs).',
    'ENTRANCES_MEDI': 'Médiane des caractéristiques relatives au logement (entrées).',
    'FLOORSMAX_MEDI': 'Médiane des caractéristiques relatives au logement (nombre d\'étages maximum).',
    'FLOORSMIN_MEDI': 'Médiane des caractéristiques relatives au logement (nombre d\'étages minimum).',
    'LANDAREA_MEDI': 'Médiane des caractéristiques relatives au logement (surface du terrain).',
    'LIVINGAPARTMENTS_MEDI': 'Médiane des caractéristiques relatives au logement (appartements habitables).',
    'LIVINGAREA_MEDI': 'Médiane des caractéristiques relatives au logement (surface habitable).',
    'NONLIVINGAPARTMENTS_MEDI': 'Médiane des caractéristiques relatives au logement (appartements non habitables).',
    'NONLIVINGAREA_MEDI': 'Médiane des caractéristiques relatives au logement (surface non habitable).',
    'FONDKAPREMONT_MODE': 'Mode du fonds du pré-montage.',
    'HOUSETYPE_MODE': 'Mode du type de maison.',
    'TOTALAREA_MODE': 'Mode de la superficie totale.',
    'WALLSMATERIAL_MODE': 'Mode du matériau des murs.',
    'EMERGENCYSTATE_MODE': 'Mode de l\'état d\'urgence.',
    'OBS_30_CNT_SOCIAL_CIRCLE': 'Observations dans le cercle social du client sur une période de 30 jours.',
    'DEF_30_CNT_SOCIAL_CIRCLE': 'Défauts dans le cercle social du client sur une période de 30 jours.',
    'OBS_60_CNT_SOCIAL_CIRCLE': 'Observations dans le cercle social du client sur une période de 60 jours.',
    'DEF_60_CNT_SOCIAL_CIRCLE': 'Défauts dans le cercle social du client sur une période de 60 jours.',
    'DAYS_LAST_PHONE_CHANGE': 'Nombre de jours depuis le dernier changement de numéro de téléphone.',
    'FLAG_DOCUMENT_2': 'Indicateur pour le document 2 fourni.',
    'FLAG_DOCUMENT_3': 'Indicateur pour le document 3 fourni.',
    'FLAG_DOCUMENT_4': 'Indicateur pour le document 4 fourni.',
    'FLAG_DOCUMENT_5': 'Indicateur pour le document 5 fourni.',
    'FLAG_DOCUMENT_6': 'Indicateur pour le document 6 fourni.',
    'FLAG_DOCUMENT_7': 'Indicateur pour le document 7 fourni.',
    'FLAG_DOCUMENT_8': 'Indicateur pour le document 8 fourni.',
    'FLAG_DOCUMENT_9': 'Indicateur pour le document 9 fourni.',
    'FLAG_DOCUMENT_10': 'Indicateur pour le document 10 fourni.',
    'FLAG_DOCUMENT_11': 'Indicateur pour le document 11 fourni.',
    'FLAG_DOCUMENT_12': 'Indicateur pour le document 12 fourni.',
    'FLAG_DOCUMENT_13': 'Indicateur pour le document 13 fourni.',
    'FLAG_DOCUMENT_14': 'Indicateur pour le document 14 fourni.',
    'FLAG_DOCUMENT_15': 'Indicateur pour le document 15 fourni.',
    'FLAG_DOCUMENT_16': 'Indicateur pour le document 16 fourni.',
    'FLAG_DOCUMENT_17': 'Indicateur pour le document 17 fourni.',
    'FLAG_DOCUMENT_18': 'Indicateur pour le document 18 fourni.',
    'FLAG_DOCUMENT_19': 'Indicateur pour le document 19 fourni.',
    'FLAG_DOCUMENT_20': 'Indicateur pour le document 20 fourni.',
    'FLAG_DOCUMENT_21': 'Indicateur pour le document 21 fourni.',
    'AMT_REQ_CREDIT_BUREAU_HOUR': 'Nombre de demandes de crédit auprès du bureau de crédit au cours de la dernière heure.',
    'AMT_REQ_CREDIT_BUREAU_DAY': 'Nombre de demandes de crédit auprès du bureau de crédit au cours de la dernière journée.',
    'AMT_REQ_CREDIT_BUREAU_WEEK': 'Nombre de demandes de crédit auprès du bureau de crédit au cours de la dernière semaine.',
    'AMT_REQ_CREDIT_BUREAU_MON': 'Nombre de demandes de crédit auprès du bureau de crédit au cours du dernier mois.',
    'AMT_REQ_CREDIT_BUREAU_QRT': 'Nombre de demandes de crédit auprès du bureau de crédit au cours du dernier trimestre.',
    'AMT_REQ_CREDIT_BUREAU_YEAR': 'Nombre de demandes de crédit auprès du bureau de crédit au cours de la dernière année.'
}

d_ORGANIZATION_TYPE = {'Kindergarten': 28, 'Self-employed': 42, 'Transport: type 3': 54, 'Business Entity Type 3': 5,
                       'Government': 11, 'Industry: type 9': 26, 'School': 39, 'Trade: type 2': 46, 'XNA': 57,
                       'Services': 43, 'Bank': 2, 'Industry: type 3': 20, 'Other': 33, 'Trade: type 6': 50,
                       'Industry: type 12': 17, 'Trade: type 7': 51, 'Postal': 35, 'Medicine': 30, 'Housing': 13,
                       'Business Entity Type 2': 4, 'Construction': 7, 'Military': 31, 'Industry: type 4': 21,
                       'Trade: type 3': 47, 'Legal Services': 29, 'Security': 40, 'Industry: type 11': 16,
                       'University': 56, 'Business Entity Type 1': 3, 'Agriculture': 1, 'Security Ministries': 41,
                       'Transport: type 2': 53, 'Industry: type 7': 24, 'Transport: type 4': 55, 'Telecom': 44,
                       'Emergency': 10, 'Police': 34, 'Industry: type 1': 14, 'Transport: type 1': 52, 'Electricity': 9,
                       'Industry: type 5': 22, 'Hotel': 12, 'Restaurant': 38, 'Advertising': 0, 'Mobile': 32,
                       'Trade: type 1': 45, 'Industry: type 8': 25, 'Realtor': 36, 'Cleaning': 6, 'Industry: type 2': 19,
                       'Trade: type 4': 48, 'Industry: type 6': 23, 'Culture': 8, 'Insurance': 27, 'Religion': 37,
                       'Industry: type 13': 18, 'Industry: type 10': 15, 'Trade: type 5': 49}

d_NAME_HOUSING_TYPE = {'House / apartment': 1, 'With parents': 5, 'Rented apartment': 4,
                       'Municipal apartment': 2, 'Office apartment': 3, 'Co-op apartment': 0}

d_NAME_FAMILY_STATUS = {'Married': 1, 'Single / not married': 3, 'Civil marriage': 0, 'Widow': 5, 'Separated': 2}

d_FLAG_OWN_REALTY = {'Y': 1, 'N': 0}

d_NAME_EDUCATION_TYPE = {'Higher education': 1, 'Secondary / secondary special': 4, 'Incomplete higher': 2,
                         'Lower secondary': 3, 'Academic degree': 0}

d_NAME_TYPE_SUITE = {'Unaccompanied': 6, 'nan': 1, 'Family': 5, 'Spouse, partner': 2,
                     'Group of people': 4, 'Other_B': 0, 'Children': 3}

d_NAME_CONTRACT_TYPE = {'Cash loans': 0, 'Revolving loans': 1}

d_FLAG_OWN_CAR = {'N': 0, 'Y': 1}

d_NAME_INCOME_TYPE = {'Working': 7, 'State servant': 4, 'Pensioner': 3,
                      'Commercial associate': 1, 'Businessman': 0, 'Student': 5, 'Unemployed': 6}

d_WEEKDAY_APPR_PROCESS_START = {'TUESDAY': 5, 'FRIDAY': 0, 'MONDAY': 1,
                                'WEDNESDAY': 6, 'THURSDAY': 4, 'SATURDAY': 2, 'SUNDAY': 3}

d_CODE_GENDER = {'F': 0, 'M': 1}

colonnes_catégoriques = ['NAME_TYPE_SUITE',
 'FLAG_OWN_CAR',
 'NAME_FAMILY_STATUS',
 'FLAG_OWN_REALTY',
 'NAME_EDUCATION_TYPE',
 'NAME_HOUSING_TYPE',
 'NAME_CONTRACT_TYPE',
 'NAME_INCOME_TYPE',
 'CODE_GENDER']
