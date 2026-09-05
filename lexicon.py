"""
Leksikon kesehatan bilingual (ID/EN) yang dipakai bersama oleh
query understanding, health context extraction, dan safety layer.

Disimpan terpisah supaya bisa diperluas tanpa menyentuh logika pipeline.
"""

import re

# Gejala umum: kanonik -> daftar sinonim/varian
SYMPTOM_TERMS = {
    "demam": ["demam", "meriang", "panas badan", "badan panas", "fever", "febrile", "hangat badan"],
    "batuk": ["batuk", "cough", "batuk kering", "batuk berdahak"],
    # "influenza" disebut eksplisit: pencocokan istilah pendek memakai batas
    # kata, sehingga "flu" tidak lagi tertangkap sebagai substring di dalamnya.
    "pilek": ["pilek", "flu", "influenza", "hidung tersumbat", "runny nose",
              "nasal congestion", "meler"],
    "sakit kepala": ["sakit kepala", "pusing kepala", "headache", "migrain", "migraine", "nyeri kepala"],
    "pusing": ["pusing", "kliyengan", "dizzy", "dizziness", "vertigo"],
    "mual": ["mual", "nausea", "eneg"],
    "muntah": ["muntah", "vomit", "vomiting"],
    "diare": ["diare", "mencret", "diarrhea", "diarrhoea", "bab cair"],
    "sembelit": ["sembelit", "konstipasi", "constipation", "susah bab",
                "susah buang air besar", "sulit buang air besar", "bab keras"],
    "nyeri perut": ["nyeri perut", "sakit perut", "perut melilit", "abdominal pain", "stomach ache", "kram perut"],
    "nyeri dada": ["nyeri dada", "sakit dada", "chest pain", "dada sesak", "dada nyeri"],
    "sesak napas": ["sesak napas", "sesak nafas", "susah bernapas", "shortness of breath", "dyspnea", "napas pendek"],
    "lelah": ["lelah", "letih", "lemas", "fatigue", "kelelahan", "tidak bertenaga", "capek"],
    "nyeri tenggorokan": ["sakit tenggorokan", "nyeri tenggorokan", "radang tenggorokan", "sore throat"],
    "ruam": ["ruam", "bintik merah", "rash", "gatal kulit", "bentol"],
    "gatal": ["gatal", "itchy", "itching", "pruritus"],
    "nyeri sendi": ["nyeri sendi", "sakit sendi", "joint pain", "pegal linu", "nyeri otot", "myalgia"],
    "sulit tidur": ["sulit tidur", "susah tidur", "insomnia", "tidak bisa tidur"],
    "nafsu makan turun": ["nafsu makan turun", "tidak nafsu makan", "loss of appetite", "malas makan"],
    "menggigil": ["menggigil", "kedinginan", "chills", "shivering"],
    "berkeringat malam": ["keringat malam", "night sweats", "berkeringat malam"],
    "berat badan turun": ["berat badan turun", "bb turun", "weight loss", "kurus mendadak"],
    "bengkak": ["bengkak", "swelling", "edema", "membengkak"],
    "pendarahan": ["pendarahan", "berdarah", "bleeding", "perdarahan"],
    "kejang": ["kejang", "seizure", "convulsion"],
    "pingsan": ["pingsan", "tidak sadarkan diri", "fainting", "syncope", "hilang kesadaran"],
    "jantung berdebar": ["jantung berdebar", "dada berdebar", "palpitasi",
                         "palpitation", "detak jantung cepat", "deg-degan"],
    "mata kuning": ["mata kuning", "jaundice", "ikterik"],
    "nyeri ulu hati": ["nyeri ulu hati", "sakit ulu hati", "perih di ulu hati",
                       "nyeri lambung", "perih lambung", "epigastric pain", "epigastrium"],
    "panas di dada": ["panas di dada", "dada panas", "rasa terbakar di dada",
                      "heartburn", "asam lambung naik", "refluks asam", "acid reflux"],
    "kembung": ["kembung", "begah", "sebah", "perut penuh", "bloating", "distensi"],
    "sendawa": ["sendawa", "belching", "burping"],
    "kesemutan": ["kesemutan", "kebas", "baal", "tingling", "numbness", "parestesia",
                  "paresthesia", "mati rasa"],
    "kram otot": ["kram otot", "kram", "kejang otot", "muscle cramp"],
    "mata berair": ["mata berair", "mata merah", "mata gatal", "watery eyes",
                    "red eyes", "konjungtivitis"],
    "bersin": ["bersin-bersin", "bersin", "sneezing"],
    "wasir": ["wasir", "ambeien", "hemoroid", "hemorrhoid", "benjolan di anus"],
    "kulit kuning": ["badan menguning", "kulit menguning", "badan kuning",
                     "kencing seperti teh", "urin gelap", "air seni gelap"],
    "nyeri pinggang": ["nyeri pinggang", "sakit pinggang", "pegal pinggang",
                       "nyeri punggung bawah", "sakit punggung", "low back pain",
                       "back pain"],
    "gangguan berkemih": ["sulit kencing", "susah kencing", "susah buang air kecil",
                          "nyeri saat kencing", "perih saat kencing", "anyang-anyangan",
                          "sering kencing", "beser", "dysuria", "painful urination",
                          "frequent urination"],
    "rewel": ["rewel", "tidak mau menyusu", "malas menetek", "tidak mau makan minum",
              "irritable", "fussy"],
    "sariawan": ["sariawan", "luka di mulut", "stomatitis", "aphthous", "mouth ulcer"],
    "gusi berdarah": ["gusi berdarah", "bleeding gums", "gusi bengkak"],
    "vertigo": ["kepala berputar", "kepala terasa berputar", "ruangan berputar",
                "spinning sensation"],
    "sulit menelan": ["sulit menelan", "susah menelan", "nyeri menelan", "disfagia",
                      "dysphagia", "odinofagia"],
    "suara serak": ["suara serak", "serak", "hoarseness", "parau"],
    "mimisan": ["mimisan", "hidung berdarah", "epistaksis", "nosebleed"],
    "telinga berdenging": ["telinga berdenging", "tinitus", "tinnitus", "denging di telinga"],
    "penglihatan kabur": ["mata kabur", "penglihatan kabur", "pandangan buram",
                          "blurred vision", "buram"],
    "gemetar": ["gemetar", "tremor", "tangan bergetar", "shaking"],
    "keringat dingin": ["keringat dingin", "berkeringat dingin", "cold sweat", "diaphoresis"],
    "sering haus": ["sering haus", "haus terus", "cepat haus", "polidipsi", "excessive thirst"],
    "luka sulit sembuh": ["luka sulit sembuh", "luka tidak kunjung sembuh",
                          "poor wound healing"],
    "nyeri haid": ["nyeri haid", "kram haid", "dismenore", "dysmenorrhea",
                   "haid tidak teratur", "menstruasi tidak teratur"],
    "keputihan": ["keputihan", "vaginal discharge", "fluor albus"],
    "benjolan": ["benjolan", "bengkak keras", "lump", "massa"],
}

# Kondisi/penyakit yang sering disebut
CONDITION_TERMS = [
    # "gula darah tinggi" wajib berada di daftar ini, bukan hanya "darah
    # tinggi". Pencocokan memilih frasa terpanjang yang menimpa satu rentang
    # teks, sehingga tanpa entri ini pertanyaan tentang gula darah terbaca
    # sebagai hipertensi dan dijawab dengan paper tekanan darah.
    "gula darah tinggi", "hiperglikemia", "gula darah rendah", "hipoglikemia",
    "diabetes", "hipertensi", "darah tinggi", "kolesterol", "asam urat",
    "asma", "tbc", "tuberkulosis", "covid", "covid-19", "dbd", "demam berdarah",
    "tifus", "tipes", "maag", "gerd", "anemia", "kanker", "tumor", "stroke",
    "jantung koroner", "gagal ginjal", "hepatitis", "migrain", "vertigo",
    "alergi", "eksim", "psoriasis", "depresi", "kecemasan", "anxiety",
    "osteoporosis", "arthritis", "rematik", "sinusitis", "bronkitis",
    "pneumonia", "radang paru", "infeksi saluran kemih", "isk", "obesitas",
    "hipotiroid", "hipertiroid", "epilepsi", "demensia", "alzheimer",
]

# Obat & istilah pengobatan
MEDICATION_TERMS = [
    "obat", "medicine", "medication", "drug", "dosis", "dose", "dosage",
    "resep", "prescription", "antibiotik", "antibiotic", "paracetamol",
    "parasetamol", "acetaminophen", "ibuprofen", "aspirin", "amoxicillin",
    "amoksisilin", "metformin", "insulin", "omeprazole", "antasida",
    "antasid", "cetirizine", "loratadine", "dexamethasone", "deksametason",
    "prednison", "vitamin", "suplemen", "supplement", "herbal", "jamu",
    "salep", "sirup obat", "tablet", "kapsul", "injeksi", "suntik",
    "efek samping", "side effect", "interaksi obat", "drug interaction",
    "kontraindikasi", "contraindication", "minum obat", "konsumsi obat",
]

# Istilah kesehatan umum (dipakai untuk menentukan in-scope / out-of-scope)
GENERAL_HEALTH_TERMS = [
    "kesehatan", "health", "medis", "medical", "penyakit", "disease",
    "gejala", "symptom", "diagnosa", "diagnosis", "dokter", "doctor",
    "rumah sakit", "hospital", "klinik", "clinic", "puskesmas", "perawat",
    "terapi", "therapy", "pengobatan", "treatment", "operasi", "surgery",
    "vaksin", "vaccine", "imunisasi", "immunization", "nutrisi", "nutrition",
    "gizi", "diet", "olahraga", "exercise", "tidur", "sleep", "stress",
    # Kosakata cairan tubuh. Tanpa entri ini pertanyaan tentang minum air tidak
    # pernah diperluas ke Bahasa Inggris dan tidak menemukan literatur apa pun.
    "air putih", "minum air", "hidrasi", "hydration", "dehidrasi",
    "dehydration", "cairan tubuh",
    "stres", "mental", "imun", "immune", "infeksi", "infection", "virus",
    "bakteri", "bacteria", "darah", "blood", "jantung", "heart", "paru",
    "lung", "ginjal", "kidney", "hati", "liver", "lambung", "stomach",
    "otak", "brain", "kulit", "skin", "tulang", "bone", "hamil", "pregnant",
    "kehamilan", "pregnancy", "menyusui", "breastfeeding", "bayi", "infant",
    "anak", "lansia", "elderly", "berat badan", "weight", "tekanan darah",
    "blood pressure", "gula darah", "blood sugar", "kolesterol", "cholesterol",
    "merokok", "smoking", "alkohol", "alcohol", "laboratorium", "rontgen",
    "usg", "ct scan", "mri", "cek darah", "medical check up",
]

# Sinyal kegawatdaruratan (§17) — harus memicu safety flag kritis
EMERGENCY_TERMS = [
    "nyeri dada hebat", "nyeri dada berat", "dada terasa ditekan",
    "sesak napas berat", "tidak bisa bernapas", "napas berhenti",
    "tidak sadarkan diri", "pingsan", "tidak merespon", "kejang",
    "kejang-kejang", "stroke", "wajah perot", "bicara pelo", "lumpuh separuh",
    "pendarahan hebat", "muntah darah", "batuk darah", "bab hitam",
    "berak darah", "darah tidak berhenti", "luka menganga",
    "bunuh diri", "mengakhiri hidup", "melukai diri", "self harm", "suicide",
    "overdosis", "keracunan", "minum racun", "tertelan racun",
    "bibir membiru", "kulit membiru", "sianosis",
    "demam tinggi terus menerus", "demam di atas 40",
    "leher kaku demam", "penurunan kesadaran", "syok", "anafilaksis",
    "bengkak wajah dan sesak", "tenggorokan menyempit",
    "chest pain", "difficulty breathing", "unconscious", "seizure",
    "severe bleeding", "coughing blood", "vomiting blood", "poisoning",
]

# Kelompok rentan yang menaikkan tingkat kehati-hatian
HIGH_RISK_POPULATIONS = [
    "hamil", "kehamilan", "pregnant", "menyusui", "breastfeeding",
    "bayi", "newborn", "balita", "infant", "toddler",
    "lansia", "elderly", "manula",
    "imunokompromais", "immunocompromised", "kemoterapi", "chemotherapy",
    "cuci darah", "dialisis", "dialysis", "transplantasi",
]

# Topik jelas di luar cakupan produk
OUT_OF_SCOPE_TERMS = [
    "saham", "kripto", "bitcoin", "trading", "forex", "investasi",
    "pemilu", "politik", "capres", "partai",
    "sepak bola", "liga", "pertandingan", "skor bola",
    "coding", "pemrograman", "javascript", "python", "database",
    "resep masakan enak", "rendang", "cara masak",
    "cuaca", "ramalan bintang", "zodiak", "lirik lagu",
]


def _flatten_symptoms():
    mapping = {}
    for canonical, variants in SYMPTOM_TERMS.items():
        for variant in variants:
            mapping[variant.lower()] = canonical
    return mapping


# variant -> gejala kanonik
SYMPTOM_LOOKUP = _flatten_symptoms()

ALL_SYMPTOM_VARIANTS = sorted(SYMPTOM_LOOKUP.keys(), key=len, reverse=True)

# Pencocokan memakai batas kata, sebab "bersin" ikut cocok di "bersinar".
# Maksimal dua kata sisipan agar "Dada saya terasa panas" tetap terkenali.
_FILLER_WORDS = (
    "saya|aku|ku|nya|anda|kamu|dia|si|"
    "terasa|rasanya|merasa|jadi|juga|masih|sering|agak|sangat|terus|sudah"
)
_FILLER = r"(?:\s+(?:" + _FILLER_WORDS + r")\b){0,2}\s+"
_BOUNDARY_L = r"(?<![a-z0-9])"
_BOUNDARY_R = r"(?![a-z0-9])"


def _build_pattern(variant: str) -> "re.Pattern":
    tokens = [re.escape(token) for token in variant.split()]
    body = _FILLER.join(tokens) if len(tokens) > 1 else tokens[0]
    return re.compile(_BOUNDARY_L + body + _BOUNDARY_R, re.IGNORECASE)


_SYMPTOM_PATTERNS = [(variant, _build_pattern(variant)) for variant in ALL_SYMPTOM_VARIANTS]


def find_symptom_variants(text: str):
    """
    Varian gejala yang benar-benar muncul sebagai kata utuh di `text`.

    Varian diperiksa dari yang terpanjang. Bila sebuah kecocokan menimpa rentang
    teks yang sudah diklaim varian lain, varian yang lebih pendek diabaikan —
    supaya "gusi berdarah" tidak sekaligus terhitung sebagai "berdarah".
    """
    low = (text or "").lower()
    if not low:
        return []

    found = []
    claimed = []  # rentang (start, end) yang sudah dipakai varian lebih panjang

    for variant, pattern in _SYMPTOM_PATTERNS:
        for match in pattern.finditer(low):
            start, end = match.span()
            if any(start < c_end and end > c_start for c_start, c_end in claimed):
                continue
            claimed.append((start, end))
            found.append(variant)
            break

    return found


# ---------------------------------------------------------------------------
# Jembatan bilingual ID -> EN
#
# Knowledge base sebagian besar berbahasa Inggris sedangkan pertanyaan datang
# dalam Bahasa Indonesia. Pemetaannya deterministik, bukan terjemahan LLM, agar
# retrieval tidak bergantung pada ketersediaan API terjemahan.
# ---------------------------------------------------------------------------

CONDITION_TRANSLATIONS = {
    "demam berdarah": ["dengue", "dengue fever", "dengue hemorrhagic fever", "dhf"],
    "dbd": ["dengue", "dengue hemorrhagic fever"],
    "tifus": ["typhoid", "typhoid fever", "enteric fever"],
    "tipes": ["typhoid", "typhoid fever"],
    "tbc": ["tuberculosis", "pulmonary tuberculosis"],
    "tuberkulosis": ["tuberculosis"],
    "hipertensi": ["hypertension", "high blood pressure"],
    "darah tinggi": ["hypertension", "high blood pressure"],
    "diabetes": ["diabetes", "diabetes mellitus", "type 2 diabetes", "hyperglycemia"],
    "kolesterol": ["cholesterol", "hyperlipidemia", "dyslipidemia"],
    "asam urat": ["gout", "hyperuricemia", "uric acid"],
    "asma": ["asthma"],
    "maag": ["dyspepsia", "gastritis"],
    "gerd": ["gastroesophageal reflux", "gerd", "reflux"],
    "anemia": ["anemia", "anaemia", "iron deficiency"],
    "kanker": ["cancer", "carcinoma", "malignancy", "tumor", "neoplasm"],
    "tumor": ["tumor", "neoplasm"],
    "stroke": ["stroke", "cerebrovascular accident"],
    "jantung koroner": ["coronary heart disease", "coronary artery disease", "ischemic heart"],
    "gagal ginjal": ["kidney failure", "renal failure", "chronic kidney disease"],
    "hepatitis": ["hepatitis", "liver inflammation"],
    "migrain": ["migraine"],
    "vertigo": ["vertigo", "dizziness"],
    "alergi": ["allergy", "allergic", "hypersensitivity"],
    "eksim": ["eczema", "atopic dermatitis"],
    "psoriasis": ["psoriasis"],
    "depresi": ["depression", "depressive disorder"],
    "kecemasan": ["anxiety", "anxiety disorder"],
    "osteoporosis": ["osteoporosis", "bone density"],
    "rematik": ["rheumatoid arthritis", "rheumatism"],
    "sinusitis": ["sinusitis"],
    "bronkitis": ["bronchitis"],
    "pneumonia": ["pneumonia"],
    "radang paru": ["pneumonia", "lung inflammation"],
    "infeksi saluran kemih": ["urinary tract infection", "uti"],
    "isk": ["urinary tract infection", "uti"],
    "obesitas": ["obesity", "overweight"],
    "epilepsi": ["epilepsy", "seizure disorder"],
    "demensia": ["dementia"],
    "alzheimer": ["alzheimer", "alzheimer disease"],
    "covid": ["covid-19", "covid", "sars-cov-2", "coronavirus"],
    "covid-19": ["covid-19", "sars-cov-2", "coronavirus"],
    "stunting": ["stunting", "child growth", "malnutrition"],
    "gizi buruk": ["malnutrition", "undernutrition"],
    # gejala yang juga berguna dipetakan ke istilah klinis Inggris
    "nyeri ulu hati": ["epigastric pain", "dyspepsia", "gastritis"],
    "panas di dada": ["heartburn", "gastroesophageal reflux", "acid reflux"],
    "kembung": ["bloating", "abdominal distension", "flatulence"],
    "gangguan berkemih": ["dysuria", "urinary symptoms", "urinary tract infection"],
    "nyeri pinggang": ["low back pain", "lumbar pain"],
    "kulit kuning": ["jaundice", "icterus", "hyperbilirubinemia"],
    "kesemutan": ["paresthesia", "numbness", "neuropathy"],
    "wasir": ["hemorrhoid", "haemorrhoid"],
    "sariawan": ["aphthous ulcer", "stomatitis", "oral ulcer"],
    "telinga berdenging": ["tinnitus"],
    "penglihatan kabur": ["blurred vision", "visual impairment"],
    "nyeri haid": ["dysmenorrhea", "menstrual pain"],
    "keputihan": ["vaginal discharge", "leukorrhea"],
}

GENERAL_TRANSLATIONS = {
    "kesehatan": ["health"],
    "penyakit": ["disease", "illness"],
    "gejala": ["symptom", "symptoms", "clinical features"],
    "diagnosa": ["diagnosis"],
    "diagnosis": ["diagnosis"],
    "pengobatan": ["treatment", "therapy", "management"],
    "terapi": ["therapy", "treatment"],
    "obat": ["drug", "medication", "medicine"],
    "dosis": ["dose", "dosage"],
    "efek samping": ["adverse effect", "side effect", "adverse event"],
    "pencegahan": ["prevention", "prophylaxis"],
    "vaksin": ["vaccine", "vaccination", "immunization"],
    "imunisasi": ["immunization", "vaccination"],
    "anak": ["children", "child", "pediatric", "under-five"],
    "bayi": ["infant", "newborn", "neonate"],
    "lansia": ["elderly", "older adults", "geriatric"],
    "hamil": ["pregnancy", "pregnant", "maternal"],
    "kehamilan": ["pregnancy", "maternal"],
    "menyusui": ["breastfeeding", "lactation"],
    "darah": ["blood"],
    "jantung": ["heart", "cardiac", "cardiovascular"],
    "paru": ["lung", "pulmonary", "respiratory"],
    "ginjal": ["kidney", "renal"],
    "hati": ["liver", "hepatic"],
    "lambung": ["stomach", "gastric"],
    "otak": ["brain", "cerebral"],
    "kulit": ["skin", "cutaneous", "dermatologic"],
    "tulang": ["bone", "skeletal"],
    "sendi": ["joint", "articular"],
    "infeksi": ["infection", "infectious"],
    "virus": ["virus", "viral"],
    "bakteri": ["bacteria", "bacterial"],
    "imun": ["immune", "immunity"],
    "nutrisi": ["nutrition", "nutritional"],
    "gizi": ["nutrition", "nutritional status"],
    "olahraga": ["exercise", "physical activity"],
    "tidur": ["sleep"],
    "merokok": ["smoking", "tobacco", "cigarette"],
    "alkohol": ["alcohol"],
    "berat badan": ["body weight", "weight"],
    "tekanan darah": ["blood pressure"],
    "gula darah": ["blood glucose", "blood sugar"],
    "air putih": ["drinking water", "water intake", "fluid intake"],
    "minum air": ["water intake", "fluid intake", "drinking water"],
    "hidrasi": ["hydration", "fluid balance", "water intake"],
    "dehidrasi": ["dehydration", "fluid loss"],
    "cairan tubuh": ["body fluid", "fluid balance"],
    "gula darah tinggi": ["hyperglycemia", "high blood glucose", "high blood sugar",
                          "elevated blood glucose", "diabetes"],
    "hiperglikemia": ["hyperglycemia", "high blood glucose"],
    "gula darah rendah": ["hypoglycemia", "low blood glucose", "low blood sugar"],
    "hipoglikemia": ["hypoglycemia", "low blood glucose"],
    "pemeriksaan": ["examination", "screening", "assessment"],
    "risiko": ["risk", "risk factor"],
    "komplikasi": ["complication"],
    "penularan": ["transmission", "spread"],
    "menular": ["contagious", "transmissible", "communicable"],
}


# ---------------------------------------------------------------------------
# Penyatuan nama penyakit
#
# "darah tinggi" dan "hipertensi" adalah penyakit yang sama tetapi string yang
# berbeda, dan gerbang fokus judul membuang paper yang justru tepat. Pasangan
# dengan terjemahan identik disimpulkan sendiri; sisanya disebut eksplisit.
_CANONICAL_OVERRIDES = {
    "gula darah tinggi": "diabetes",
    "hiperglikemia": "diabetes",
    # Terjemahannya bertumpuk tetapi tidak identik ("tbc" menambahkan
    # "pulmonary tuberculosis"), sehingga tidak tertangkap penyimpulan otomatis.
    "tbc": "tuberkulosis",
}


def _build_canonical_map():
    by_translation = {}
    for term, translations in CONDITION_TRANSLATIONS.items():
        key = tuple(sorted(t.lower() for t in translations))
        by_translation.setdefault(key, []).append(term)

    canonical = {}
    for terms in by_translation.values():
        if len(terms) < 2:
            continue
        # Sebutan terpanjang dipakai sebagai wakil. Pilihan mana pun konsisten
        # selama seluruh anggota dipetakan ke wakil yang sama, tetapi bentuk
        # yang eksplisit ("infeksi saluran kemih", bukan "isk") membuat log dan
        # pemadanan berjenjang tetap terbaca.
        head = max(terms, key=len)
        for term in terms:
            canonical[term] = head

    canonical.update(_CANONICAL_OVERRIDES)
    return canonical


CONDITION_CANONICAL = _build_canonical_map()


def canonical_condition(term: str) -> str:
    """Satu nama baku untuk penyakit yang punya banyak sebutan."""
    term = (term or "").strip().lower()
    return CONDITION_CANONICAL.get(term, term)


def bilingual_variants(term: str):
    """
    Kembalikan seluruh varian sebuah istilah dalam ID dan EN.

    Urutan hasil: istilah asli lebih dulu, lalu varian lain (tanpa duplikat).
    Istilah yang tidak dikenal dikembalikan apa adanya.
    """
    low = (term or "").strip().lower()
    if not low:
        return []

    variants = [low]

    def extend(values):
        for value in values or []:
            value = value.strip().lower()
            if value and value not in variants:
                variants.append(value)

    # Gejala: SYMPTOM_TERMS sudah memuat varian ID dan EN sekaligus.
    if low in SYMPTOM_TERMS:
        extend(SYMPTOM_TERMS[low])
    else:
        canonical = SYMPTOM_LOOKUP.get(low)
        if canonical:
            extend([canonical])
            extend(SYMPTOM_TERMS.get(canonical, []))

    extend(CONDITION_TRANSLATIONS.get(low))
    extend(GENERAL_TRANSLATIONS.get(low))

    return variants


# ---------------------------------------------------------------------------
# Aspek pertanyaan
#
# Topik saja tidak cukup: "apa gejala demam berdarah" dan "bagaimana mencegah
# demam berdarah" bertopik sama tetapi butuh paper berbeda. Aspek adalah hal
# yang ingin diketahui dari topik itu.
# ---------------------------------------------------------------------------

ASPECT_TERMS = {
    "gejala": [
        "gejala", "tanda", "keluhan", "manifestasi",
        "symptom", "symptoms", "sign", "signs", "clinical features",
        "clinical presentation", "manifestation", "presenting",
    ],
    "penyebab": [
        "penyebab", "sebab", "etiologi", "dipicu", "faktor penyebab",
        "cause", "causes", "caused by", "etiology", "aetiology",
        "pathogenesis", "risk factor", "risk factors", "determinant",
    ],
    "pengobatan": [
        "pengobatan", "penanganan", "terapi", "obat", "menyembuhkan",
        "mengobati", "tatalaksana", "manajemen",
        "treatment", "therapy", "management", "therapeutic",
        "intervention", "regimen", "cure",
    ],
    "pencegahan": [
        "pencegahan", "mencegah", "profilaksis", "vaksin", "imunisasi",
        "prevention", "prevent", "preventive", "prophylaxis",
        "vaccination", "vaccine", "immunization",
    ],
    "penularan": [
        "penularan", "menular", "ditularkan", "penyebaran",
        "transmission", "transmitted", "spread", "contagious",
        "infectious", "incidence",
    ],
    "diagnosis": [
        "diagnosis", "diagnosa", "pemeriksaan", "deteksi", "tes", "skrining",
        "diagnostic", "detection", "screening", "testing", "assay",
        "laboratory", "imaging",
    ],
    "komplikasi": [
        "komplikasi", "dampak", "akibat", "prognosis", "keparahan",
        "complication", "complications", "outcome", "outcomes",
        "severity", "mortality", "morbidity", "sequelae",
    ],
    "keamanan": [
        "aman", "keamanan", "efek samping", "risiko", "bahaya",
        "safety", "safe", "adverse", "side effect", "side effects",
        "toxicity", "contraindication",
    ],
    "dosis": [
        "dosis", "takaran", "aturan pakai", "berapa kali",
        "dose", "dosage", "dosing", "administration",
    ],
    "manfaat": [
        "manfaat", "khasiat", "kegunaan", "efektif", "efektivitas", "membantu",
        "benefit", "benefits", "efficacy", "effective", "effectiveness",
        "sehat", "menyehatkan", "menyembuhkan", "healthy", "improves",
    ],
    "durasi": [
        "berapa lama", "durasi", "lama", "berlangsung",
        "duration", "how long", "length of", "persistence",
    ],
    "epidemiologi": [
        "prevalensi", "angka kejadian", "seberapa umum",
        "prevalence", "epidemiology", "burden", "frequency",
    ],
}


def _flatten_aspects():
    mapping = {}
    for canonical, variants in ASPECT_TERMS.items():
        for variant in variants:
            mapping[variant.lower()] = canonical
    return mapping


ASPECT_LOOKUP = _flatten_aspects()
ALL_ASPECT_VARIANTS = sorted(ASPECT_LOOKUP.keys(), key=len, reverse=True)

_ASPECT_PATTERNS = [
    (variant, _build_pattern(variant)) for variant in ALL_ASPECT_VARIANTS
]


def find_aspects(text: str):
    """Aspek kanonik yang muncul di `text` (pencocokan kata utuh)."""
    low = (text or "").lower()
    if not low:
        return []
    found = []
    for variant, pattern in _ASPECT_PATTERNS:
        canonical = ASPECT_LOOKUP[variant]
        if canonical in found:
            continue
        if pattern.search(low):
            found.append(canonical)
    return found


def aspect_variants(aspect: str):
    """Seluruh varian sebuah aspek kanonik, untuk pencocokan ke dokumen."""
    return ASPECT_TERMS.get((aspect or "").strip().lower(), [])


# ---------------------------------------------------------------------------
# Peta balik EN -> kanonik ID
#
# Judul "Urinary Tract Infection" harus menghasilkan konsep yang sama dengan
# pertanyaan "infeksi saluran kemih", agar tidak dinilai di luar topik.
# ---------------------------------------------------------------------------

def _build_reverse_translations():
    mapping = {}
    for source in (CONDITION_TRANSLATIONS, GENERAL_TRANSLATIONS):
        for canonical, variants in source.items():
            for variant in variants:
                variant = variant.strip().lower()
                # Istilah yang lebih spesifik menang bila terjadi tabrakan.
                if variant and variant not in mapping:
                    mapping[variant] = canonical
    return mapping


REVERSE_TRANSLATIONS = _build_reverse_translations()
ALL_TRANSLATED_VARIANTS = sorted(REVERSE_TRANSLATIONS.keys(), key=len, reverse=True)

_TRANSLATED_PATTERNS = [
    (variant, _build_pattern(variant)) for variant in ALL_TRANSLATED_VARIANTS
]


def find_translated_concepts(text: str):
    """Konsep kanonik (Bahasa Indonesia) yang muncul di teks berbahasa apa pun."""
    low = (text or "").lower()
    if not low:
        return []
    found = []
    for variant, pattern in _TRANSLATED_PATTERNS:
        canonical = REVERSE_TRANSLATIONS[variant]
        if canonical in found:
            continue
        if pattern.search(low):
            found.append(canonical)
    return found
