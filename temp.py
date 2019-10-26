import pandas as pd

all_tags = ["no voice", "singer", "duet", "plucking", "hard rock", "world", "bongos", "harpsichord", "female singing",
            "clasical", "sitar", "chorus", "female opera", "male vocal", "vocals", "clarinet", "heavy", "silence",
            "beats", "men", "woodwind", "funky", "no strings", "chimes", "foreign", "no piano", "horns", "classical",
            "female", "no voices", "soft rock", "eerie", "spacey", "jazz", "guitar", "quiet", "no beat", "banjo",
            "electric", "solo", "violins", "folk", "female voice", "wind", "happy", "ambient", "new age", "synth",
            "funk", "no singing", "middle eastern", "trumpet", "percussion", "drum", "airy", "voice", "repetitive",
            "birds", "space", "strings", "bass", "harpsicord", "medieval", "male voice", "girl", "keyboard", "acoustic",
            "loud", "classic", "string", "drums", "electronic", "not classical", "chanting", "no violin", "not rock",
            "no guitar", "organ", "no vocal", "talking", "choral", "weird", "opera", "soprano", "fast",
            "acoustic guitar", "electric guitar", "male singer", "man singing", "classical guitar", "country", "violin",
            "electro", "reggae", "tribal", "dark", "male opera", "no vocals", "irish", "electronica", "horn",
            "operatic", "arabic", "lol", "low", "instrumental", "trance", "chant", "strange", "drone", "synthesizer",
            "heavy metal", "modern", "disco", "bells", "man", "deep", "fast beat", "industrial", "hard", "harp",
            "no flute", "jungle", "pop", "lute", "female vocal", "oboe", "mellow", "orchestral", "viola", "light",
            "echo", "piano", "celtic", "male vocals", "orchestra", "eastern", "old", "flutes", "punk", "spanish", "sad",
            "sax", "slow", "male", "blues", "vocal", "indian", "no singer", "scary", "india", "woman", "woman singing",
            "rock", "dance", "piano solo", "guitars", "no drums", "jazzy", "singing", "cello", "calm", "female vocals",
            "voices", "different", "techno", "clapping", "house", "monks", "flute", "not opera", "not english",
            "oriental", "beat", "upbeat", "soft", "noise", "choir", "female singer", "rap", "metal", "hip hop", "quick",
            "water", "baroque", "women", "fiddle", "english"]

tags = ['choral', 'female voice', 'metal', 'country', 'weird', 'no voice',
        'cello', 'harp', 'beats', 'female vocal', 'male voice', 'dance',
        'new age', 'voice', 'choir', 'classic', 'man', 'solo', 'sitar', 'soft',
        'pop', 'no vocal', 'male vocal', 'woman', 'flute', 'quiet', 'loud',
        'harpsichord', 'no vocals', 'vocals', 'singing', 'male', 'opera',
        'indian', 'female', 'synth', 'vocal', 'violin', 'beat', 'ambient',
        'piano', 'fast', 'rock', 'electronic', 'drums', 'strings', 'techno',
        'slow', 'guitar']

genres = ["hard rock", "classical", "jazz", "electric", "folk", "electronic",
          "country", "electro", "electronica", "heavy metal", "modern", "punk", "blues", "rock", "jazzy",
          "metal", "hip hop", "baroque"]
genre = ['classical',
         'techno',
         'electronic',
         'rock',
         # 'opera',
         'pop',
         # 'classic',
         'country',
         'metal',
         'jazz',
         'modern',
         # 'jazzy',
         'baroque',
         # 'hard rock',
         # 'electric',
         'folk',
         'punk']

df = pd.read_csv('/home/range/Data/MTAT/raw/annotations_final.csv', delimiter='\t')

temp = genre

labels = df[temp].values

counter = 0
for i in labels:
    if i.sum() == 0:
        counter += 1

print(counter)

l = []

num = labels.sum(axis=0)
for i in range(len(temp)):
    l.append((temp[i], num[i]))

l = sorted(l, key=lambda x: x[1])
l = reversed(l)
for i in l:
    print(i)

print(len(labels) - counter)
