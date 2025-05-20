import glob
from collections import Counter

esc50 = [f.split('-')[-1].replace(".wav", "") for f in glob.glob("../audio/*.wav")]
counter = Counter(esc50)
print(counter)