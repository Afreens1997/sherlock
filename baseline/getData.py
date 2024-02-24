import requests, zipfile, io

r = requests.get("https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip")
print("Requests ok: ", r.ok)
z = zipfile.ZipFile(io.BytesIO(r.content))
print("DOwnload complete")
z.extractall("../../../../data/tir/projects/tir4/users/svarna/Sherlock/data/VG")