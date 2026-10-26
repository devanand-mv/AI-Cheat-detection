import cheatdetection
print("IMPORT_OK")
print("sd available=", bool(getattr(cheatdetection, 'sd', None)))
print("webrtcvad available=", bool(getattr(cheatdetection, 'webrtcvad', None)))
