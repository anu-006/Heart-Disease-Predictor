import pickle
import os
print('numpy version', __import__('numpy').__version__)
try:
    with open('data.pkl','rb') as f:
        d = pickle.load(f)
    print('loaded type', type(d))
except Exception as e:
    print('error', repr(e))

print('pipe exists?', os.path.exists('pipe.pkl'))
